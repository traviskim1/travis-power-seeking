"""Launch attempts of a run and rollout directories archived next to it.

Sources: the console log(s) written by the launch commands (``runs/<name>*.console.log``: each launch appends a header
"===== attempt N (...): <UTC time> =====" (the first launch may have none), the pipeline's own log lines (local time)
and "exit code: N" when the command returns), and directories named ``runs/<name>_*``, which hold rollouts moved out
of the run by hand. A launch segment without an exit code was stopped from outside (or is still running); a rollout
with steps but no rollout.json never finished its teardown.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

HEADER_RE = re.compile(r"^===== attempt (\d+)(?: \((.*?)\))?: (.+?) =====\s*$", re.M)
LOG_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} (\w+) ([\w.]+): (.*)$")
STAGE_RE = re.compile(r"=== stage: (\w+) ===")
EXIT_RE = re.compile(r"^exit code: (\d+)\s*$", re.M)
EXIT_MEANING = {0: "finished", 1: "aborted at the confirmation", 3: "error", 4: "budget stop", 5: "routing problem, nothing spent",
                130: "interrupted by hand (SIGINT)", 137: "killed (SIGKILL)", 143: "terminated (SIGTERM)"}
STALL_MIN = 30.0     # a launch idle this long before being stopped from outside counts as stalled
TB_RE = re.compile(r"^Traceback \(most recent call last\):\n((?:[ \t].*\n|\n)*?)(\S.*)$", re.M)
FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')


def tracebacks(body: str) -> list[dict]:
    """Every traceback printed in a launch segment: the exception line and the innermost frame of this project (if any)."""
    out = []
    for m in TB_RE.finditer(body):
        frames = FRAME_RE.findall(m.group(1))
        own = [f for f in frames if "/site-packages/" not in f[0] and "/lib/python" not in f[0]]
        where = (own or frames or [("", "", "")])[-1]
        out.append({"exception": m.group(2).strip()[:200], "file": where[0], "line": where[1], "function": where[2],
                    "innermost": f"{frames[-1][0].split('/')[-1]}:{frames[-1][1]} {frames[-1][2]}" if frames else ""})
    return out


def _header_time(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%d %H:%M:%S UTC"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _local_to_utc(s: str, offset: timedelta) -> datetime:
    return (datetime.strptime(s, "%Y-%m-%d %H:%M:%S") - offset).replace(tzinfo=timezone.utc)


def _fmt(t: datetime | None, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    return t.strftime(fmt) if t else ""


def launch_attempts(run) -> dict:
    """One row per launch segment of every console log next to the run directory."""
    offset = datetime.now().astimezone().utcoffset() or timedelta(0)   # the pipeline logs local time; headers are UTC
    rows: list[dict] = []
    problems: list[str] = []
    notes: list[str] = []
    for p in run.console_logs():
        text = p.read_text(encoding="utf-8", errors="replace")
        heads = list(HEADER_RE.finditer(text))
        segs: list[tuple[str, str, datetime | None, str]] = []
        if heads:
            pre = text[:heads[0].start()]
            if pre.strip():
                n0 = int(heads[0].group(1)) - 1
                segs.append((str(n0) if n0 >= 1 else "unnumbered", "", None, pre))
            for i, h in enumerate(heads):
                end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
                segs.append((h.group(1), h.group(2) or "", _header_time(h.group(3)), text[h.end():end]))
        else:
            segs.append(("1", "", None, text))
        parsed = []
        for num, label, started, body in segs:
            logs = [m for m in (LOG_RE.match(l) for l in body.splitlines()) if m]
            parsed.append((num, label, started, body, logs))
        for i, (num, label, started, body, logs) in enumerate(parsed):
            stages = STAGE_RE.findall(body)
            exit_m = EXIT_RE.search(body)
            code = int(exit_m.group(1)) if exit_m else None
            first_log = _local_to_utc(logs[0].group(1), offset) if logs else None
            last_log = _local_to_utc(logs[-1].group(1), offset) if logs else None
            next_start = None
            if i + 1 < len(parsed):
                nxt = parsed[i + 1]
                next_start = nxt[2] or (_local_to_utc(nxt[4][0].group(1), offset) if nxt[4] else None)
            ref = last_log or started
            gap_min = round((next_start - ref).total_seconds() / 60, 1) if (next_start and ref) else None
            tbs = tracebacks(body)
            n_tb = max(body.count("Traceback (most recent call last)"), len(tbs))
            tb_kinds = {}
            for t in tbs:
                key = (re.sub(r"position \d+", "position N", t["exception"]), t["file"].split("/")[-1], t["line"], t["function"], t["innermost"])
                tb_kinds[key] = tb_kinds.get(key, 0) + 1
            tb_desc = "; ".join(f"{n} x {k[0]} (raised at {k[1]}:{k[2]} in {k[3]}, innermost {k[4]})" if k[1] else f"{n} x {k[0]}"
                                for k, n in sorted(tb_kinds.items(), key=lambda kv: -kv[1]))
            errors = [m.group(4)[:160] for m in logs if m.group(2) in ("ERROR", "CRITICAL")]
            routing = [l[:160] for l in body.splitlines() if l.startswith("ROUTING:")]
            last_seg = i + 1 == len(parsed)
            if code is None:
                ended = "no exit code: still running, or stopped from outside" if last_seg else "no exit code: stopped from outside (killed)"
            else:
                ended = f"exit code {code} ({EXIT_MEANING.get(code, 'unknown meaning')})"
            stalled = code is None and gap_min is not None and gap_min >= STALL_MIN
            rows.append({"console_log": p.name, "attempt": num, "label": label, "started_utc": _fmt(started, "%Y-%m-%d %H:%M UTC") or (_fmt(first_log) + " (first log line)" if first_log else ""),
                         "last_log_line_utc": _fmt(last_log), "last_stage": stages[-1] if stages else "", "stages": " > ".join(dict.fromkeys(stages)),
                         "ended": ended, "exit_code": code, "tracebacks": n_tb, "error_lines": len(errors), "routing_lines": len(routing),
                         "first_error": (errors or routing or [""])[0], "traceback_kinds": tb_desc, "minutes_idle_before_next_launch": gap_min, "stalled": stalled})
            if stalled:
                problems.append(f"launch attempt {num} ({p.name}) wrote its last log line at {_fmt(last_log)} in stage '{stages[-1] if stages else '?'}' "
                                f"and was stopped from outside {gap_min:.0f} min later with no exit code: the pipeline was stalled (tables/health_attempts.csv)")
            elif code in (3, 4) or (code is not None and code not in EXIT_MEANING):
                problems.append(f"launch attempt {num} ({p.name}) ended with {ended}; last stage '{stages[-1] if stages else '?'}' (tables/health_attempts.csv)")
            elif n_tb and code not in (5, 130):
                problems.append(f"launch attempt {num} ({p.name}) printed {n_tb} traceback(s): {tb_desc or 'see the console log'} (tables/health_attempts.csv)")
    if rows:
        parts = []
        for r in rows:
            if r["stalled"]:
                parts.append(f"attempt {r['attempt']} stalled {r['minutes_idle_before_next_launch']:.0f} min in stage '{r['last_stage']}' and was stopped from outside")
            else:
                parts.append(f"attempt {r['attempt']}: {r['ended']}" + (f", last stage '{r['last_stage']}'" if r["last_stage"] else ""))
        summary = f"{len(rows)} launches; " + "; ".join(parts)
        notes.append("launches (console log): " + summary)
    else:
        summary = ""
    return {"rows": rows, "problems": problems, "notes": notes, "summary": summary}


def archived_rollouts(run) -> dict:
    """Rollout directories under runs/<name>_* (archives moved out of the run by hand), with teardown evidence."""
    rows: list[dict] = []
    problems: list[str] = []
    notes: list[str] = []
    parent = run.run_dir.parent
    for d in sorted(parent.glob(run.run_dir.name + "_*")):
        if not d.is_dir():
            continue
        n_with, n_without = 0, 0
        for sp in sorted(d.rglob("steps.jsonl")):
            rd = sp.parent
            steps = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
            last = steps[-1] if steps else {}
            after = None
            rec = rd / "server_recording.jsonl"
            if rec.exists() and last.get("ts"):
                after = 0
                with open(rec, encoding="utf-8") as f:
                    for l in f:
                        if l.strip():
                            try:
                                if float(json.loads(l).get("t", 0)) > float(last["ts"]):
                                    after += 1
                            except (json.JSONDecodeError, TypeError, ValueError):
                                pass
            rel = rd.relative_to(parent)
            kind = next((x for x in rel.parts if x in ("baseline", "traj", "forks")), "")
            has_result = (rd / "rollout.json").exists()
            n_with += has_result
            n_without += not has_result
            rows.append({"archive": d.name, "rollout_dir": str(rel), "kind": kind, "n_steps": len(steps), "last_action": last.get("action", ""),
                         "final": bool(last.get("final")), "has_rollout_json": has_result, "has_judge": (rd / "judge").exists(),
                         "recording_entries_after_last_step": after,
                         "last_step_utc": _fmt(datetime.fromtimestamp(float(last["ts"]), timezone.utc)) if last.get("ts") else ""})
            if steps and not has_result:
                msg = (f"archived rollout {rel}: {len(steps)} steps ending in action '{last.get('action')}' (final={bool(last.get('final'))}) "
                       f"but no rollout.json, so its teardown never finished")
                if after:
                    msg += f"; its server_recording.jsonl holds {after} responses recorded after the final step, i.e. the browser context stayed open"
                problems.append(msg + " (tables/health_archives.csv)")
        notes.append(f"archive {d.name}: {n_with} rollouts with rollout.json and {n_without} without; nothing in it is used by this analysis")
    return {"rows": rows, "problems": problems, "notes": notes}
