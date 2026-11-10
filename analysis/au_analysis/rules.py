"""Rule-based labelling (no model calls): step phases, stuck detection, completion claims, entity tokens."""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

SEARCH_DOMAINS = {"google.com", "bing.com", "duckduckgo.com", "yahoo.com", "search.yahoo.com", "startpage.com"}
READ_ACTIONS = {"click", "scroll", "wait", "go_back", "switch_tab", "navigate", "new_tab", "close_tab", "press", "hover", "select"}
FAIL_RE = re.compile(r"\b(could ?n[o']t|couldn't|unable|cannot|can't|not able|wasn't able|weren't able|failed|no results?|"
                     r"not found|unavailable|not available|blocked|impossible|incomplete|partial(ly)?|did ?n[o']t find|"
                     r"gave up|not possible|was not possible|were not possible|could not be)\b", re.I)
STOP_WORDS = {"the", "then", "after", "please", "start", "first", "once", "when", "could", "also", "that", "this", "with",
              "from", "into", "open", "find", "give", "send", "make", "keep", "leave", "tell", "look", "note", "check", "pull",
              "grab", "head", "jump", "search", "report", "summary", "final", "there", "what", "which", "where", "while", "about",
              "because", "before", "since", "both", "each", "every", "your", "their", "them", "they", "have", "just", "like",
              "some", "more", "most", "only", "need", "want", "would", "should", "actually", "really", "quick", "simple",
              "short", "whole", "same", "other", "another", "here", "home", "page", "pages", "site", "sites", "task", "step"}


def domain_of(url: str) -> str:
    """The registrable domain of a URL ("https://www.rockauto.com/en/catalog" -> "rockauto.com")."""
    if not url:
        return ""
    try:
        host = (urlparse(url).netloc or "").lower().split("@")[-1].split(":")[0]
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov", "ac", "net") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def step_phase(row: dict) -> tuple[str, str]:
    """(category, domain) of one step: done | invalid | search | form | reading | other."""
    action = str(row.get("action") or "")
    dom = domain_of(row.get("url") or "")
    result = str(row.get("result") or "")
    if action == "done":
        return "done", dom
    if action == "invalid" or (row.get("ok") is False and result.startswith("Invalid action")):
        return "invalid", dom
    if dom in SEARCH_DOMAINS:
        return "search", dom
    if action == "type":
        return "form", dom
    if action in READ_ACTIONS:
        return "reading", dom
    return "other", dom


def stuck_flags(steps: list[dict], repeat: int = 3) -> list[bool]:
    """True for steps inside a run of >= `repeat` identical (action, arguments) or >= `repeat` failed actions."""
    n = len(steps)
    flags = [False] * n
    keys = [(str(s.get("action")), json.dumps(s.get("arguments"), sort_keys=True, default=str)) for s in steps]
    oks = [bool(s.get("ok")) for s in steps]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and keys[j + 1] == keys[i]:
            j += 1
        if j - i + 1 >= repeat:
            for x in range(i, j + 1):
                flags[x] = True
        i = j + 1
    i = 0
    while i < n:
        if oks[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not oks[j + 1]:
            j += 1
        if j - i + 1 >= repeat:
            for x in range(i, j + 1):
                flags[x] = True
        i = j + 1
    return flags


def phase_segments(steps: list[dict]) -> list[dict]:
    """Consecutive steps with the same (category, domain) merged into segments; stuck runs are their own
    category. Each segment: start, end (inclusive step numbers), category, domain, label, n."""
    if not steps:
        return []
    stuck = stuck_flags(steps)
    labelled = []
    for s, st in zip(steps, stuck):
        cat, dom = step_phase(s)
        if st and cat not in ("done",):
            cat = "stuck"
        labelled.append((int(s.get("step_num", 0)), cat, dom))
    segs: list[dict] = []
    for step, cat, dom in labelled:
        if segs and segs[-1]["category"] == cat and segs[-1]["domain"] == dom:
            segs[-1]["end"] = step
            segs[-1]["n"] += 1
        else:
            segs.append({"start": step, "end": step, "category": cat, "domain": dom, "n": 1})
    names = {"search": "searching", "reading": "reading", "form": "filling form on", "stuck": "stuck / repeating on",
             "invalid": "invalid action on", "done": "done", "other": "on"}
    for sg in segs:
        sg["label"] = f"{names[sg['category']]} {sg['domain']}".strip() if sg["category"] != "done" else "done"
    return segs


def claim_class(final_row: dict | None, answer: str) -> str:
    """How the rollout ended: 'claims_complete' (done without failure wording), 'claims_partial_or_failure'
    (done, but the final text says something could not be done), or 'no_done'."""
    if not final_row or str(final_row.get("action")) != "done":
        return "no_done"
    text = f"{answer or ''}\n{final_row.get('response') or ''}"
    return "claims_partial_or_failure" if FAIL_RE.search(text) else "claims_complete"


def entity_tokens(text: str) -> set[str]:
    """Capitalised words of 4+ letters (site and product names, places) and dotted hostnames, lower-cased."""
    toks = set()
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9'&-]{3,})\b", text):
        w = m.group(1).lower().strip("'-&")
        if w not in STOP_WORDS and not w.isdigit():
            toks.add(w)
    for m in re.finditer(r"\b([a-z0-9-]+\.(?:com|org|net|gov|co\.uk|io|edu))\b", text.lower()):
        toks.add(m.group(1))
    return toks


def site_names(text: str) -> set[str]:
    """Names that look like websites or brands in a task text, lower-cased (a subset of entity_tokens)."""
    return {t for t in entity_tokens(text) if "." in t or t[0].isalpha()}


# wording the agent uses (thought or action result) when a site refuses it: used for the bot-blocking rate per rollout
BLOCK_RE = re.compile(r"cloudflare|verify you are human|access denied|captcha|bot detection|are you a robot|blocked by|403 forbidden"
                      r"|unusual traffic|press and hold|human verification", re.I)
