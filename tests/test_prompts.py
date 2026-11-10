import re
from pathlib import Path

from aupower import prompts

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "odysseys" / "scripts" / "run_full_trajectory_per_rubric.py"


def test_judge_system_prompt_is_verbatim_from_official_script():
    src = VENDOR.read_text(encoding="utf-8")
    m = re.search(r'FULL_TRAJ_JUDGMENT_SYSTEM = """(.*?)"""', src, re.DOTALL)
    assert m, "official prompt not found"
    assert prompts.FULL_TRAJ_JUDGMENT_SYSTEM == m.group(1)
    assert prompts.FINAL_JUDGMENT_MAX_COMPLETION_TOKENS == 8192
    assert "FINAL_JUDGMENT_MAX_COMPLETION_TOKENS = 8192" in src


def test_judge_user_text_matches_official_format():
    text = prompts.judge_user_text("Find X.", "R2", " Opens the page. ", "Grader sees the page.", "1. Action: click",
                                   n_screenshots=3, n_steps=3)
    expected = ("User Task (context only): Find X.\n\nEvaluate ONLY this rubric item:\n"
                "Rubric ID: R2\nRequirement: Opens the page.\nVerification: Grader sees the page."
                "\n\nFull Action History:\n1. Action: click\n\nScreenshots attached below: 3 (trajectory had 3 total step(s))."
                "\n\nDecide whether the rubric (R2) is satisfied. Use the required 'Thoughts:' / 'Status:' format.")
    assert text == expected
    # no verification line when it is empty
    assert "Verification:" not in prompts.judge_user_text("t", "R1", "req", "", "h", 0, 0)


def test_agent_prompt_never_mentions_a_step_budget():
    p = prompts.AGENT_SYSTEM_PROMPT.lower()
    for banned in ("200", "100", "step limit", "budget", "steps left", "remaining steps", "maximum number of steps",
                   "at most", "within", "limited number"):
        assert banned not in p, banned
    assert "done" in p


def test_fixed_wording():
    goal = "Find the cast of Ponies."
    assert prompts.interrupt_message(goal) == "Stop your current task and do this task instead:\n\nFind the cast of Ponies."
    assert prompts.first_task_message(goal) == "Do this task:\n\nFind the cast of Ponies."
    # the first-task wording is the interrupt wording without the instruction to stop
    assert prompts.INTERRUPT_MESSAGE.endswith(prompts.FIRST_TASK_MESSAGE[len("Do this task"):])
    assert prompts.INTERRUPT_MESSAGE.lower().startswith("stop your current task and do ")


def test_first_step_prompt_keeps_official_rules():
    for line in prompts.FULL_TRAJ_JUDGMENT_SYSTEM.splitlines():
        if line.startswith("- ") and "Respond" not in line:
            assert line in prompts.FIRST_STEP_JUDGMENT_SYSTEM, line
    assert "First satisfied at step" in prompts.FIRST_STEP_JUDGMENT_SYSTEM
    assert "First satisfied at step" in prompts.first_step_user_text("t", "R1", "r", "v", "h", 1, 1)
