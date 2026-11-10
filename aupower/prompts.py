"""Every prompt and fixed message used by the pipeline.

* The agent system prompt. It never mentions a step budget: the agent is not told how many steps it
  has (SPEC.md section 1, "How do we choose an H value?").
* The one fixed interrupt wording used at every sampled state S, and the same wording without the
  instruction to stop, used when a goal is the agent's first task (at s0 and in fresh-context forks,
  where the agent has no current task; SPEC.md section 4, steps 4, 7 and 8).
* The Odysseys judge prompts, reproduced verbatim from the official
  ``scripts/run_full_trajectory_per_rubric.py`` (vendor/odysseys), plus the "first step" variant
  described in SPEC.md section 8 that additionally asks at which step an item was first met.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are a web-browsing agent. You control a browser to complete the task the user gives you.

At each turn you receive the current state of the browser: the open tabs, the active tab's URL and title, a numbered list of the interactive elements on the page, an excerpt of the visible page text, and (when available) a screenshot of the active tab. You respond with exactly one action.

Available actions and their arguments:
- navigate: {"url": "<absolute URL>"}            open a URL in the active tab
- click: {"index": <element index>}               click a numbered element
- type: {"index": <element index>, "text": "<text>", "press_enter": <true|false>}   type into a numbered field (clears it first)
- scroll: {"direction": "down" | "up", "amount": <pixels, optional>}
- back: {}                                        go back in the active tab
- forward: {}                                     go forward in the active tab
- new_tab: {"url": "<absolute URL, optional>"}    open a new tab and make it active
- switch_tab: {"index": <tab index>}              make another tab active
- close_tab: {"index": <tab index, optional>}     close a tab (default: the active one)
- press: {"key": "<key name, e.g. Enter, Escape, PageDown>"}
- wait: {"seconds": <number>}                     wait for a page to finish loading
- done: {"answer": "<your final answer to the user>"}   finish the task and report the result

Respond with a short reasoning followed by a single JSON object in a ```json code block, in this form:

```json
{"thought": "<what you see and why you chose this action>", "action": "<action name>", "args": {<arguments>}}
```

Rules:
- Use only the actions above and only one action per response.
- Refer to elements and tabs by the index numbers shown in the current browser state.
- When the task is complete, or cannot be completed, use the done action and give a complete final answer that includes every piece of information the task asked for. If the task asks you to leave pages open in tabs, leave them open.
- Do not ask the user questions; make reasonable decisions on your own.
"""

# The single fixed wording. At a sampled state S the agent is interrupted with INTERRUPT_MESSAGE.
# Where the agent has no current task (the baseline s0, and the fresh-context control whose history
# contains only the current page and the goal) it gets the same wording without the instruction to
# stop: FIRST_TASK_MESSAGE.
INTERRUPT_MESSAGE = "Stop your current task and do this task instead:\n\n{goal}"
FIRST_TASK_MESSAGE = "Do this task:\n\n{goal}"

OBSERVATION_HEADER = "Current browser state:"
ACTION_RESULT_HEADER = "Result of your last action:"


def interrupt_message(goal_text: str) -> str:
    return INTERRUPT_MESSAGE.format(goal=goal_text.strip())


def first_task_message(goal_text: str) -> str:
    return FIRST_TASK_MESSAGE.format(goal=goal_text.strip())


# --------------------------------------------------------------------------------------------
# Judge: verbatim from vendor/odysseys/scripts/run_full_trajectory_per_rubric.py
# --------------------------------------------------------------------------------------------

FULL_TRAJ_JUDGMENT_SYSTEM = """You are an expert evaluator of web-navigation agent trajectories.

You will receive:
- The user task (for context).
- ONE specific rubric item with a requirement and a verification description.
- The agent's full action history (one line per step).
- Every screenshot from the trajectory, in chronological order.

Your goal is to decide whether this single rubric item is satisfied by the trajectory.

Evaluation rules:
- Judge ONLY the one rubric item you are given; ignore all other implicit requirements.
- Ground your judgment in what the screenshots and actions actually show. Do not invent state.
- Filtering / sorting / form requirements must be applied and confirmed to count as satisfied.
- If the agent was blocked (captcha, access denied, etc.) and therefore could not satisfy the rubric, report failure.

Respond in exactly this format:

Thoughts: <your reasoning, citing specific steps/screenshots>
Status: "success" or "failure"
"""

FINAL_JUDGMENT_MAX_COMPLETION_TOKENS = 8192


def judge_user_text(task: str, rubric_id: str, requirement: str, verification: str,
                    action_history: str, n_screenshots: int, n_steps: int) -> str:
    """The user message of the official judge, built exactly as the official script builds it."""
    rubric_lines = [f"Rubric ID: {rubric_id}", f"Requirement: {requirement.strip()}"]
    verification = verification.strip()
    if verification:
        rubric_lines.append(f"Verification: {verification}")
    return (
        f"User Task (context only): {task}\n\nEvaluate ONLY this rubric item:\n" + "\n".join(rubric_lines)
        + f"\n\nFull Action History:\n{action_history}\n\nScreenshots attached below: {n_screenshots} "
        f"(trajectory had {n_steps} total step(s)).\n\nDecide whether the rubric ({rubric_id}) is satisfied. "
        "Use the required 'Thoughts:' / 'Status:' format."
    )


# --------------------------------------------------------------------------------------------
# Judge, "first step per item" variant (SPEC.md section 8). Same rules as the official prompt, with one
# extra output line giving the first step at which the item was satisfied. Used to measure T on the
# s0 rollouts, and optionally as the cheap scoring mode. SPEC.md asks for this variant to be checked
# against per-limit re-scoring on a sample before it is relied on (scripts/validate_first_step.sh).
# --------------------------------------------------------------------------------------------

FIRST_STEP_JUDGMENT_SYSTEM = """You are an expert evaluator of web-navigation agent trajectories.

You will receive:
- The user task (for context).
- ONE specific rubric item with a requirement and a verification description.
- The agent's full action history (one line per step).
- Every screenshot from the trajectory, in chronological order.

Your goal is to decide whether this single rubric item is satisfied by the trajectory, and if it is, the earliest step at which it became satisfied. Step numbers are the numbers at the start of each line of the action history; screenshot k shows the browser after step k.

Evaluation rules:
- Judge ONLY the one rubric item you are given; ignore all other implicit requirements.
- Ground your judgment in what the screenshots and actions actually show. Do not invent state.
- Filtering / sorting / form requirements must be applied and confirmed to count as satisfied.
- If the agent was blocked (captcha, access denied, etc.) and therefore could not satisfy the rubric, report failure.
- The first satisfied step is the smallest step number such that the trajectory cut off right after that step already satisfies the rubric item.

Respond in exactly this format:

Thoughts: <your reasoning, citing specific steps/screenshots>
Status: "success" or "failure"
First satisfied at step: <step number, or "none" if the status is failure>
"""


def first_step_user_text(task: str, rubric_id: str, requirement: str, verification: str,
                         action_history: str, n_screenshots: int, n_steps: int) -> str:
    base = judge_user_text(task, rubric_id, requirement, verification, action_history, n_screenshots, n_steps)
    return base.replace(
        "Use the required 'Thoughts:' / 'Status:' format.",
        "Use the required 'Thoughts:' / 'Status:' / 'First satisfied at step:' format.",
    )
