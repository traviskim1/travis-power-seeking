# Source

Copied unmodified from the official Odysseys repository (MIT license).

- Repository: https://github.com/ljang0/Odysseys
- Commit: 837814633ef948479abb3d142458f0acdb73fa65
- Retrieved: 2026-11-05
- Files: data/odysseys.json (200 tasks with rubrics), scripts/run_full_trajectory_per_rubric.py
  (the official per-rubric judge), scripts/convert_odysseys_to_osworld.py (task converter), README.md, LICENSE.

The pipeline in this project reads the tasks from data/odysseys.json and reproduces the judge
prompt from scripts/run_full_trajectory_per_rubric.py verbatim (see aupower/prompts.py). Rollouts
are written in the OSWorld layout (steps.jsonl + screenshots + result.txt) that the official judge
script expects, so it can be run on them directly as a cross-check.
