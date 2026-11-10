from aupower.tasks import derive_seed, sample_aux_goals, sample_trajectory_tasks

import pytest


def test_vendor_tasks_load(tasks):
    assert len(tasks) == 200
    assert len(tasks.pool("easy")) == 45
    assert len(tasks.pool("medium")) == 46
    assert len(tasks.pool("hard")) == 109
    for t in tasks.tasks:
        assert 3 <= t.m <= 12
        assert len(t.task_id) == 40
        assert t.website.startswith("http")
        for r in t.rubrics:
            assert r.requirement and r.verification
    # the released data carries no rubric weights; they are recorded as None
    assert all(w is None for t in tasks.tasks for w in t.weights)


def test_goal_sampling_is_seeded_and_excludes_the_trajectory_task(tasks):
    pool = tasks.pool("easy")
    g = pool[0].task_id
    seed = derive_seed(20261105, "goals", "easy", g)
    goals = sample_aux_goals(pool, g, 30, seed)
    assert len(goals) == 30 and len(set(goals)) == 30
    assert g not in goals
    assert all(tasks[y].level == "easy" for y in goals)
    assert sample_aux_goals(pool, g, 30, seed) == goals                    # same seed, same draw
    assert sample_aux_goals(pool, g, 30, seed + 1) != goals                # different seed, different draw
    assert sample_aux_goals(list(reversed(pool)), g, 30, seed) == goals    # independent of input order


def test_goal_sampling_pool_limits(tasks):
    pool = tasks.pool("easy")
    g = pool[0].task_id
    assert len(sample_aux_goals(pool, g, 44, 1)) == 44
    with pytest.raises(ValueError):
        sample_aux_goals(pool, g, 45, 1)


def test_trajectory_sampling(tasks):
    pool = tasks.pool("easy")
    ids = sample_trajectory_tasks(pool, 5, 123)
    assert len(ids) == 5 and len(set(ids)) == 5
    assert ids == sample_trajectory_tasks(pool, 5, 123)


def test_derive_seed_is_stable():
    assert derive_seed(1, "a", "b") == derive_seed(1, "a", "b")
    assert derive_seed(1, "a", "b") != derive_seed(2, "a", "b")
    assert derive_seed(1, "a", "b") != derive_seed(1, "b", "a")
