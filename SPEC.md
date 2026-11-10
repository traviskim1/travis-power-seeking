> **Handoff note.** This is the Markdown version of the artifact "Attainable Utility in Odysseys" (Travis Kim), exported November 5, 2026. Math is in LaTeX (`$...$` inline, `$$...$$` display). Citations are bracketed numbers that refer to the References list at the end.

# Measuring Attainable Utility for Suboptimal Policies in Odysseys

An outline for transferring Turner's concepts of attainable utility\[1,2\] with suboptimal policies\[3\] in the environment Odysseys\[4,5\]. By sampling states on a trajectory to goal g, at each state we run 30 different goal trajectories y and score them at a range of step limits H. We score the outcome of the model using the Odysseys rubric and compare to the same performance on goal trajectory y at the same step limits H starting from the baseline state.

## 1. Theory

In Turner's framework, the attainable utility of a goal is how well an agent could do that goal from the current state. In terms of the optimal policy, we can say that the attainable utility of a goal at a state equals how well the best possible agent could do that goal starting from that state.

$$
\mathrm{AU}_R(s) \;=\; V^{*}_{R}(s) \tag{1}
$$

*Where:*

- $\mathrm{AU}_R(s)$: attainable utility of goal $R$ at state $s$
- $R$: the goal, written as a reward function
- $s$: the current state
- $V^{*}_{R}(s)$: how much reward an optimal agent pursuing $R$ collects from $s$

POWER builds on this by averaging attainable utility over many possible goals. The power of a state is how well an optimal agent could do each goal from that state, averaged over goals drawn from a distribution $\mathcal{D}$. In this average, the optimal agent's future reward on each goal is discounted at rate $\gamma$. The reward the agent gets just for being at the state is subtracted. POWER therefore counts only the reward the agent could still earn from that state onward by acting optimally on each goal.

$$
\mathrm{POWER}_{\mathcal{D}}(s,\gamma) \;=\; \frac{1-\gamma}{\gamma}\;\mathbb{E}_{R\sim\mathcal{D}}\!\left[\,V^{*}_{R}(s,\gamma) - R(s)\,\right] \tag{2}
$$

*Where:*

- $\mathcal{D}$: the distribution the goals are drawn from
- $\gamma$: discount rate between 0 and 1: how much the agent cares about later reward
- $\mathbb{E}_{R\sim\mathcal{D}}$: average over goals $R$ drawn from $\mathcal{D}$
- $V^{*}_{R}(s,\gamma)$: optimal value of goal $R$ from $s$ under discount $\gamma$
- $R(s)$: reward received just for being at $s$, subtracted out
- $\tfrac{1-\gamma}{\gamma}$: normalizing factor that keeps POWER on a comparable scale across $\gamma$

Attainable Utility Preservation (AUP) turns attainable utility into a way to penalize side effects. It does this by comparing two things for each of $N$ auxiliary goals: how well an optimal agent could do the goal after taking an action $a$, and how well it could do the goal after doing nothing ($\varnothing$). The penalty adds up the size of these differences, so gains and losses both count.

$$
\mathrm{Penalty}(s,a) \;=\; \sum_{i=1}^{N} \left|\, Q^{*}_{R_i}(s,a) - Q^{*}_{R_i}(s,\varnothing) \,\right| \tag{3}
$$

*Where:*

- $a$: the action being judged
- $\varnothing$: the no-op action (doing nothing)
- $R_1 \dots R_N$: the $N$ auxiliary goals
- $Q^{*}_{R_i}(s,a)$: how well an optimal agent could do goal $R_i$ after taking action $a$ at $s$
- $\lvert\cdot\rvert$: absolute value: gains and losses both count

These definitions all assume an optimal agent. A real agent is not optimal, so in applying this theory to evaluations, we adapt our equation from the one involving the optimal policy to instead involve the suboptimal policy $\pi$ that we actually run. For an auxiliary goal $y$ and step limit $H$, the attainable utility of $y$ at state $s$ for our agent is the probability that it completes $y$ within $H$ steps when it starts from $s$. Equivalently, it is the average over rollouts of the success of reaching auxiliary goal $y$ within $H$ steps. Averaging this over auxiliary goals drawn from a distribution of tasks $\mathcal{D}$ gives our policy's power at $s$.

$$
\mathrm{POWER}^{\pi}_{\mathcal{D}}(s, H) \;=\; \mathbb{E}_{y\sim\mathcal{D}}\Big[P\big(\pi \text{ completes } y \text{ within } H \text{ steps} \;\big|\; \text{start at } s\big)\Big] \tag{4}
$$

*Where:*

- $\pi$: our policy: the model plus its prompts, scaffold, and settings
- $\mathcal{D}$: the distribution of tasks the auxiliary goals are drawn from
- $\mathbb{E}_{y\sim\mathcal{D}}$: average over auxiliary goals $y$ drawn from $\mathcal{D}$
- $y$: the auxiliary goal (a reassigned Odysseys task)
- $H$: the step limit for the auxiliary goal
- $s$: the state the rollout starts from

The limitation of this formalization is that we can't average over all auxiliary goals because, in any given state, we can't figure out what all possible goals might be. Instead, we start by sampling 30 auxiliary goals at each state, though this number is an arbitrary choice. We then get an estimate of the POWER in [equation 4](#eq-4), which is the attainable utility averaged across those 30 sampled goals.

$$
\widehat{\mathrm{POWER}}^{\pi}_{\mathcal{D}}(s, H) \;=\; \frac{1}{30}\sum_{i=1}^{30} P\big(\pi \text{ completes } y_i \text{ within } H \text{ steps} \;\big|\; \text{start at } s\big) \tag{5}
$$

*Where:*

- $\widehat{\mathrm{POWER}}$: an estimate of power
- $y_1 \dots y_{30}$: the 30 auxiliary goals sampled from $\mathcal{D}$
- $\tfrac{1}{30}\sum_{i=1}^{30}$: average over those 30 goals, replacing $\mathbb{E}_{y\sim\mathcal{D}}$

Unlike Turner's POWER, this estimate uses our policy $\pi$ instead of the optimal policy, so it reflects what our agent actually does on each auxiliary goal rather than what the best possible agent could do from the same state. See [Context confounds](#context-confounds) below for the limitations of this.

Because no policy can beat the optimal one, how well our agent does on a goal from a state is never more than how well the best possible agent could do on it. Every policy's value is therefore a lower bound on attainable utility.

$$
\mathrm{AU}^{\pi}_{y,H}(s) \;\le\; \mathrm{AU}^{*}_{y,H}(s) \quad \text{for every } \pi \tag{6}
$$

*Where:*

- $\mathrm{AU}^{\pi}_{y,H}(s)$: our agent's attainable utility
- $\mathrm{AU}^{*}_{y,H}(s)$: the optimal agent's attainable utility

### Scoring with the Odysseys rubric

The Odysseys environment gives us an alternative way of scoring goals. Instead of scoring goals as simply completed or not, each task has a rubric with 3 to 12 items, where a judge scores the model performance separately. We can use this rubric scoring to see partial progress on goals that are not fully completed, to reduce noise between rollouts, and to see which parts of a goal a state makes easier or harder.

This fine-grained detail matters because full completion is difficult on the Odysseys tasks, especially for weaker models (see the [table of agent scores by difficulty](#difficulty-table) in section 3). Because these tests are difficult, with a simple completed-or-not score, most rollouts would score zero from every state. Changes in attainable utility would be difficult to detect unless they were large enough to score perfectly from one state and imperfectly from another. The Odysseys authors make the same point. They found that the pass/fail evaluation is inadequate for long-horizon tasks because no credit is given to an agent completing most of a task, and rubric scoring agreed better with human graders than singular judgments of the whole trajectory.

The limitation of this rubric adaptation is that the score measures expected progress rather than the probability of reaching the goal. This diverges from our idea of attainable utility because it's not defined by the ability to reach more goals, but rather by the ability to do better at reaching goals without necessarily reaching the goal exactly. To use the rubric, we replace the success indicator with the averaged rubric score that Odysseys uses: the mean of the judge's 1-or-0 verdicts on the goal's rubric items (section 6).

$$
\begin{aligned} \text{success:}\quad \mathrm{AU}^{\pi}_{y,H}(s) &= P\big(\pi \text{ completes } y \text{ within } H \text{ steps} \;\big|\; \text{start at } s\big) \\[4pt] \text{rubric:}\quad \mathrm{AU}^{\pi}_{y,H}(s) &= \mathbb{E}_{\tau\sim\pi \mid s}\Big[\,\bar{r}(\tau, y)\,\Big] \;=\; \mathbb{E}_{\tau\sim\pi \mid s}\Big[\,\frac{1}{m}\sum_{j=1}^{m} r_j(\tau)\,\Big] \end{aligned} \tag{7}
$$

*Where:*

- $\bar{r}(\tau, y)$: averaged rubric score: the mean of the judge's verdicts on $y$'s rubric items within $H$ steps
- $m$: number of rubric items for goal $y$
- $r_j(\tau)$: the judge's verdict on rubric item $j$ in rollout $\tau$: 1 for success, 0 for failure

### Attainable utility depends on the step limit H

While Turner's definitions use an infinite step horizon with a discount rate $\gamma$, our language-model agents have no explicit $\gamma$ because they are not trained to maximize a discounted sum of rewards earned at each step. Their training typically scores a whole response or episode at the end, so how far ahead they plan is implicit in their behavior rather than set by a parameter we can read off or change. To mimic the role of $\gamma$, we use a hard limit of $H$ steps, because like $\gamma$ it sets how far ahead "attainable" reaches.

Turner's $\gamma$ runs from near 0 to near 1. With $\gamma$ near 0, an optimal agent values only reward it can collect right away, so attainable utility reflects what is immediately at hand. As $\gamma$ approaches 1, reward far in the future counts almost as much as reward now, so attainable utility reflects how many goals stay reachable in the long run.

$H$ follows the same trend. With a small $H$, the agent has only a few steps, so attainable utility mostly rewards states that are already close to a goal. As $H$ grows, distant goals come within reach, and attainable utility increasingly rewards states that keep many goals reachable.

Our agent's attainable utility for auxiliary goal $y$ is the average rubric score attained for $y$ over its first $H$ steps after state $s$.

$$
\mathrm{AU}^{\pi}_{y,H}(s) \;=\; \mathbb{E}_{\tau\sim\pi \mid s}\Big[\,\bar{r}(\tau_{0:H}, y)\,\Big] \;=\; \mathbb{E}_{\tau\sim\pi \mid s}\Big[\,\frac{1}{m}\sum_{j=1}^{m} r_j(\tau_{0:H})\,\Big] \tag{8}
$$

*Where:*

- $\tau \sim \pi \mid s$: a rollout of our policy starting at state $s$
- $\tau_{0:H}$: the first $H$ steps of that rollout after the fork
- $m$: number of rubric items for goal $y$
- $r_j(\tau_{0:H})$: the judge's verdict on rubric item $j$ within those $H$ steps: 1 for success, 0 for failure

From here on, the probability in [equations 4](#eq-4) and [5](#eq-5) is replaced by this rubric-based attainable utility.

### How do we choose an H value?

We do not need a separate run for each step limit. The agent is not told its step budget, and each rollout runs once up to 200 steps, the largest step budget the Odysseys authors tested. Because the agent does not know how many steps it has, it behaves the same way in its first $H$ steps whether it will be stopped at step $H$ or allowed to continue to step 200. One long run for each auxiliary goal therefore contains a rollout for every smaller limit. To score a smaller limit, we cut the rollout off at that step and have the judge re-score the shortened trajectory against the rubric.

For each auxiliary goal, let $T$ be the average number of steps the baseline ($s_0$) rollouts take to reach their final rubric score. We sample the step limit $H$ at ten percentages of $T$, and the same absolute step limits are used for every fork of that goal, from every state and for every model.

| Percentage of $T$ | Step limit    |
|-----------------------------|---------------|
| 5%                          | $H = 0.05 \times T$ |
| 10%                         | $H = 0.1 \times T$ |
| 20%                         | $H = 0.2 \times T$ |
| 30%                         | $H = 0.3 \times T$ |
| 50%                         | $H = 0.5 \times T$ |
| 60%                         | $H = 0.6 \times T$ |
| 80%                         | $H = 0.8 \times T$ |
| 90%                         | $H = 0.9 \times T$ |
| 95%                         | $H = 0.95 \times T$ |
| 100%                        | $H = 1 \times T$ |

More $H$ values only mean more judge calls, not more agent rollouts, so adding sample points is cheap. By seeing how power changes with $H$, we hope to find a tractable way to calculate which $H$ to use when measuring power in general.

### Context confounds

An optimal policy is not affected by its context, but weaker models can be overwhelmed by it. At later states along the trajectory the agent carries a longer history, so a weaker model may reach fewer auxiliary goals. A drop in its score would then come from a limit of the model's capability rather than from a change in power.

We therefore also run forks with a fresh context. They start from the same frozen state, but the agent's history contains only the current page and the auxiliary goal. This is not a perfect control. If the power change persists with a fresh context, it likely came from the world state itself. If it disappears, the agent's capability was likely confounding the result. We expect this effect to be smaller for more capable models.

## 2. Mapping onto Odysseys

Odysseys is a benchmark of 200 long-horizon web tasks on the live internet: [project website](https://odysseysbench.com/). Each task is graded by 3 to 12 rubric items (about six on average), and each item is scored separately by a judge model. For the purposes of our experiments, this is how we map the objects in Odysseys to the formal concepts in Turner's power equation.

| Formal concept                  | In Odysseys                                                                                                                              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| State $s$             | Agent context and memory + browser and client state (tabs, cookies, logins, downloads, form contents) + state of the websites themselves |
| $s_0$                   | The initial state: fresh agent context, clean browser profile                                                                            |
| Trajectory goal $g$   | The original Odysseys task the agent is working on                                                                                       |
| Sampled state $S$     | A state on the trajectory toward $g$, reached after $k$ steps                                                        |
| Action $a$            | One browser action (click, type, navigate, tab operation)                                                                                |
| Auxiliary goal $y$    | A different Odysseys task of the same difficulty as $g$, assigned after interrupting the agent                                 |
| Reward for $y$        | Averaged rubric score for $y$: the mean of the judge's 1-or-0 verdicts on its rubric items (section 6)                         |
| Step limit $H$        | The step at which a rollout is scored. The rollout itself runs up to 200 steps.                                                          |
| Policy $\pi$            | Model weights + system prompt + scaffold + decoding settings, all held fixed                                                             |
| Goal distribution $\mathcal{D}$ | The tasks of the same difficulty as $g$, from which the 30 auxiliary goals are drawn                                           |

In practice, a state is a saved snapshot of the agent, its browser, and the websites, and starting from a state means restoring that snapshot. Pursuing an auxiliary goal $y$ means telling the agent to stop its current task and do $y$. At the baseline, where there is no current task, the agent is simply given $y$. The reward for $y$ is its rubric score from the judge, which the agent never sees. Section 4 gives the full method.

## 3. Task difficulty and goal sampling

Odysseys sorts its 200 tasks into easy (45), medium (46), and hard (109) levels, and we run the full power measurement separately at each level.

Full completion gets much harder as difficulty rises. The table below shows, for each agent on the Odysseys leaderboard\[6\], the share of tasks it completed with every rubric item met at each difficulty level.

| Agent                                   | Easy (45) | Medium (46) | Hard (109) |
|-----------------------------------------|-----------|-------------|------------|
| Skyvern (Claude Opus 5) \[hybrid\]      | 91.1%     | 89.1%       | 90.8%      |
| BrowserCode (GPT-5.6 Luna) \[terminal\] | 91.1%     | 81.9%       | 85.6%      |
| BrowserCode (Opus 4.7) \[terminal\]     | 84.4%     | 67.4%       | 56.0%      |
| Claude Opus 4.6 \[CUA\]                 | 97.8%     | 71.7%       | 11.0%      |
| GPT-5.4 \[CUA\]                         | 80.0%     | 58.7%       | 3.7%       |
| Claude Sonnet 4.6 \[CUA\]               | 82.2%     | 50.0%       | 1.8%       |
| Qwen-3.5-VL-9B \[CUA\]                  | 31.1%     | 17.4%       | 4.6%       |
| Qwen-3.5-VL-4B \[CUA\]                  | 28.9%     | 8.7%        | 3.8%       |
| GPT-5.4 Mini \[CUA\]                    | 22.2%     | 19.6%       | 1.8%       |
| Qwen-3.5-VL-35B-A3B \[CUA\]             | 24.4%     | 4.3%        | 0.0%       |
| UI-TARS-1.5-7B \[CUA\]                  | 4.4%      | 0.0%        | 0.0%       |

Percentage of tasks with every rubric item met, by difficulty, for the 11 leaderboard entries that report a difficulty breakdown, as listed on November 3, 2026. A full bar is 100%, where an optimal policy would sit. Agent type in brackets: CUA agents read screenshots and emit GUI actions, hybrid agents add structured browser tools, terminal agents drive the browser by writing code. Colors follow the leaderboard's own difficulty colors.

For a trajectory toward goal $g$, the 30 auxiliary goals at every sampled state are drawn uniformly at random, without replacement, from the other tasks at $g$'s difficulty. The trajectory task itself is excluded. The same 30 goals are reused at every sampled state on that trajectory, at the baseline, and for every model, so differences come from the state and the policy rather than from which goals were drawn. Fix the random seed for each trajectory and record the drawn task IDs.

Matching difficulty keeps the auxiliary goals comparable to the task the agent was actually doing. It also means each difficulty level has its own goal distribution $\mathcal{D}$, so POWER is estimated separately per level, and comparisons across levels are comparisons between three distributions, not one.

**Small pools.** Easy and Medium have only 44 and 45 candidate goals per trajectory, so 30 goals covers most of the pool. Different trajectories at the same level will share many goals. That is fine for the comparison, but treat goals as a shared random effect when pooling results across trajectories.

## 4. Power Measurement Method

1.  Run the agent on its trajectory task $g$ from $s_0$.
2.  At each sampled state $S$ along the trajectory (after $k$ steps), freeze all three layers in the table below.
3.  Restore 90 identical copies of $S$: three rollouts for each of the 30 auxiliary goals. No rollout ever starts from where another one ended.
4.  Add one message to each copy telling it to stop its current task and do one of the auxiliary goals $y_1 \dots y_{30}$, drawn as in section 3, with three copies per goal.
5.  Let each copy act on its goal for up to 200 steps, without telling it its budget.
6.  Have the judge score each copy on its goal's rubric at every step limit $H$ (the percentages of $T$ in section 1), by cutting the same rollout off at that step.
7.  Repeat steps 3–6 from the baseline state $s_0$ with the same 30 goals, the same $H$ values, and the same policy. At $s_0$ the agent has no current task, so it receives $y$ as its first task, using the same wording without the instruction to stop.
8.  Repeat steps 3–6 from $S$ with a fresh agent context that contains only the current page and the auxiliary goal (the control in section 1). This doubles the forks run from each sampled state.
9.  Compare, pairing by goal.

### What is frozen at each sampled state

Odysseys runs its agents inside a virtual Ubuntu machine with Chrome, so the browser and the rest of the client machine can be frozen together as one VM snapshot. The aim is that each fork resumes at exactly the moment of the interrupt, with nothing changed except the instruction it receives next.

| Layer                    | What is frozen                                                                                                                                | How it is saved                                                                                                                                                                                                        | How a fork resumes                                                                                                                                                    |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Agent state              | Full message history (system prompt, trajectory task, and every observation and action so far), scratchpad or memory files, and step counter  | Serialized to JSON alongside the snapshot ID                                                                                                                                                                           | Load the history, append one fixed interrupt message containing the auxiliary goal $y$, and reset the step counter to 0 so the goal gets the full 200 steps |
| Browser and client state | Open tabs and their URLs, active tab, scroll position, half-filled forms, cookies and logins, local and session storage, and downloaded files | Snapshot of the whole VM, including memory, so live page state survives. As a lighter fallback, Playwright `storage_state()` plus a copy of the browser profile, re-opening tabs from saved URLs (loses in-page state) | Restore the snapshot into a fresh VM, check that the active tab URL matches the record, then hand control to the agent                                                |
| Server state             | The responses the websites gave during the trajectory                                                                                         | Live web: every HTTP request and response recorded through a proxy (mitmproxy, pywb, or a Playwright HAR). Self-hosted sites: container and data-volume snapshots                                                      | Route the fork's browser through a replay proxy serving the recording. Requests that were never recorded go to the live web, and each one is logged                   |

The interrupt message is word-for-word identical across forks except for the goal text, so the only things that differ between copies are which goal they were given and the model's own sampling.

**Live-web limits.** The websites cannot truly be reset. Forks working on new goals request pages that were never recorded, and stateful actions (logins, carts, posts) cannot be replayed. Interleave baseline and $S$ rollouts in time, exclude goals with irreversible external effects, report the live-fallback fraction per rollout, and check whether $\Delta\mathrm{AU}$ holds on rollouts that were mostly replayed.

### Estimating the change

The change in attainable utility at a sampled state is the estimate of power from [equation 5](#eq-5) at the sampled state minus the same estimate at the baseline. Because both use the same 30 goals, this equals each goal's score from the sampled state minus its score from the baseline, averaged over the goals. Each goal's score is its averaged rubric score within $H$ steps ([equation 8](#eq-8)), averaged over its three rollouts. A negative value means the state lost options. The ratio compares the same two estimates by dividing instead of subtracting.

$$
\Delta\mathrm{AU} \;=\; \widehat{\mathrm{POWER}}^{\pi}_{\mathcal{D}}(S, H) - \widehat{\mathrm{POWER}}^{\pi}_{\mathcal{D}}(s_0, H) \qquad \text{ratio} \;=\; \frac{\widehat{\mathrm{POWER}}^{\pi}_{\mathcal{D}}(S, H)}{\widehat{\mathrm{POWER}}^{\pi}_{\mathcal{D}}(s_0, H)} \tag{9}
$$

*Where:*

- $\Delta\mathrm{AU}$: change in attainable utility; negative means the state lost options
- $S,\ s_0$: the sampled state and the baseline

$\Delta\mathrm{AU}$ is computed separately at each step limit $H$, giving one value per $H$ for each sampled state. With three rollouts per goal, each goal's score is still a noisy estimate of that goal's attainable utility, so the average over the 30 goals is what we read. Reading attainable utility for individual goals, for example to see which goals a state closed off, needs about 10 rollouts per goal (see below).

### Rollouts per goal and noise

We will look at how the number of rollouts per goal $r$ changes the noise in $\Delta\mathrm{AU}$. The noise has two sources: goals genuinely differ in how much $S$ affects them, and repeated rollouts of the same goal give different scores because the model samples differently each time. With $G$ goals and $r$ rollouts per goal, the uncertainty in our estimate is the variation in how differently goals respond to the state, divided by the number of goals, plus the rollout-to-rollout randomness, divided by the number of goals times the rollouts per goal.

$$
\mathrm{SE}(\Delta\mathrm{AU}) \;\approx\; \sqrt{\;\frac{\sigma^2_{\Delta}}{G} \;+\; \frac{2\,\sigma^2_{w}}{G\,r}\;} \tag{10}
$$

*Where:*

- $\mathrm{SE}$: standard error: the typical size of the estimate's error
- $\sigma^2_{\Delta}$: variance across goals in how much $S$ changes them
- $\sigma^2_{w}$: variance between repeated rollouts of the same goal (the 2 is because both $S$ and $s_0$ are sampled)
- $G$: number of goals (30 here)
- $r$: rollouts per goal

More rollouts per goal only shrink the second term. Once $2\sigma^2_w / r$ is small compared with $\sigma^2_\Delta$, extra rollouts barely help, and more goals would help more. Because the Easy and Medium pools are nearly used up at 30 goals, though, more rollouts per goal is the main way left to reduce noise at those levels.

- **Pilot:** run $r = 3$ on a few states at each difficulty to estimate $\sigma^2_w$ and $\sigma^2_\Delta$ for each model.
- **Default for the POWER estimate:** $r = 3$. It roughly halves the within-goal part of the noise compared with $r = 1$ and gives an estimate of $\sigma^2_w$ to report.
- **Raise $r$** if the pilot shows $\sigma^2_w$ dominating, which is likely for weak models whose scores swing between rollouts.
- **For per-goal readings:** about $r = 10$. A rubric score lies between 0 and 1, so its standard deviation is at most 0.5, and with 10 rollouts the standard error of one goal's mean is at most about 0.16.

## 5. Counterfactual baseline

Each baseline is forked and scored exactly like $S$: the same 30 goals at the same $H$ values.

We propose three potential baselines for $S$. The first is the starting state $s_0$. The second is a state reached after $k$ random steps, which would show that a change came from what the agent did and not merely from the number of steps. The third is a legitimate path to similar progress on $g$, which would separate good power from bad power.

We compare against $s_0$. On the live web, $k$ random steps can land almost anywhere, so a random-steps baseline would vary too much between draws. The legitimate-path baseline matters less for Odysseys, whose tasks have no designed exploit, and building a matched path for every state is unlikely to be worth the cost.

### Comparing S with s₀

The baseline is the start of the trajectory: a fresh agent with a clean browser. The comparison asks how the agent's ability to do other tasks changed between the beginning and the moment it reached $S$. The $s_0$ forks run up to 200 steps and are scored at the same $H$ values as the $S$ forks. $H$ is counted from the fork, so neither side gets more or fewer steps for the auxiliary goal.

Because every sampled state on a trajectory shares the same $s_0$, the baseline forks are run once per trajectory and reused for every $S$ on it. The cost is that the comparison mixes the effect of what the agent did with the effects of a longer history and a busier browser. The fresh-context control in [section 1](#context-confounds) is how we tell those apart.

**Pros**

- One baseline per trajectory, shared by every sampled state on it
- Cheap: the $s_0$ forks are run once and reused
- Easy to explain

**Cons**

- Mixes the effect of the agent's actions with the effect of elapsed steps and a longer context
- Cannot say which actions caused a change

**Hidden budgets.** Check that no site session timeout or global episode cap cuts a fork short of its full 200 steps. A drop caused by the clock is not lost options.

## 6. Scoring rollouts

Finally, we score each rollout the way the Odysseys authors score each task. A judge model reads the rollout and gives each of the auxiliary goal's rubric items a binary verdict: 1 for success and 0 for failure. The rollout's score is the averaged rubric score, the mean of those verdicts. The authors also call a task perfect when every item scores 1, which is what their success rate counts. This is the averaged rubric score $\bar{r}$ in [equation 7](#eq-7), and our agent's attainable utility for the goal is its average over rollouts within $H$ steps ([equation 8](#eq-8)).

Store the full trajectory and the judge's verdict on each rubric item. Scores at the smaller step limits are computed by having the judge re-score the trajectory cut off at that step (section 1).

Rubric items in Odysseys also carry weights that the judge is shown, but the paper does not say how the weights enter the reported scores. We use the unweighted average and record the weights so a weighted version can be computed later.

## 7. Policy strength

Policy strength is mostly determined by the model. The policy, though, is everything that maps what the agent sees to what it does, so strength also depends on how much the model thinks per action, how many steps it gets, and the scaffold around it. The leaderboard shows how much the scaffold alone matters. GPT-5.4 reaches a 33.5% perfect rate as a screenshot-only agent and 60.1% inside the WebWright terminal agent.

### Candidate models

Besides our two current models, several cheap or open-weight models are worth considering. The most useful ones already have Odysseys leaderboard scores, because running them from $s_0$ in our harness checks that our setup reproduces the published numbers.

| Model                               | Access                                       | Leaderboard rubric avg / perfect | Why include it                                                                                                        | Status           |
|-------------------------------------|----------------------------------------------|----------------------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|
| Nemotron-3.5-Lightning-30B-A3B\[7\] | Open weights, via Tinker; also on OpenRouter | Not listed                       | Our weakest reference policy. It has about 3B active parameters per token.                                            | Running          |
| DeepSeek V4.1 Flash\[8\]            | OpenRouter                                   | Not listed                       | Our middle reference policy.                                                                                          | Running          |
| Qwen-3.5-VL-35B-A3B                 | Open weights, also on OpenRouter\[9\]        | 28.5% / 6.5%                     | The closest leaderboard model to Nemotron in active size. It tells us roughly where Nemotron should land.             | Candidate        |
| Qwen-3.5-VL-9B                      | Open weights                                 | 42.6% / 13.5%                    | The strongest open-weight screenshot agent on the leaderboard, and cheap to run. A good calibration model.            | Candidate        |
| Qwen-3.5-VL-4B                      | Open weights                                 | 42.9% / 10.7%                    | Very cheap. It scores close to the 9B model on rubric average but lower on perfect rate.                              | Candidate        |
| UI-TARS-1.5-7B                      | Open weights                                 | 10.0% / 1.0%                     | A floor reference. It may be too weak to reach different outcomes for different goals, which limits what it can show. | Candidate        |
| GPT-5.4 Mini                        | API                                          | 38.4% / 10.5%                    | A cheap API model that uses far fewer steps per task than the others (41.7 on average).                               | Candidate        |
| Gemini 3.5 Flash                    | API, also on OpenRouter\[10\]                | 64.7% / 48.0%                    | The strongest screenshot agent on the leaderboard and a relatively cheap one. Our best option for the stronger model. | Candidate        |
| Claude Opus 4.6 or GPT-5.4          | API                                          | 68.9% / 44.5% and 55.4% / 33.5%  | Frontier references if budget allows. They are the closest available stand-ins for $V^{*}$.                     | If budget allows |

Leaderboard numbers are for screenshot-based agents on the full benchmark, as listed on November 3, 2026. Check current pricing before choosing.

### How the candidates scale against each other

Because we score with the rubric, the rubric average is the more relevant ranking. On that measure the candidates form a fairly even ladder: UI-TARS-1.5-7B at 10.0%, Qwen-3.5-VL-35B-A3B at 28.5%, GPT-5.4 Mini at 38.4%, the two smaller Qwen models near 43%, and Gemini 3.5 Flash at 64.7%. A spread like this is what the policy-strength comparison needs. A good set of three or four evaluators would be one model from the bottom, one from the middle, and one from the top. Nemotron and DeepSeek V4.1 Flash need their own calibration runs to place them on this ladder.

The ranking shifts with difficulty and step count. On Hard tasks, every screenshot agent with a reported breakdown except Claude Opus 4.6 completes fewer than 5% of tasks, so differences between models there show up mainly in the rubric average. Models also use very different numbers of steps. GPT-5.4 Mini averages 41.7 steps per task and the Qwen models average 78 to 86. At a small $H$, a model that works in fewer steps will look stronger than it does at a large $H$. This is another reason to report attainable utility across the full range of $H$ values.

## 8. Initial run and next steps

The first run uses only the easy tasks and only DeepSeek V4.1 Flash. Its purpose is to check that the whole pipeline works end to end and to give a first look at how power changes along a trajectory, before we spend money on harder tasks and more models.

### What the run looks like

1.  **Baseline first.** Run DeepSeek from $s_0$ on all 45 easy tasks, three rollouts each, for 135 rollouts. This gives the baseline score for every possible auxiliary goal and the value of $T$ for each goal. Because every trajectory starts from the same clean $s_0$, these rollouts are shared by all trajectories.
2.  **Trajectories.** Pick 5 easy tasks as trajectory tasks $g$. Run DeepSeek once on each and save snapshots at three states, a quarter, half, and three quarters of the way through, for 15 sampled states.
3.  **Forks.** At each state, run 30 auxiliary goals with three rollouts each, once with the full context and once with a fresh context. That is 180 rollouts per state and 2,700 in total.
4.  **Scoring.** Have the judge score every rollout at the ten step limits from section 1, then compute $\Delta\mathrm{AU}$ for each state and step limit, with and without the fresh context.

In total this is about 2,835 rollouts.

### Estimated cost on OpenRouter

DeepSeek V4.1 Flash costs about \$0.30 per million input tokens and \$1.20 per million output tokens at peak rates, with cached input at about \$0.006 per million and off-peak rates at half price. The Odysseys judge, Gemini 3.1 Flash Lite Preview, costs about \$0.25 per million input tokens and \$1.50 per million output tokens\[11\]. Prices change often, so check them before running.

The token counts below are planning assumptions, not measurements. We assume each rollout averages 50 steps, each step reads about 25,000 input tokens (instructions, history, and a screenshot) and writes about 1,500 output tokens, and each judge call reads about 20,000 tokens for a full trajectory.

| Item                                | Assumption                                                                          | Cost per rollout | Total (2,835 rollouts) |
|-------------------------------------|-------------------------------------------------------------------------------------|------------------|------------------------|
| Agent, no caching                   | 1.25M input and 75K output tokens per rollout                                       | about \$0.47     | about \$1,320          |
| Agent, with caching                 | 80% of input tokens served from cache                                               | about \$0.17     | about \$490            |
| Judge, re-scored at each step limit | About 6 rubric items × 10 step limits, on trajectories averaging 54% of full length | about \$0.19     | about \$540            |
| Judge, first step per item          | One call per rubric item that reports the first step at which it was met            | about \$0.03     | about \$95             |

With caching and the judge re-scoring at each step limit, the run costs about \$1,000. Running the agent off-peak and asking the judge once per item for the step at which it was first met brings it to about \$350. Without caching it could reach about \$1,850. Asking for the first step is a change from how the Odysseys authors judge, so it should be checked against per-limit re-scoring on a sample of rollouts before relying on it.

Before the full run, a smoke test checks that snapshots restore correctly and the judge works: one trajectory, two sampled states, one rollout per goal, and no fresh-context forks. That is about 105 rollouts and costs roughly \$40.

### Next steps

- Check the pipeline: whether restored snapshots match the original state, how often forks fall back to the live web, and whether any session timeouts cut rollouts short.
- Use the measured step counts and token usage to replace the planning assumptions above and re-estimate costs.
- Look at how $\Delta\mathrm{AU}$ changes along the trajectory and with $H$, and whether the fresh-context forks change the result.
- Extend to medium and hard tasks, then to the other models in section 7.

## References

1.  Turner, A. M., Hadfield-Menell, D., & Tadepalli, P. (2020). Conservative agency via attainable utility preservation. In *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*. [https://arxiv.org/abs/1902.09725](https://arxiv.org/abs/1902.09725)
2.  Turner, A. M., Smith, L., Shah, R., Critch, A., & Tadepalli, P. (2021). Optimal policies tend to seek power. In *Advances in Neural Information Processing Systems* (Vol. 34). [https://arxiv.org/abs/1912.01683](https://arxiv.org/abs/1912.01683)
3.  Turner, A. M., & Tadepalli, P. (2022). Parametrically retargetable decision-makers tend to seek power. In *Advances in Neural Information Processing Systems* (Vol. 35). [https://arxiv.org/abs/2206.13477](https://arxiv.org/abs/2206.13477)
4.  Jang, L. K., Koh, J. Y., Fried, D., & Salakhutdinov, R. (2026). *Odysseys: Benchmarking web agents on realistic long horizon tasks* (arXiv:2604.24964). arXiv. [https://arxiv.org/abs/2604.24964](https://arxiv.org/abs/2604.24964)
5.  Odysseys. (n.d.). *Odysseys: A benchmark for long-horizon web agents*. Retrieved November 3, 2026, from [https://odysseysbench.com/](https://odysseysbench.com/)
6.  Odysseys. (n.d.). *Leaderboard*. Retrieved November 3, 2026, from [https://odysseysbench.com/leaderboard](https://odysseysbench.com/leaderboard)
7.  NVIDIA. (2026). *NVIDIA Nemotron 3.5 Lightning 30B A3B* \[Large language model\]. Vast.ai model library. Retrieved November 3, 2026, from [https://vast.ai/model/nvidia-nemotron-35-lightning-30b-a3b-bf16](https://vast.ai/model/nvidia-nemotron-35-lightning-30b-a3b-bf16)
8.  DeepSeek. (2026). *DeepSeek V4.1 Flash* \[Large language model\]. OpenRouter. Retrieved November 3, 2026, from [https://openrouter.ai/deepseek/deepseek-v4.1-flash-20260910](https://openrouter.ai/deepseek/deepseek-v4.1-flash-20260910)
9.  OpenRouter. (n.d.). *Qwen3.5-35B-A3B* \[Large language model listing\]. Retrieved November 4, 2026, from [https://openrouter.ai/qwen/qwen3.5-35b-a3b-20260224](https://openrouter.ai/qwen/qwen3.5-35b-a3b-20260224)
10. OpenRouter. (n.d.). *Gemini 3.5 Flash Lite vs. Gemini 3.5 Flash* \[Model comparison\]. Retrieved November 4, 2026, from [https://openrouter.ai/compare/google/gemini-3.5-flash-lite/google/gemini-3.5-flash](https://openrouter.ai/compare/google/gemini-3.5-flash-lite/google/gemini-3.5-flash)
11. Price Per Token. (n.d.). *Google Gemini 3.1 Flash Lite Preview pricing*. Retrieved November 5, 2026, from [https://pricepertoken.com/pricing-page/model/google-gemini-3.1-flash-lite-preview](https://pricepertoken.com/pricing-page/model/google-gemini-3.1-flash-lite-preview)
