# The autonomy model — tiers, guardrails, ledgers

How an autonomous session decides what to do without asking, and the state files that
make those decisions safe and auditable. Thresholds shown are reference defaults; the
ones marked *(tune)* are meant to be reviewed and set per project (see "Adapting the
thresholds" at the end).

## Action tiers (T0–T3)

Classify every candidate action. The boundary that matters: **reversible + bounded
failure cost → can be automated; irreversible or self-modifying → must escalate.**

### T0 — always allowed (no approval, not surfaced)
Reads, greps, audits; local edits and commits; smoke tests (<5 min); updating your own
state ledgers; memory writes; running planners/linters; dispatching **read-only**
subagents. These are observation + bookkeeping — they can't burn the cluster or change
the science.

### T1 — auto-submit allowed (logged; surfaced only if anomalous)
A single HPC job **≤ a walltime cap** *(tune; e.g. ≤24h)* — fired only if **all five
guardrails pass** (below). Also: tournament/posterior updates after a status change,
auto-deriving alternative hypotheses on a walk-back. **Log every T1 fire** to an
append-only `OVERNIGHT_AUTONOMOUS_FIRES.md` with timestamp, job ID, rationale, and the
rollback procedure. Surface proactively if: >50% of daily budget consumed, ≥`FAIL_STREAK_MAX`
failures, a hypothesis crosses a confirm/walk-back threshold, or queue wait >8h.

### T2 — act, but surface in the next heartbeat
New experiment *designs* without explicit user direction; code changes >~200 LOC;
runs on a **new model/checkpoint** (new risk profile); dispatching a subagent that
itself has submit authority. Reviewable-after, not blocked-before.

### T3 — never auto; escalate to the human
Destructive ops (`rm -rf`, `git push --force`, `scancel` on jobs you didn't submit);
external communication (PRs, email, Slack); **modifying your own protocol** (the
heartbeat, this autonomy policy) or the user's `CLAUDE.md`; touching restricted or
production-deployed artifacts; acquiring new external resources (accounts,
allocations). Write these to a `USER_INPUT_TRACKER.md` OPEN list and wait.

## The five guardrails (gate every T1 auto-submit)

| | Name | Rule | Why |
|---|---|---|---|
| **G1** | Smoke-test gate | New code runs a 60s single-input test (local or `gpu-short`) and passes — or the script is byte-identical to a prior success. | ~5000:1 cost asymmetry vs an H200-hour. Catches wrong wrapper, OOM, offline-mode failure, wiring bugs for free. |
| **G2** | Pre-registration | Falsification/pass criteria written **before** the run, in the claim record or a `*_preregistration.md`. | Kills post-hoc threshold tuning and p-hacking. |
| **G3** | Info-gain | The experiment meaningfully discriminates between currently-uncertain hypotheses (score ≥ 0.3 *(tune)* if you rank them). | Prevents spending budget on runs that don't move the needle. |
| **G4** | Daily budget | Under `DAILY_GPU_BUDGET` H200-h today; checked against `.compute_budget.json`. | Stops one decision from eating the cluster (6×8h = 48h in a tick). Forces pacing. |
| **G5** | Failure-streak rollback | Not under auto-rollback: `< FAIL_STREAK_MAX` consecutive **agent-code** failures, per `.failure_streak.json`. | Bug-loops (broken code → fail → broken fix → fail) are the priciest failure mode. |

**G5's classification rule is load-bearing:** only **new agent-written-code bugs**
increment the streak. **Infrastructure failures do NOT count** — HF API timeout, scp
race, node OOM, a walltime-kill of code that was working, a **node launch-kill** (sbatch
dies in seconds with a signal-style ExitCode and no log written — the node failed before
your first line ran, fix: `--exclude` the bad node), a **walltime-timeout of working
code** (the job finished its real work but the short partition cut it off before the
final aggregation; recover by re-aggregating on the login node, no code change), and
**partition contention** (a tiny job pends or gets launch-killed because free slots are
reserved for a higher-priority partition — see `scontrol show reservations` /
`PreemptMode`, not your code). When in doubt, ask: "did my code have a bug, or did the
cluster misbehave?" Record the classification explicitly.

**Triage from evidence, not vibes.** First stop for any dead job:
`sacct -j <id> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,NodeList`, then:

| Signature | Read it as | Streak? |
|---|---|---|
| `TIMEOUT`, code was progressing (checkpoints advanced) | walltime under-provisioned — infrastructure | no |
| `TIMEOUT`, no checkpoints ever written | code likely hung — agent code | yes |
| `FAILED`, MaxRSS ≈ ReqMem | OOM: raise `--mem` and reclassify only if the leak is yours | usually no |
| `FAILED` + Python traceback in `.err` | agent code | yes |
| Login-node command produced **no output at all** | silent login-node SIGKILL (`cluster-setup.md` §1) — infrastructure | no |
| Job "succeeded" but with-vs-without outputs are identical | silent no-op — agent code (wiring bug; see the end-to-end check) | yes |
| `NODE_FAIL`, `CANCELLED` by admin, HF 5xx during pre-cache | infrastructure | no |

## Supporting safety rules (the load-bearing few)

- **Persistent-Monitor rule (>1h jobs).** A default-timeout Monitor dies mid-watch
  on a 7h job and silently misses the verdict. `persistent: true`; a missing output
  file is normal early on, not a dead-monitor signal.
- **Offline-mode rule (compute nodes).** (See `cluster-setup.md` §4.) Non-negotiable.
- **End-to-end wiring check.** Before any with-module-vs-vanilla comparison,
  generate one token with the module active and assert the metric (e.g. an EOS-logit
  delta) is **nonzero**. Catches the "hook registered but never invoked" bug that makes
  a no-op look like a real null result. A 4.5h H200 bake-off was lost to exactly this.
- **Clean-output-dir-on-resubmit rule.** Skip-existing logic silently inherits
  bug-mode data from the prior run; the aggregator then computes from stale bad output
  and your fix never shows up. Always write the resubmit to a fresh `*_clean/` dir.

## Ledger files (transparent, append-only state)

### `.compute_budget.json` — G4 daily H200-hour tracker
```json
{
  "soft_cap_h200_hours_per_day": 24.0,
  "by_day": {
    "2026-01-01": {                                   // example day — your own dates
      "h200_hours_used": 3.0,
      "jobs": [
        {"slurm_id": "1000001", "name": "example-job",
         "elapsed_hours": 3.0, "state": "COMPLETED",
         "submitted_at": "2026-01-01T12:44:00-08:00"}
      ]
    }
  },
  "_last_updated": "2026-01-01T20:16:00-08:00",
  "_format_version": 1
}
```
- `h200_hours_used` is recomputed as the sum of the day's job `elapsed_hours`.
- Keyed by local date; **resets at local midnight.** Pick ONE timezone and use it
  consistently (this example uses `-08:00`), because it drives the reset boundary.
- The decision loop reads "remaining today" before any T1 submit. Negative → pause T1.

### `.failure_streak.json` — G5 consecutive-failure rollback
```json
{
  "current_streak": 3,
  "max_streak": 3,
  "auto_rollback_triggered": true,
  "auto_rollback_at": "2026-01-01T02:01:00-08:00",
  "auto_rollback_reason": "example: 3 consecutive agent-code bugs (1000010, 1000011, 1000012) — see history",
  "history": [
    {"timestamp": "2026-01-01T01:30:00-08:00", "slurm_id": "1000009",
     "reason": "example: login-node timeout — INFRASTRUCTURE", "counted_in_streak": false},
    {"timestamp": "2026-01-01T02:00:00-08:00", "slurm_id": "1000012",
     "reason": "example: OOM in preprocessing — AGENT CODE", "counted_in_streak": true}
  ],
  "_format_version": 1
}
```
- `record_failure(slurm_id, reason, is_agent_code_bug)` appends to `history`, and
  increments `current_streak` **only** when `is_agent_code_bug` is true.
- When `current_streak >= max_streak`, flip `auto_rollback_triggered = true`. All T1
  actions downgrade to "surface for approval" until the **human** resets the streak —
  the agent does not clear its own rollback. (This is itself a T3 boundary.)
- A clean new-code success resets `current_streak` to 0.

Both ledgers should be written atomically (tmp file + rename) with stdlib-only code so
they're cheap to touch on every tick. Keep starter copies in `assets/`.

## Adapting the thresholds

Portable as-is: the **tier framing**, the **five-guardrail shape**, the
**agent-code-vs-infrastructure** distinction, the **append-only ledgers**. Re-tune for
your project: the walltime cap, `DAILY_GPU_BUDGET`, `FAIL_STREAK_MAX`, the G3 info-gain
score, and *what counts as T2 vs T3* for your domain (e.g. which artifacts are
"restricted").
