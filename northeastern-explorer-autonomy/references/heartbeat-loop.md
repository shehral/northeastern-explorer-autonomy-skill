# The heartbeat loop — keeping an autonomous session alive over hours

How a Claude Code session sustains itself across a multi-hour / overnight cluster run:
a bounded recurring **tick**, the scheduling mechanisms that fire it, the **persistent
Monitor** that catches job completions promptly, and how they compose.

## The tick: bounded, fixed-sequence, self-rescheduling

A heartbeat tick runs on a ~30-minute cadence and is deliberately **bounded** so it
never runs away:

- **≤ 25 tool calls per tick** — a hard ceiling on compute per wake.
- **≤ 300-word user-facing report** — stays readable when the human catches up after
  sleeping.

Each tick walks the same step sequence (adapt the bookkeeping steps to your project;
the *shape* is what's portable):

1. **Anchor time.** Use one consistent timezone for all timestamps — it defines the
   daily budget reset boundary. (Don't mix laptop-local and cluster-local; if your
   laptop and the cluster are in different timezones, a several-hour skew corrupts the
   day-bucket.)
2. **Consistency check.** Run your repo's drift/consistency linter first; fix drift
   before doing anything else, so you never build on an inconsistent state.
3. **Delta audit.** `ssh -o BatchMode=yes explorer 'squeue -u <user>; sacct --starttime=now-35min ...'`
   plus a local `ls -lt` of the results dir. Anchor to the *previous* tick's notes —
   don't recount from scratch.
4. **Integrate landed jobs.** For each terminal job: read its result, update your claim/
   experiment records, and **record the outcome in the failure-streak ledger**
   (agent-code bug vs infrastructure — see `autonomy-model.md`).
5. **Update budget.** Add the job's `elapsed_hours` to today's `.compute_budget.json`.
6. **Decide & maybe submit.** Pick the top candidate experiment; run the **five
   guardrails** (G1–G5). If all pass and it's within the T1 walltime cap, submit and
   **arm a persistent Monitor** (next section). Otherwise, surface it as a
   recommendation (T2) rather than firing.
7. **Log.** Append every auto-submit to an append-only `OVERNIGHT_AUTONOMOUS_FIRES.md`
   (timestamp, job ID, sbatch path, rationale, rollback procedure).
8. **Report & reschedule.** Emit the ≤300-word report, then **reschedule the next
   tick** (below). The reschedule call is what makes the loop self-sustaining.

Cadence-based deeper actions (optional, project-specific): every Nth tick, spawn a
critic/rigor subagent to audit the last few ticks for overclaim, run a literature scan,
or close out an epoch. A small `.tick_counter.json` makes these fire on a modulo and
stays idempotent if a tick re-runs after a crash.

## Scheduling mechanisms (pick by horizon and durability need)

| Mechanism | Use when | Durability |
|---|---|---|
| `/loop <prompt>` (no interval, self-paced) | Let the model self-pace each iteration; it picks the delay per the cache-window rule below and repeats the tick. | Session-only (ends with the session). |
| `/loop 30m <prompt>` (fixed interval) | Interval-driven recurring task in an active session; good fallback to re-arm after a session break. | Session-only. |
| `CronCreate(cron, prompt, recurring=True)` | Fixed wall-clock cadence (e.g. `7,37 * * * *`). Offset minutes (`:07`/`:37`, not `:00`/`:30`) to avoid thundering-herd collisions. | Session-only unless `durable: true`. |
| `/schedule` skill | **Cross-session / cloud-resident** recurring agents that survive the session closing. The real answer for "keep running after I close my laptop." | Durable (cloud). |

**Durability caveat:** `CronCreate` defaults to session-only. Even with `durable: true`
the UI has been observed to report "Session-only" — verify with
`cat ~/.claude/scheduled_tasks.json`. For genuinely unattended multi-day operation,
prefer the `/schedule` skill (cloud remote agents). Note that any scheduled remote run
must be able to re-establish the SSH/ControlMaster session and clear Duo — a headless
run that can't authenticate to Explorer will stall at step 3.

## The cache-window delay rule (don't pick 300s)

The Anthropic prompt cache has a ~5-minute TTL. A self-paced delay choice interacts
with it:

- **1200–1800s (20–30 min)** — the default for idle research ticks. One cache miss
  buys a long, useful wait.
- **60–270s** — only when you're actively polling external state that changes on that
  timescale (a queued job about to start).
- **Never 300s** — worst of both: you pay the cache miss without amortizing it. If
  tempted by "5 minutes," drop to 270s (stay cached) or commit to 1200s+.

## Persistent Monitor: event-driven job completion

Heartbeat ticks are periodic (≤ every 30 min), so a job that finishes at :09 waits
until :37 to be noticed. A **persistent Monitor** closes that gap — it fires within
~1 min of a terminal state:

- Watch the SLURM log, filtered to terminal states:
  `tail -f <log> | grep -E 'COMPLETED|FAILED|TIMEOUT|CANCELLED'`.
- **`persistent: true` for any job > 1h** (the persistent-Monitor rule). A
  default-timeout Monitor silently dies mid-watch on a 7h job and you miss the verdict.
- **A missing log file is NOT a dead-monitor signal.** Early on, the job is queued and
  no compute node is allocated yet — the log doesn't exist. The Monitor should keep
  waiting; only a terminal-state match (or an explicit stop) ends it.
- On fire, run a fast single-job intake (process only that job's result) rather than a
  full polling scan, then let the next tick pick up anything the Monitor queued.

## How it composes into a hands-off overnight run

```
submit job  ──▶  persistent Monitor armed
   │                     │  (fires ~1 min after terminal state)
   │                     ▼
   │            single-job intake: read result, update budget + streak ledgers,
   │            update claims/tournament, queue any review request
   ▼                     │
cron/loop heartbeat ◀────┘   every ~30 min: re-anchor, run guardrails,
   │                          submit the NEXT job if G1–G5 pass, emit report,
   └─────────────────────────  reschedule itself
```

Two independent triggers (periodic heartbeat + event-driven Monitor) mean results get
integrated promptly **and** the loop survives if one mechanism dies — if the Monitor
is lost, the next heartbeat still catches the completed job via `sacct`. That
redundancy is the point.

## Recovering a stalled session

If the session died overnight and jobs are orphaned but still running on the cluster:
re-connect (`ssh explorer`, clear Duo), `squeue -u <user>` to see what's alive, read
the `OVERNIGHT_AUTONOMOUS_FIRES.md` tail and the result dirs to rebuild state, then
re-arm the loop with `/loop 30m <heartbeat prompt>`. The cluster jobs persist
independently of the session; only the watching loop needs rebuilding.
