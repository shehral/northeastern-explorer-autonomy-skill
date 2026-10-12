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
3. **Delta audit.** `ssh -o BatchMode=yes <alias> 'squeue -u <user>; sacct --starttime=now-35min ...'`
   plus a local `ls -lt` of the results dir. Anchor to the *previous* tick's notes —
   don't recount from scratch.
4. **Integrate landed jobs.** For each terminal job: read its result, update your claim/
   experiment records, and **record the outcome in the failure-streak ledger**
   (agent-code bug vs infrastructure — see `autonomy-model.md`).
5. **Update budget.** Add the job's GPU-hours (`elapsed_hours` × GPUs) under its GPU
   type to today's `.compute_budget.json`.
6. **Decide & maybe submit.** Pick the top candidate experiment; run the **five
   guardrails** (G1–G5). If all pass and it's within the T1 walltime cap, choose its GPU
   type (`cluster-setup.md` §5.1; also move any of your project's jobs that have pended
   ~15 min while a fitting type sits free), submit, and
   **arm a persistent Monitor** (next section). Otherwise, surface it as a
   recommendation (T2) rather than firing.
7. **Log.** Append every auto-submit to an append-only `OVERNIGHT_AUTONOMOUS_FIRES.md`
   (timestamp, job ID, sbatch path, rationale, rollback procedure).
8. **Report & reschedule.** Emit the ≤300-word report, then **reschedule the next
   tick** (below). The reschedule call is what makes the loop self-sustaining.

With a persistent Monitor on every running job, the tick is a backstop. If nothing
changed since the last tick, log one line and reschedule within a few tool calls. A
30-min recurring task may also start up to 15 min late (scheduler jitter).

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
| Desktop scheduled task | A durable local tick with your files and SSH config. It needs the machine on, and key auth with no MFA (see "overnight auth"). | Durable (local). |
| `/schedule` (cloud routine) | Cluster-free work only. A routine runs in a fresh cloud clone with no local files and cannot use your SSH key or clear Duo, so it cannot reach Explorer. | Durable (cloud). |

**Durability caveat.** Recurring `/loop` and `CronCreate` tasks expire **7 days** after
creation. A self-paced `/loop`, Monitor tasks and background Bash tasks are **not
restored** on `--resume`. So **step 0 of every tick** is:
1. Run `CronList`.
2. If the heartbeat is missing or more than 6 days old, delete and re-create it.
3. Re-arm a Monitor for every running job.

Keep the tick prompt in a tracked `.claude/loop.md` (bare `/loop` uses it). Checked
against the Claude Code scheduled-tasks docs, 2026-10; this is version-dependent, so
re-check.

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

- **Watch the scheduler, not the log.** Slurm never writes COMPLETED or FAILED into a
  job's output file; on a time limit it writes `CANCELLED AT ... DUE TO TIME LIMIT`, not
  "TIMEOUT"; a launch-kill writes no log at all. So poll `sacct`:
  `until s=$(ssh -o BatchMode=yes <alias> "sacct -j <id> -X -n -o State" | head -1); echo "$s" | grep -qE 'COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL'; do sleep 120; done; echo "$s"`.
  The 120 s poll keeps each ssh call short. Grep the log only for your own sentinel
  lines (e.g. the template's `=== <job> done ===`).
- **`persistent: true` for any job > 1h** (the persistent-Monitor rule). A
  default-timeout Monitor silently dies mid-watch on a 7h job and you miss the verdict.
- **A missing log file is NOT a dead-monitor signal.** Early on, the job is queued and
  no compute node is allocated yet — the log doesn't exist. The Monitor keeps polling
  the scheduler; only a terminal state (or an explicit stop) ends it.
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

## The overnight auth reality (plan around it, don't fight it)

If the L1 `BatchMode` probe succeeds on a **cold** socket (none open yet), key auth
needs no MFA and an overnight loop can reconnect. Otherwise the ControlMaster window
(~8 h) is the limit, since a headless session cannot re-clear Duo: front-load
submissions and expect a read-only gap. In that case plan the night as:

1. **Front-load** the submissions while auth is warm — get every guardrail-passing
   job into the queue *before* sleep; the jobs run server-side and need nothing from
   your session.
2. **Expect a read-only gap.** When the socket dies mid-night, `BatchMode` probes
   start failing — the tick should detect this, log it, **queue** its intended
   actions locally, and keep re-anchoring from local state instead of retrying SSH
   blindly (each retry is a wasted tool call against a dead socket).
3. **Morning recovery is the human's first act:** re-`ssh`, clear Duo, then run the
   stalled-session recovery below — the queued actions and `sacct` catch-up drain in
   one tick.

## Local-side (laptop) gotchas that fake a stall

These macOS traps have produced false "no progress overnight" alarms or silently
wrong local commands — check them before declaring work dead:

- **BSD `find` rejects `-newermt` with relative times by returning nothing** instead
  of erroring — "no files modified since X" may mean your predicate was silently
  invalid, not that nothing ran. Prefer `ls -lt | head` or GNU find, and treat an
  *empty* result from a freshness probe as "verify the probe" first.
- **No `timeout(1)` on stock macOS**, and `python3` can resolve to a stray venv —
  pin absolute paths for anything a tick runs locally.
- **zsh, the macOS default shell, does not word-split an unquoted `$var`,** so
  `for x in $list` or `set -- $cfg` sees one word. Use arrays, or run the snippet under
  `bash`.

## Recovering a stalled session

If the session died overnight and jobs are orphaned but still running on the cluster:
re-connect (`ssh <alias>`, clear Duo if asked), `squeue -u <user>` to see what's alive, read
the `OVERNIGHT_AUTONOMOUS_FIRES.md` tail and the result dirs to rebuild state, then
re-arm the loop with `/loop 30m <heartbeat prompt>` and a Monitor for every running job
(neither survives a resume). The cluster jobs persist
independently of the session; only the watching loop needs rebuilding.
