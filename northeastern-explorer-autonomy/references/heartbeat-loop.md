# Waiting and waking

How a Claude Code session waits on cluster jobs and keeps going over hours or
overnight. The default is **event-driven**: every submitted job gets a wait that
re-invokes the session when the job ends. A clock-driven heartbeat is only a
fallback, for unattended runs and for things nothing can notify on. Every wake,
whatever caused it, runs the same bounded checklist. Harness behaviour is
version-dependent: check the assumptions table at the end at session start.

## Waiting on a job: a background until-loop on `sacct`

Watch the scheduler, not the log. Slurm never writes COMPLETED or FAILED into a job's
output file; on a time limit it writes `CANCELLED AT ... DUE TO TIME LIMIT`, not
"TIMEOUT"; a launch-kill writes no log at all.

Run this with the Bash tool's `run_in_background`. It runs detached, survives across
turns, and re-invokes the session once, when it exits: on a terminal state of every
row (array tasks included), or after 5 failed ssh polls in a row (the overnight-auth
case below). Foreground `sleep` is blocked (the Bash tool description says so), and a
chained `sleep N; cmd` was refused when tried (observed in 2.1.296; not stated in every
build's description): waiting goes in a loop like this one. The Bash description points
one-event waits at Monitor, while Monitor's own description points them at a background
until-loop; this skill follows Monitor's guidance, since a Monitor expires after at most
30 min.

```bash
J=<jobid>; fails=0
T='^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)'
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 \
            -o ServerAliveCountMax=2 <alias> "$@"; }
while :; do
  if s=$(rsh "sacct -j $J -X -n -P -o JobID,State"); then
    fails=0   # done when sacct lists the job and every row (array tasks too) is terminal
    printf '%s\n' "$s" | awk -F'|' -v t="$T" 'NF {n++; if ($2 !~ t) busy = 1}
      END {exit !(n && !busy)}' && break
  else
    fails=$((fails + 1))
    [ "$fails" -ge 5 ] && { echo "SSH_DOWN: job $J unknown after 5 polls"; exit 3; }
  fi
  sleep 120
done
printf '%s\n' "$s"
rsh "tail -n 30 <log-dir>/<jobname>-$J.out <log-dir>/<jobname>-$J.err"
```

It runs under bash and zsh. Its exit 3 is a watcher code (`SSH_DOWN`), not a job exit
code: in the job contract 3 means a failed self-check. `-P` matters: without it `sacct`
pads the State column to 10 characters, `OUT_OF_MEMORY` prints as `OUT_OF_ME+`, and the
wait never ends. The ssh options bound each call (no `timeout(1)` on stock macOS;
single login-node calls have taken over 2 min, field-observed 2026-10), and the 120 s
poll keeps load on the login node low. A just-submitted job can be missing from
`sacct` for a moment; the loop treats that as "not done".

- **A missing log is not a dead job.** A queued job has no node and no log yet. Nor is a
  silent log at start-up a hang: field-observed vLLM model loads took 2 to 18 min
  (`cluster-setup.md` §7 Serving engines in jobs).
- **On wake, intake that job only** (its result, its ledger rows), then run the
  checklist below. If it woke on NODE_FAIL or PREEMPTED and `sacct` now shows the job
  pending or running again (a requeue keeps the JobID), re-arm the wait (untested: an
  interim terminal row for a requeued job has not been observed on Explorer).

### Many jobs: one watcher, one waker

Instead of a wait per job, one laptop watcher polls `sacct` for all your project's jobs
and appends each terminal event once to a file in the project tree; a background waker
on that file re-invokes the session once per event, with no idle wakes (field-observed
2026-10 to keep costs low). A requeued job keeps its JobID, so an event's key is
(JobID, State, Start), not the JobID alone. Array ids contain `[ ] % -`, so the dedupe
compares them as strings, never as a regex.

```bash
# watcher: run under bash (not zsh) from the project dir; its state stays in the project tree
E=.cluster_events.log; touch "$E"; fails=0
T='^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)'
while :; do
  if out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 \
             -o ServerAliveCountMax=2 <alias> \
             "sacct -u <user> -X -n -P -S now-2days -o JobID,JobName,State,Start"); then
    fails=0
    printf '%s\n' "$out" | awk -F'|' -v p='<project-prefix>-' -v t="$T" \
        'index($2, p) == 1 && $3 ~ t' |
      while IFS='|' read -r id name st start; do   # ids hold [ ] % -: compare as strings
        awk -F'|' -v j="$id" -v s="$st" -v b="$start" \
            '$1 == j && $3 == s && $4 == b {f = 1} END {exit !f}' "$E" ||
          echo "${id}|${name}|${st}|${start}" >> "$E"
      done
  else
    fails=$((fails + 1)); [ "$fails" = 5 ] && echo "SSH_DOWN|-|$(date '+%F %T %Z')" >> "$E"
  fi
  sleep 120
done
```
```bash
# waker: background task; exits when undelivered events exist (re-invokes the session)
E=<project-dir>/.cluster_events.log; C="$E.seen"; touch "$E"
n=$(cat "$C" 2>/dev/null); n=${n:-0}   # cursor: lines already delivered
until [ "$(wc -l < "$E")" -gt "$n" ]; do sleep 30; done
m=$(wc -l < "$E"); sed -n "$((n + 1)),$((m))p" "$E"; echo "$((m))" > "$C"
```

Start the watcher first (its first pass records the last two days' events), then the
waker; re-arm the waker after each event. The cursor file (`.cluster_events.log.seen`)
makes a re-armed waker continue from the last line it delivered, so an event the watcher
appends between one wake and the next re-arm still wakes the session. The first waker
delivers the watcher's first pass; to skip that history, run `wc -l < "$E" > "$E.seen"`
before arming it. Both stall while the laptop sleeps.

### Streams: Monitor

Monitor turns each stdout line of a command into an event. Use it for streams
(progress lines, several events from one job), not for "tell me when the job ends":

```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 \
    <alias> "tail -n 0 -F <log-dir>/<jobname>-<jobid>.out <log-dir>/<jobname>-<jobid>.err" |
  grep --line-buffered -E 'step=|Traceback|Error|FATAL|CANCELLED|Killed|exited|done:'
```

- **Silence is not success.** The filter must match failure lines as well as progress,
  and `grep` needs `--line-buffered` (`awk` needs `fflush()`) or events sit in a buffer.
  The log never shows the end state, so pair a stream with the `sacct` wait above.
- **It expires.** `timeout_ms` defaults to 5 min and is capped at 30 min; there is no
  persistent flag. Set the maximum and re-arm on each expiry.

**Which one:** one event (the job ended) → background until-loop. Many jobs → watcher +
waker. A stream of lines you will act on → Monitor, re-armed, alongside the wait.

## The fallback clock

Use a clock only when the run is unattended (a backstop if a wait dies) or when you
wait on something nothing can notify on: a queue start estimate improving, a submit
slot freeing under the per-user cap. Never wake just to keep the prompt cache warm.

| Mechanism | Use when | Durability |
|---|---|---|
| `/loop <prompt>` (self-paced) | The model picks each delay with `ScheduleWakeup` (main session only); `noop=true` for a quiet tick, `stop=true` ends the loop. | Session-only. |
| `/loop 30m <prompt>` (fixed interval) | A fixed cadence in an active session. | Session-only; recurring tasks expire after 7 days. |
| `CronCreate(cron, prompt, recurring=True)` | A wall-clock cadence such as `7,37 * * * *` (avoid minutes :00 and :30). | Session-only (the `durable` flag has no effect); expires after 7 days. |

- **`ScheduleWakeup` exists only in a main session running a self-paced `/loop`**
  (subagent sessions do not have it) and clamps the delay to 60–3600 s (per the 2.1.296
  main-session schema). The prompt cache lasts 1 hour (per the 2.1.296 ScheduleWakeup
  schema; re-check in your build), so there is no 5-minute cache cliff and no reason to
  avoid 300 s. Choose the delay by what you wait for: the real change
  rate of an external state the harness cannot notify on; 1200 s or more for a fallback
  heartbeat while a wait is the primary wake; 1200–1800 s when idle.
- **Cron fires only while the session is idle**, with jitter up to 10% of the period
  (max 15 min). It is not for live watching.
- Keep the wake prompt in a tracked `.claude/loop.md` (bare `/loop` uses it).

## Durability: what survives a session exit

Cluster jobs survive; background tasks, Monitors, session crons and `/loop` do not, and
none is restored when the session restarts. For a durable local tick, use a **Desktop
scheduled task**: it has your files and SSH config, needs the machine on, and needs key
auth with no MFA (overnight auth below). A cloud routine (`/schedule`, RemoteTrigger)
runs in a fresh cloud clone without your SSH key or Duo, so it cannot reach Explorer
(inferred, not field-tested); use it only for cluster-free work. Progress that must not
depend on the session or the laptop belongs on the cluster as dependency chains
(`cluster-setup.md` §5 Dependencies).

## Notifications

`PushNotification` sends a desktop notification, and a phone push when Remote Control
is connected. Send one only when the owner has likely walked away and something needs
them (a G5 rollback, `SSH_DOWN` needing Duo, a T3 decision) or a long campaign
finished. Under 200 characters, one line, the action first ("Duo needed: 3 jobs
unwatched since 02:10"). Not for routine job completions.

## Subagents and workflows

- **Background subagents** read large logs and sweep result dirs, and return
  conclusions, so the main context stays small.
- **The Workflow tool** (deterministic multi-agent fan-out, verify, review) exists only
  in a main session and runs only on the owner's explicit opt-in. State the agent count and the cost before launching.

## Compaction and re-anchoring

Long sessions get summarised, and the summary drops detail. What survives is on disk:
a RESUME file, the ledgers, `OVERNIGHT_AUTONOMOUS_FIRES.md`. Re-anchor from them after
every wake and after every summary, not on a clock, and update the RESUME file before
every wait you arm, recording its background task id and job id.

## The checklist to run on every wake

Bounded so a wake never runs away: **≤ 25 tool calls, ≤ 300-word report.** Adapt the
bookkeeping to your project; the shape is what's portable.

1. **Re-anchor.** Read the RESUME file and ledgers. Check that the waits recorded there
   (background task id, job id) are still armed, since there may be no tool that lists
   background tasks; re-arm any that died. If the harness disagrees with the assumptions
   table, log it to the field log (`evolution.md`).
2. **Anchor time.** Take every timestamp from a `date` call in the same step, never an
   estimate (field-observed 2026-10: estimated times ran 10 to 20 min off and made logs
   disagree with the ledger). Use one timezone for the ledgers, since it defines the
   daily budget reset, and label which clock each timestamp uses: `sacct` prints
   US/Eastern while the laptop may be in another timezone (e.g. US/Pacific), where a
   late-evening job can land in the next Eastern day. Correct a wrong time in place ("first written as HH:MM").
3. **Consistency check.** Run your repo's drift linter first; fix drift before anything
   else.
4. **Delta audit.** `squeue -u <user>` plus `sacct -S <time of last wake>` through the
   alias, and `ls -lt` of the results dir. Anchor to the previous wake's notes; don't
   recount from scratch.
5. **Integrate landed jobs.** Read each terminal job's result, update your records, and
   record the outcome in the failure-streak ledger (agent code vs infrastructure:
   `autonomy-model.md` triage table). A cluster behaviour the skill does not cover goes
   to the field log too.
6. **Update budget.** Add `elapsed_hours` × GPUs under the job's GPU type to today's
   `.compute_budget.json`, the one ledger of record (`autonomy-model.md` "Ledger
   files"). Book hand-submitted jobs and jobs cancelled while pending (0 GPU-h) too, so
   the ledger matches `sacct`.
7. **Decide and maybe submit.** Run G1–G5 on the top candidate. If all pass and it is
   within the T1 walltime cap, choose its GPU type (`cluster-setup.md` §5.1; also move
   your project's jobs that have pended ~15 min while a fitting type sits free), submit,
   and arm its wait. Otherwise surface it as a T2 recommendation.
8. **Log.** Append every auto-submit to the append-only `OVERNIGHT_AUTONOMOUS_FIRES.md`
   (timestamp, job ID, sbatch path, rationale, rollback procedure).
9. **Report and re-arm.** Emit the report, update the RESUME file, and make sure the
   next wake exists: a wait per running job, plus the fallback clock if unattended.

If nothing changed, log one line and re-arm within a few tool calls. Deeper actions
(a critic subagent auditing the last few wakes for overclaim, a literature scan,
closing an epoch) can fire every Nth wake from a small `.tick_counter.json`, which
stays idempotent if a wake re-runs after a crash.

## Harness assumptions (check at session start)

Observed in Claude Code 2.1.296 on 2026-10-24. Rows marked *main session* were read
from the 2.1.296 main-session tool schemas; a subagent session has neither
`ScheduleWakeup` nor the Workflow tool. Rows marked *observed* are behaviour seen in
that build, not text in a schema. Compare with your own tool list; log any drift to the
field log (`evolution.md`) and follow what your schemas say.

| Assumption | How to check |
|---|---|
| Bash `run_in_background` runs detached across turns and re-invokes the session once when the command exits | Bash tool description |
| Foreground `sleep` is blocked (Bash description); a chained `sleep N; cmd` is refused (*observed* in 2.1.296, not in every build's description) | Bash tool description; one try |
| One-event waits: the Bash description says Monitor with an until-loop, Monitor's says Bash `run_in_background`; this skill follows Monitor's | Both descriptions |
| Monitor: each stdout line is an event; `timeout_ms` default 5 min, cap 30 min; no persistent flag | `ToolSearch "select:Monitor"` |
| *Main session:* `ScheduleWakeup` exists only in a main session's self-paced `/loop` (absent in subagent sessions); delay clamped to 60–3600 s; prompt cache 1 h (per the 2.1.296 ScheduleWakeup schema; re-check in your build) | `ToolSearch "select:ScheduleWakeup"` from inside a `/loop` |
| `CronCreate` is session-only (`durable` has no effect); recurring jobs expire after 7 days; fire only when idle; jitter up to 10% (max 15 min) | `ToolSearch "select:CronCreate,CronList"` |
| `PushNotification`: desktop, plus phone when Remote Control is connected; under 200 characters | `ToolSearch "select:PushNotification"` |
| Cloud routines (`/schedule`, RemoteTrigger) run without your SSH key or Duo | `ToolSearch "select:RemoteTrigger"`; a routine's run log |
| *Main session:* the Workflow tool exists only in a main session, and runs only with the owner's opt-in | the session's tool list |
| Background tasks, Monitors and session crons end with the session; cluster jobs do not | `CronList`, plus the waits recorded in the RESUME file, after a restart (a task-listing tool may not exist) |

## The overnight auth reality (plan around it, don't fight it)

If the L1 `BatchMode` probe succeeds on a **cold** socket (none open yet), key auth
needs no MFA and an overnight loop can reconnect. Otherwise the ControlMaster window
(~8 h) is the limit, since a headless session cannot re-clear Duo:

1. **Front-load** every guardrail-passing submission while auth is warm, and pre-queue
   dependent work as cluster-side chains. The jobs need nothing from the session.
2. **Expect a read-only gap.** When the socket dies, the waits above exit with
   `SSH_DOWN` instead of polling a dead socket all night. On that wake, log it, queue
   the intended actions locally, notify the owner once, and keep re-anchoring from
   local state without retrying SSH.
3. **Morning recovery is the human's first act:** re-`ssh`, clear Duo, then run the
   stalled-session recovery below; the queued actions and the `sacct` catch-up drain in
   one wake.

## Local-side (laptop) gotchas that fake a stall

Check these before declaring work dead:

- **The laptop is the weak link in long chains.** A sleeping Mac stalls every
  laptop-side watcher and feeder, and it deep-sleeps with the lid closed even under
  `caffeinate -i`. The cluster does not wait, but the next submission does. The Claude
  Code session scratchpad is pruned around midnight: a 4-day laptop helper lost its
  script and queue file there (a loop already running survived on its open file
  descriptor). Keep watcher and feeder state in the project tree, run long laptop
  chains only with the lid open and power connected, and **pre-queue dependent work on
  the cluster** (`afterany`/`afternotok` chains, wrapper jobs; `cluster-setup.md` §5
  Dependencies).
- **BSD `find` rejects `-newermt` with relative times by returning nothing** instead of
  erroring, so "no files modified since X" may mean the predicate was invalid. Prefer
  `ls -lt | head` or GNU find, and treat an empty freshness probe as "verify the probe".
- **No `timeout(1)` on stock macOS**, and `python3` can resolve to a stray venv: bound
  ssh calls with its own options (above) and pin absolute paths for anything a wake
  runs locally.
- **zsh, the macOS default shell, does not word-split an unquoted `$var`,** so
  `for x in $list` or `set -- $cfg` sees one word. It also applies **history
  modifiers** after a colon, even inside double quotes: `"$H:refs/heads/x"` is `$H`
  with the `:r` modifier, giving `<sha>efs/heads/x`. Brace every variable a colon
  follows (`${H}:refs/...`), use arrays or functions, or build remote command strings in
  a bash script or Python.

## Recovering a stalled session

If the session died and jobs are still running on the cluster: reconnect (`ssh <alias>`,
clear Duo if asked), `squeue -u <user>` to see what is alive, and rebuild state from the
RESUME file, the `OVERNIGHT_AUTONOMOUS_FIRES.md` tail and the result dirs. Nothing the
old session armed survives, so re-arm a wait for every running job (or the watcher and
waker), and the fallback clock if the run stays unattended. Check that any laptop-side
watcher or feeder is still running and its files still exist. The cluster jobs persist
independently; only the waiting needs rebuilding.
