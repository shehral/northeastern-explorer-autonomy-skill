# Self-evolution: the field log and the evolve procedure (human-gated)

The cluster drifts (drivers, partitions, QOS caps) and so does the harness (tools,
limits, caching). This file is how agents record where the skill is wrong or silent,
and how the owner folds that evidence back in. Two rules hold throughout:

- **Agents never edit the skill mid-task.** Logging evidence is T0. Proposing a skill
  change is T0/T2 (surface it in the next report). Applying one is T3: an evolve pass
  edits only on a branch the owner opted into, and only the owner commits, pushes,
  merges or syncs the installed copy (`autonomy-model.md` tiers).
- **The field log is private; the skill is public.** Project tags, job ids, paths and
  evidence stay in the log and the audit dir. Only a sanitized, generalized lesson
  reaches the skill (§6).

## 1. What to log, and when

Append one line the moment the cluster or the harness disagrees with or adds to the
skill, not at session end (a context summary can erase it first):

- a failure no triage row matches (`autonomy-model.md` triage table), or one that
  matched the wrong row;
- a node, driver or probe result not yet in your driver table (§3), or a probe job's
  answer to an open question;
- a queue or placement trick that worked or failed (`cluster-setup.md` §5.1 queue cascade);
- a harness tool behaving differently from the harness assumptions table (§4);
- a skill instruction or snippet that was wrong (`skill-error`).

Do not log routine successes, project results or anything secret (tokens, keys).

**Where.** `.explorer_field_log.jsonl` at the project root, gitignored or in a private
repo (add it to `.gitignore` when you create it). One file per project; append only.

## 2. The schema (one JSON object per line)

`assets/field_log.schema.json` is the JSON Schema for one line:

| Field | Required | What |
|---|---|---|
| `date` | yes | from a `date` call: `YYYY-MM-DD`, optionally with time and offset |
| `project_tag` | yes | free text; private |
| `category` | yes | `driver`, `node`, `queue`, `partition`, `submission`, `signal`, `storage`, `transfer`, `environment`, `loader`, `serving`, `triage`, `harness`, `skill-error`, `other` |
| `observation` | yes | what happened, one or two sentences |
| `evidence` | yes | job id and sacct line, log line, tool schema text; private |
| `confirmation` | yes | `observed` (a real run), `probed` (a deliberate test), `inferred` (reasoning only) |
| `generalizes` | yes | true if it would hold for other users or projects |
| `skill_ref` | no | the skill file and section it concerns |
| `proposed_change` | no | suggested wording or fix |

Two example lines (placeholders; real lines carry real, private evidence):

```json
{"date": "2026-01-02", "project_tag": "<project-tag>", "category": "driver", "observation": "H200 NVL node <node> runs driver 595; with the cuda-13.0 compat libcuda torch reports CUDA unavailable, without it a cu130 smoke passes.", "evidence": "job <jobid>: NODE_DRIVER node=<node> gpu=NVIDIA H200 NVL driver=595.58.03; smoke log", "confirmation": "observed", "generalizes": true, "skill_ref": "references/cluster-setup.md section 3 driver gate", "proposed_change": "Load the compat libcuda only on 570-579 drivers."}
{"date": "2026-01-03", "project_tag": "<project-tag>", "category": "harness", "observation": "A subagent session's tool list has no ScheduleWakeup and no Workflow tool.", "evidence": "ToolSearch select:ScheduleWakeup returned nothing in session <session-tag>", "confirmation": "probed", "generalizes": true, "skill_ref": "references/heartbeat-loop.md Harness assumptions"}
```

Validate the log after appending (laptop; uses `jsonschema` when installed, else checks
keys, enums and the date):

```bash
LOG=.explorer_field_log.jsonl SCHEMA="<skill-dir>/assets/field_log.schema.json"
python3 - "$LOG" "$SCHEMA" <<'PY'
import json, re, sys
log, schema = sys.argv[1], json.load(open(sys.argv[2]))
try:
    from jsonschema import validate
except ImportError:                      # stdlib fallback: keys, enums, date
    def validate(e, s):
        p = s["properties"]
        missing = [k for k in s["required"] if k not in e]
        bad = [k for k in ("category", "confirmation") if k in e and e[k] not in p[k]["enum"]]
        bad += [] if re.match(p["date"]["pattern"], str(e.get("date", ""))) else ["date"]
        bad += [k for k in ("project_tag", "observation", "evidence")
                if k in e and not (isinstance(e[k], str) and e[k].strip())]
        bad += [] if isinstance(e.get("generalizes"), bool) else ["generalizes"]
        extra = [k for k in e if k not in p]
        if missing or bad or extra:
            raise ValueError(f"missing={missing} bad_enum={bad} unknown={extra}")
n = 0
for i, line in enumerate(open(log), 1):
    if line.strip():
        try:
            validate(json.loads(line), schema)
        except Exception as ex:
            sys.exit(f"line {i}: {str(ex).splitlines()[0]}")
        n += 1
print(f"{n} entries OK")
PY
```

## 3. The driver table

Every template job prints one line in a fixed, greppable format:
`NODE_DRIVER node=<host> gpu=<name> driver=<version>` (the GPU name may contain
spaces). `assets/fold_drivers.py` folds them into a driver table JSON
(`cluster-setup.md` §3): `python3 fold_drivers.py --table <driver-table.json> <log-dir>`
(`--dry-run` to preview). It orders readings by each log's full mtime (kept as
`last_seen_ts`), so pull logs with their mtimes (`scp -p`, `rsync -a`, `tar`): a plain
copy stamps them all with the copy time. Reruns on the same logs change nothing. A
generic `host:` or `node=` line names a node only if the host fully matches `--node-re`
(default `^[a-z][0-9]{4}$`). Keep one table for all your projects, next to your shared
exclude list (`cluster-setup.md` §9.3). It is Python, so run it on the laptop over
pulled logs or in a `sharing` CPU job, never on the login node. A quick look needs only
grep, which the login node allows:

```bash
grep -rh --include='*.out' '^NODE_DRIVER ' "<log-dir>" | sort | uniq -c
```

A node or driver the table did not have is a field-log entry (`category: driver`).
Node names and driver versions are cluster-wide facts and may enter the skill; job ids
may not.

## 4. Session-start drift check

At session start, check the table "Harness assumptions (check at session start)" in
`heartbeat-loop.md` against your own tool list (ToolSearch `select:` on the tools it
names, plus the Bash tool's description). For each row that differs (a tool missing, a
limit changed, a flag added), log one `harness` entry with `confirmation: probed`, and
follow what your schemas say for the rest of the session. Do not edit the table: the
evolve procedure does that.

## 5. The evolve procedure

**Trigger.** About 15 new entries across the listed projects since the newest release
in `CHANGELOG.md`, or monthly, whichever comes first. Count them (laptop; the release
day counts, so entries logged after a release that day are not missed, and the Gap
step drops any the release already covers):

```bash
REPO="<repo-dir>"; PROJECTS=("<project-dir>" "<project-dir-2>")
SINCE=$(sed -n 's/^## [0-9][0-9.]* — \([0-9-]*\).*/\1/p' "$REPO/CHANGELOG.md" | head -n 1)
for p in "${PROJECTS[@]}"; do
  python3 - "$p/.explorer_field_log.jsonl" "$SINCE" <<'PY'
import json, os, sys
f, since = sys.argv[1], sys.argv[2]
n = sum(1 for l in open(f) if l.strip() and json.loads(l)["date"][:10] >= since) if os.path.exists(f) else 0
print(f"{n} new since {since}: {f}")
PY
done
```

Keep the project list in a private file, never in the skill.

**Opt-in.** The agent proposes the run; the owner opts in. `assets/evolve_workflow.js`
runs it with the Workflow tool, which exists only in a main session with the owner's
opt-in (read from the 2.1.296 main-session schemas; re-check in your build). Before
asking, state the agent count (P projects + 1 gap + E target files + 2 reviewers + 1
fix, all opus) and a rough cost: each of those agents reads the whole skill. Without
the Workflow tool (a subagent session, or a build without it), run the same steps by
hand, in order.

**Prepare (owner).** Create a new branch in the repo clone
(`git -C <repo-dir> switch -c docs/evolve-<date>`) and a private audit dir outside the
repo (`<audit-dir>`). Optionally put the owner's private values, one per line, in
`<audit-dir>/private_patterns.txt` for the sanitization grep (§6). Then run:

```
Workflow({scriptPath: "<skill-dir>/assets/evolve_workflow.js",
          args: {skill_dir: "<skill-dir>", repo_dir: "<repo-dir>",
                 projects: ["<project-dir>", "<project-dir-2>"],
                 branch: "docs/evolve-<date>", audit_dir: "<audit-dir>"}})
```

The steps (each agent passes results through files in the audit dir, not prompts):

1. **Harvest**, one read-only agent per project: its field log from the last release
   date on, plus continuity docs (resume and status notes, ledgers, job READMEs) for
   learnings never logged. Writes `harvest/<n>.json` in field-log shape, private
   evidence kept.
2. **Gap**, one agent: deduplicates across projects and diffs each item against the
   skill as NEW (skill silent), REFINES (right but incomplete), CONTRADICTS (skill says
   otherwise) or OUTDATED (no longer true), with target section, quote, proposed
   wording and evidence pointers. Writes only `GAP.json`; touches no skill file.
3. **Edit**, one editor per target file, on the given branch, never committing:
   integrate rather than append, keep the voice and the dated labels, never strengthen
   a claim, cut length, placeholders only, test every snippet it adds or changes. Its
   report in `edit/<file>.json` lists, as `cross_file`, each change another file needs.
4. **Review**, two agents: a sanitization review (§6) and an adversarial refutation
   review that tries to refute every changed claim against the evidence, runs every
   added or changed snippet and checks each `cross_file` item. Both write findings
   (blocker / fix / nit).
5. **Fix**, one agent: applies blocker and fix findings, applies or skips (with a
   reason) every `cross_file` item in `edit/*.json`, re-runs the grep and the snippet
   tests, and drafts a `## Unreleased` CHANGELOG entry.
6. **Summary** for the owner: what changed per file, what was not applied and why,
   remaining blockers, and the next steps below.

**Owner gate.** Read `git -C <repo-dir> diff` and the audit files, then commit, push
(from the account that owns the public repo), merge, and turn `Unreleased` into a
version. None of that is automatic.

## 6. Sanitization checklist

Never in the public skill: usernames or home paths, emails, Slurm accounts, allocation
paths, project, lab, company, course or person names, job ids, a study's model sizes,
checkpoints or model set, private file names, secrets, session ids, internal IPs, and
anecdotes that identify one group (its allocation size, its fairshare numbers).
Cluster-wide facts are fine: the login host, partition and QOS names, node names,
driver and module versions, policy URLs, `<placeholder>` tokens.

Grep the added lines of the branch, its untracked files and its commit messages for
these pattern classes, review every hit, and grep for the owner's private values as
fixed strings (the audit dir is private and outside the repo):

```bash
REPO="<repo-dir>"; AUDIT="<audit-dir>"
BASE=$(git -C "$REPO" merge-base HEAD "<default-branch>")
{ git -C "$REPO" diff -U0 "$BASE" | grep '^+' | grep -v '^+++'
  git -C "$REPO" ls-files -o --exclude-standard | while IFS= read -r f; do sed "s|^|+$f: |" "$REPO/$f"; done
  git -C "$REPO" log --format='+commit: %s%n%b' "$BASE..HEAD"; } > "$AUDIT/added.txt"
cat > "$AUDIT/public_patterns.txt" <<'PAT'
/(home|Users|scratch|projects|work)/[A-Za-z0-9]
[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z.]{2,}
(--account[= ]|SBATCH_ACCOUNT=|-A )[A-Za-z0-9]
(^|[^0-9.])[0-9]{7,9}([^0-9.]|$)
(^|[^A-Za-z0-9.])[0-9]+(\.[0-9]+)?[BM]([^A-Za-z0-9]|$)
(^|[^A-Za-z])[A-Z]{2,4}[0-9]{4}([^0-9]|$)
([0-9]{1,3}\.){3}[0-9]{1,3}
(hf_[A-Za-z0-9]{20,}|ghp_|github_pat_|sk-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)
[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}
models--[A-Za-z0-9]
PAT
grep -nE -f "$AUDIT/public_patterns.txt" "$AUDIT/added.txt"
[ -f "$AUDIT/private_patterns.txt" ] && grep -niF -f "$AUDIT/private_patterns.txt" "$AUDIT/added.txt"
```

The classes, in order: personal paths, emails, accounts, job ids (7-9 digits not part
of a version), model sizes (a number then `B` or `M`), course codes, IP addresses,
secrets, session UUIDs, model cache names. Expected hits: `127.0.0.1`, the placeholder
ids `0000000`-`0000001` and `1000001`-`1000012`, public example models, and this file's
own pattern list. A grep misses soft fingerprints, so the reviewer also reads the diff.

## 7. Syncing the installed copy (after the owner approves)

The installed copy (`~/.claude/skills/northeastern-explorer-autonomy`) is not a git
checkout. Sync it additively, then compare:

```bash
SRC="<repo-dir>/northeastern-explorer-autonomy"; DST="$HOME/.claude/skills/northeastern-explorer-autonomy"
diff -rq "$SRC" "$DST"                      # first: is anything in the installed copy yours (filled-in values)?
rsync -a --checksum --exclude=.DS_Store --exclude=__pycache__ --exclude='*.pyc' "$SRC/" "$DST/"   # additive: no --delete
diff -rq "$SRC" "$DST"                      # expect only "Only in $DST" lines for files the repo removed
```

`--checksum`, because rsync's quick check skips a file whose size and mtime match
(tested locally with macOS openrsync: an edit of equal size in the same second was not
copied). Delete "Only in" files by hand once you have read them. Filled-in placeholders
in the installed copy are overwritten: keep site values in project-local files
(`local.env`) instead. Restart Claude Code so it re-reads the skill.
