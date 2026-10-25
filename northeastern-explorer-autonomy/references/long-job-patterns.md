# Long-job patterns for multi-day GPU campaigns

Field-tested on Explorer during a multi-day, backprop-heavy fitting campaign
whose numbers later survived an adversarial publication audit, plus failures
observed across several projects in 2026-10 (dated where they appear). Each
pattern earned its place by saving real money or catching a real error. They
compose: sharding gives you resumability *and* a free validation instrument; stamping gives you provenance
*and* an audit trail.

## 1. Interleaved sharding, not chunked

Split a corpus across N jobs as `items[shard::N]` (stride), never as contiguous
blocks. Two payoffs:

- **Every shard is corpus-representative**, so any single shard is a cheap
  preview of the full result, and shard-vs-shard agreement becomes a
  *validation instrument for free*: in the campaign, the agreement between two
  disjoint shards went straight into the publication as the stability evidence.
- **A lost shard costs coverage, not a whole document class.** A chunked shard
  that dies takes all of one corpus region with it.

Merge by averaging (or the estimator's natural combine), then **check the
merge**: compare merged-vs-single-shard outputs on a held-out probe set and
record the agreement. If merged ≈ shard, the merge is sound and you have the
receipt.

**Array index maps are per stage.** If two pipeline stages order their units
differently, submitting stage B with stage A's indices computes the wrong units,
and some tasks die on missing inputs (field-observed 2026-10). Before each array
submission, print the index-to-unit map from the manifest the job will read and
check it against the units you intend.

## 2. Resumable + idempotent shard jobs (kill/resubmit freely)

Each shard job:
- writes an **atomic per-item checkpoint** (write temp file, `mv` into place)
  after every item, so a walltime kill costs at most one item;
- **exits 0 immediately if its final artifact exists and is complete** —
  resubmission of an already-done shard is a no-op;
- **checks completeness before its own exit 0**: the expected row count, and no
  `.partial` left behind;
- **exits 75, not 0, when it stops early** on the wall signal or the stop file after
  writing a checkpoint. An early exit 0 reads as done: `kill_invalid_depend` cancels
  the `afternotok` insurance chunk, and an `afterok` successor starts on partial work. The
  full exit-code contract is in the header of `assets/job_template.sbatch`.

This turns SLURM's failure modes (walltime kills, preemption, node flake) from
disasters into noise: the recovery procedure for *anything* is "resubmit the
same array," which an autonomous heartbeat can do safely without diagnosis.
This is the job-level twin of the heartbeat's tick idempotence.

**Smoke outputs need their own roots.** "Exit 0 if the final artifact exists"
turns a smoke that wrote into the full run's path into a silent no-op: the full
job sees a file and skips all the work. Give smokes separate output roots and
run ids (`*_smoke/`), and never pass `--limit`/`--smoke` toward a final output
path. A job that would overwrite rather than resume (one final output, not
per-item shards) refuses to start when that output exists unless `FORCE=1`.
(Recorded as a design rule after review; not yet observed as a failure.)

```bash
OUT=$PROJ/results/$RUN_ID/rows.jsonl            # a smoke uses RUN_ID=<name>_smoke
if [ -e "$OUT" ] && [ "${FORCE:-0}" != 1 ]; then echo "REFUSE: $OUT exists (FORCE=1 redoes it)"; exit 2; fi
# ... work ...
n=$(wc -l < "$OUT")
[ "$n" -eq "$EXPECTED_ROWS" ] || { echo "FATAL: $n rows, expected $EXPECTED_ROWS"; exit 3; }
if [ -n "$(find "$(dirname "$OUT")" -name '*.partial' -print -quit)" ]; then echo "FATAL: .partial left"; exit 3; fi
```

Exit 2 is a pre-flight refusal and 3 a failed self-check (template contract). Not
`find ... | grep -q .`: under `pipefail`, `grep -q` exits on the first match, `find`
dies of SIGPIPE, and the `if` takes the "no partial" branch (tested locally: 3000
`.partial` files went undetected in 5 of 5 runs).

### 2.1 Required outputs first, optional phases last

Three CPU statistics jobs finished their required results within 40 min, then
TIMED OUT in an unbounded exploratory step before writing anything, so the
results were lost three times (field-observed 2026-10). Write and flush every
required artifact first. Put optional or open-ended analysis after it, behind a
skip flag and its own time budget. An agent-chosen wall too short for a
non-resumable job is agent code in G5 (`autonomy-model.md`).

## 3. Budget estimates (for G4 planning and `--time`)

**Fit the wall from the smoke's max per-item time (not the median) × items,
plus fixed start-up, × 1.25.** Smoke per-item rates usually overstate time
(field-observed 2026-10), so they are a safe upper bound for `--time` and
pessimistic for G4:

- Smokes include start-up: a training smoke ran at 11-16k tokens/s against 29k
  in full runs.
- Small batches understate engine throughput: a timing smoke measured about
  8 s per item, while the full vLLM run averaged about 1.2 s, because the
  engine batches more on a large set.

Start-up itself does not follow that rule: a warm-cache re-smoke started in
under 2 min where the cold load had taken 19 min, so a warm smoke can understate
a cold full run by 10+ min. Add the cold-load figure separately
(`cluster-setup.md` §7 Serving engines in jobs). Projections resting on 1-3 timed
calls are thin. Choose
`gpu-short` only when the projection fits its 2 h. Split any job projected
above about 6 h of the 8 h `gpu` wall into resumable pieces (§2).

CPU-only fits deserve a per-cell timing smoke too: one fit ran at about 12 min
per cell on `short` (8 CPUs) against about 4 min for the whole fit on a laptop.
Thread settings are a suspect, not a confirmed cause, so pin them to the
allocation and measure:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

**GPU-type ratios.** On one backprop-heavy fp32 workload, A100-80GB took about 2× the H200 per-shard time.
Measure your own ratio. The ratio matters when H200s are queued out: there is no
separate A100 partition (the GPU type is a gres on the shared GPU partitions), so
pick the type with the procedure in `cluster-setup.md` §5.1. A slower type that
starts now can still win on wall-clock.

## 4. Memory-adaptive batch size (one script, any GPU)

Auto-scale the inner batch dimension from detected GPU memory instead of
hardcoding per-partition scripts:

```python
gb = torch.cuda.get_device_properties(0).total_memory / 1e9
DIM_BATCH = 32 if gb >= 100 else 12 if gb >= 70 else 4   # H200 / A100-80 / smaller
```

Pick the dtype from the device the same way
(`torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32`;
not `torch.cuda.is_bf16_supported()`, which also returns True on T4/V100 through
emulation). The
same sbatch then runs correctly on whatever GPU type has capacity, which is
exactly what an autonomous scheduler needs — type choice becomes a queue-time
decision (`cluster-setup.md` §5.1), not a code change.

## 5. Saturation check before spending the full budget

Before fitting on the full corpus, fit on an interleaved half and measure the
drift when you add more data (compare the estimator's profile at N/2 vs N). In
the campaign, a negligible drift meant a second fit could run on the half corpus
at **half the cost** with evidence — not hope — that it converged.
"How much data is enough" should be a measured decision, and the measurement
costs one cheap comparison.

## 6. Stamp every output artifact (provenance = audit survival)

Every result JSON the campaign wrote embedded, at write time:

- `torch` + `transformers` versions and the **pinned model revision** used;
- the **md5 of the input artifacts** it was computed from (e.g. downstream
  tables recorded the exact md5s of the fitted artifacts they read).

Cost: three lines per writer. Payoff: months later, a publication fact-audit
could tie every public number to the exact artifact + code version that
produced it, and version drift is *detectable* instead of silent. If a number
can't be traced to a stamped file, treat it as if it doesn't exist.

## 7. fp16-store / fp32-load for large artifacts

Big derived tensors (stacks of large square matrices) were stored fp16 and
loaded fp32 — half the storage and transfer cost with
no measurable effect on downstream numbers *for artifacts that are averages of
many samples* (verify once on your own artifact class before adopting).

## 8. Two small env exports that prevent big annoyances

Both are in the bundled `job_template.sbatch`:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — prevents
  fragmentation OOM on multi-hour jobs with many transient allocations
  (the failure appears hours in, after the smoke test passed).
- `HF_HUB_DISABLE_PROGRESS_BARS=1` — a single scripted HF transfer can dump
  ~1 MB of `\r` progress-bar frames into a captured log; with bars off, logs
  stay greppable by a Monitor and readable by agents.

## 9. Model-loading trust check (generalize the §7 lm_head example)

`cluster-setup.md` §7's tie-workaround is one instance of a general rule:
**when a checkpoint's config disagrees with its weights** (missing head, wrong
`tie_word_embeddings`, absent keys), HF may silently "fix" it with random
init. After any nonstandard load path, assert an invariant that only the real
weights satisfy — for a tied head, `cos(lm_head, embed_tokens) > 0.999` — and
fail the job on violation. A wrong head produces plausible-looking garbage
that no downstream metric will flag for you.

## 10. Batch-and-replay instead of a live server

When a local tool drives a model, don't keep an engine waiting on a slow laptop
client: that idles the GPU and invites IdleBot (`cluster-setup.md` §5). Export
every prompt as JSONL keyed by a stable recording key, run one offline batch job
that writes key-to-answer JSONL, and replay the answers locally, failing on any
missing key. GPU utilisation stays high and cluster runs match local settings
byte for byte. Engine start-up, ports and memory share are in
`cluster-setup.md` §7 Serving engines in jobs.

If JS logic must run inside a GPU job, `module load nodejs/v22.11.0` was
available (field-observed 2026-10; Node 22, not newer), so bundle the code for
node22 and check `node --version` before loading the model.
