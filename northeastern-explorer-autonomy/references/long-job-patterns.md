# Long-job patterns for multi-day GPU campaigns

Field-tested on Explorer during a multi-day, backprop-heavy fitting campaign
whose numbers later survived an adversarial publication audit. Each pattern below
earned its place
by saving real money or catching a real error. They compose: sharding gives you
resumability *and* a free validation instrument; stamping gives you provenance
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

## 2. Resumable + idempotent shard jobs (kill/resubmit freely)

Each shard job:
- writes an **atomic per-item checkpoint** (write temp file, `mv` into place)
  after every item, so a walltime kill costs at most one item;
- **exits 0 immediately if its final artifact exists** — resubmission of an
  already-done shard is a no-op.

This turns SLURM's failure modes (walltime kills, preemption, node flake) from
disasters into noise: the recovery procedure for *anything* is "resubmit the
same array," which an autonomous heartbeat can do safely without diagnosis.
This is the job-level twin of the heartbeat's tick idempotence.

## 3. Budget estimates (for G4 planning)

**Estimate from a smoke-tested single shard, multiply, add 20%.** On one
backprop-heavy fp32 workload, A100-80GB took about 2× the H200 per-shard time.
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
(`torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32`). The
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
