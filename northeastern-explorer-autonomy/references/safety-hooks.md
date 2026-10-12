# The safety layer — PreToolUse hooks + per-job verifiers

Autonomy is only safe if the **submit** action is hard to get wrong. This layer makes
the verified path the only path: a hook intercepts the `sbatch` *before* it lands and
runs a per-job verifier that refuses to let a broken experiment burn GPU-hours.

Two complementary mechanisms:
1. **PreToolUse hooks** — intercept a tool call before it executes; block on failure.
2. **Per-job verifiers** — `verify_<job>.sh` scripts the hook dispatches, each
   enforcing experiment-specific invariants.

## PreToolUse hook wiring

In `.claude/settings.json`, declare hooks keyed by the tool they match. Each hook is a
command that receives the tool call as JSON on stdin and exits **0 to allow** / **2 to
block** (a blocking exit prints stderr so you see the reason inline):

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash",
        "hooks": [{ "type": "command",
          "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/pretooluse_hpc_submit_verify.sh",
          "timeout": 180 }] },
      { "matcher": "Edit|Write",
        "hooks": [{ "type": "command",
          "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/pretooluse_artifact_lint.sh",
          "timeout": 30 }] }
    ]
  }
}
```

The submit hook parses the command (stdlib Python, no `jq`), checks whether it's an
`sbatch <file>`, derives the job basename, looks up `scripts/verifiers/verify_<basename>.sh`,
and runs it. If the verifier fails, the hook blocks the submission. A **grace-period**
pattern (allow with a WARN until a cutoff date, then hard-block) lets you roll out a new
verifier without breaking in-flight work.

**Two layers, by design:** the PreToolUse hook catches the submit *before* it lands
(Claude sees the failure immediately); a git **pre-commit** hook is the backstop for
anything that bypasses the tool path. Never `--no-verify` past either.

### Distinguish safety hooks from discipline hooks
- **Safety** (mandatory for autonomy): the `sbatch` verifier — it's what stops a bad job
  from spending real money.
- **Discipline** (optional): artifact/consistency linters that keep your research
  records in sync. Valuable, but a new project can adopt the safety hook alone and add
  discipline hooks later.

## The per-job verifier pattern

Each experiment gets a hand-written `verify_<job>.sh`. Hand-written, because *what's
worth checking* needs human judgment — but they share a small helper library so they
stay short. The reusable `_lib.sh`:

```bash
# verify_outputs_differ FILE1 FILE2 MSG   -> 0 if byte-different, 1 if identical/missing
# verify_python_assertion 'CODE' MSG       -> exit code of `python -c CODE`
# verify_file_nonempty FILE MSG            -> 0 if exists and non-empty
# log_verifier_event NAME STATUS DETAILS   -> append to docs/VERIFIER_LOG.md (audit trail)
```

### The general checklist every verifier enforces
Synthesized across real verifiers — adapt the domain-specific ones:

1. **Existence** — the referenced Python script and sbatch file both exist.
2. **Static structure** — required flags present (`--smoke`), required pre-registration
   constant present (G2), required env in the sbatch (`HF_HUB_OFFLINE=1` and the inherited
   proxy unset — the offline rule, see `cluster-setup.md` §4).
3. **Smoke integrity** — `--smoke` runs without crashing, and produces a *discriminating*
   signal (e.g. a trained direction's norm > 0, an AUC that differs from a shuffled
   baseline). A smoke test that passes trivially isn't a gate.
4. **End-to-end wiring check** — for any with-intervention-vs-baseline
   experiment, assert the two outputs **differ**. This is the highest-value check: it
   catches the "hook was registered but never actually invoked" bug, where outputs are
   identical and a no-op masquerades as a real (null) result. A 4.5h H200 bake-off was
   lost to exactly this before the check existed.
5. **Norm-matched controls (where applicable)** — if you use an inverse/random control,
   verify its vector norm is within ~10% of the real perturbation, so a "control
   perturbs more" result isn't just norm-shock.

### Degrade gracefully, never block on missing infrastructure
A verifier run on a dev box (no GPU, no smoke fixture) should fall back to **static
checks only** and PASS with a WARN — never FAIL for lack of a fixture. In a short CPU
job on the cluster (fixture cached; never on the login node, which kills heavy work
silently), it runs the full smoke with numerical assertions. The rule:
always provide *some* coverage; never make the absence of infrastructure a hard stop.

## Why this layer is what makes autonomy sustainable

The tier model says *what* may be automated; this layer makes the automated thing
*safe to get wrong*. Combined with the ledgers (`autonomy-model.md`):

- a broken experiment is caught by its **verifier** before it submits;
- a runaway spend is caught by the **budget** guardrail (G4);
- a bug-loop is caught by the **failure-streak** rollback (G5);
- a silent no-op is caught by the **end-to-end wiring check**.

Each defends a different failure mode. Together they're why a session can fire jobs
unattended without it being reckless. Start by adopting just the submit verifier + the
two ledgers; add the discipline linters once the safety path is trusted.

## Adapting for a non-LLM / different project
- The offline-mode check is HPC-LLM-specific — drop it if your jobs don't pull
  from a model hub, or replace it with your own "inputs are present and local" check.
- The smoke-fixture path should be configurable (an env var), not hard-coded.
- Consider a small `verifiers.yaml` registry (`{job, checks: [...]}`) so new jobs
  declare what to verify instead of each re-implementing bash — the existing
  hand-written approach is the main thing worth improving for scale.
