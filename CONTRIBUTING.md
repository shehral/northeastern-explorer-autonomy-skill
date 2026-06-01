# Contributing to northeastern-explorer-autonomy

Thanks for your interest in improving this skill. This is a public packaging of a Claude Code skill for operating an HPC cluster autonomously. The contributions that help most are ones that sharpen the *portable* methodology or fix genuine cluster mechanics. Please keep the design principles below intact.

## 1. The design principle to preserve

The skill deliberately separates two layers. Keep them separate.

- **Cluster-specific infrastructure** — connection mechanics, module versions, partition names, offline rules, sbatch anatomy, loader gotchas. This belongs in `references/cluster-setup.md` and the `assets/`.
- **Portable autonomy methodology** — the tiered autonomy model (T0-T3), the five guardrails (G1-G5), the verified submit path, and the self-rescheduling heartbeat. This is the part anyone on any SLURM cluster can reuse.

A change that entangles the two (e.g. hard-coding an Explorer detail into the guardrail logic) will be asked to be refactored. When in doubt, ask: "would this still make sense on a different cluster?" If yes, it's methodology; keep it generic.

**Account-specific values stay as `<placeholder>` tokens.** Everything personal lives in one place: the **Customization Points** table in `SKILL.md`. Login targets, allocations, accounts, env paths, and model IDs are written as `<user>`, `<group>`, `<slurm-account>`, `<conda-env>`, `<org>/<model>`, etc. Explorer-wide constants (module versions, the proxy, partition names) may appear literally, since they are correct for every user of the cluster. Never replace a placeholder with a real value.

## 2. How to propose changes

1. Fork the repo and branch with a clear prefix: `feat/`, `fix/`, `docs/`, `refactor/`.
2. Make your change. Open a PR against the default branch with a short description of what and why.
3. **Keep `SKILL.md` lean — under ~500 lines.** It is the always-loaded entry point and the part the model reads to decide whether to trigger. Push depth into `references/`: detailed procedures, edge cases, copy-paste configs, and long explanations belong in the relevant reference file, with a one-line pointer from `SKILL.md`.
4. If you add a reference file, link it from the "Reference files" section and give it a one-line "read this when..." summary.

## 3. How to test a change

Use the skill-creator eval loop. The prompt set lives at `evals/evals.json` — three task prompts that exercise the core behaviors (overnight self-managing run, heartbeat/Monitor setup, garbage-output and no-op debug), each with an `expected_output` describing what a correct answer must cover.

The benchmark idea, at a high level:

- **With-skill run:** answer each eval prompt in a session that has the skill loaded.
- **Baseline run:** answer the same prompts with no skill loaded.
- Compare both against each prompt's `expected_output`. A good change should hold or improve the with-skill answers (correct offline-mode setup, the budget and fail-streak guardrails, the persistent-Monitor wiring, the end-to-end wiring no-op check) while keeping a clear lift over baseline.

If your change adds or alters a behavior, add or update an eval so the new behavior is covered. Keep eval prompts realistic and free of any private values (same placeholder rule applies).

## 4. No private attribution — placeholders only

This is a public repo. Contributions **must not** add:

- personal names or usernames
- email addresses
- lab, group, organization, company, or team names
- internal project names, hostnames beyond the public cluster login host, paths tied to a specific person, or any other internal reference

The only acceptable way to represent an account-specific value is a generic `<placeholder>` token. If you find any such value in existing content, removing it is a welcome PR.

## 5. Pre-PR checklist

Before opening a PR, confirm:

- [ ] **No private attribution** — no names, usernames, emails, or lab/org/internal references anywhere in the diff.
- [ ] **Placeholders consistent** — account-specific values use `<placeholder>` tokens, named the same way as the Customization Points table; no real values substituted in.
- [ ] **Cross-references valid** — every `references/...` and `assets/...` path mentioned in `SKILL.md` exists, and new reference files are linked back from `SKILL.md`.
- [ ] **`SKILL.md` under ~500 lines** and depth pushed to `references/`.
- [ ] **Description still triggers well** — the `description` frontmatter still clearly fires on the intended cases ("submit this to the cluster", "overnight run", "set up my Explorer environment", etc.); if you changed scope, update it to match.
- [ ] **Evals updated** — new or changed behavior is covered by `evals/evals.json`, and prompts contain no private values.

Keep changes focused, keep the two layers separate, and keep it placeholder-clean. That is what keeps this skill safe to share and easy to reuse.
