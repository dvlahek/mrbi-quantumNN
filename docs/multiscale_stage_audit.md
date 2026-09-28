# Stage-level audit of MRBI multiscale continuation

This is an exploratory **diagnosis**, not an optimized new method and not
another QNN benchmark. It uses the completed 97-method corrected-continuation
campaign as its only source reference. The existing full-campaign and
final-sigma ablation results remain unchanged.

## Controlled design

For each of 9 tasks and 5 seeds, take the first 8 **training** rows after
the exact benchmark stratified split and PCA transform. The sampling rule is
fixed before inspecting the numerical outcomes, and no class labels enter
the MRBI optimization or stage-level selection. The implicit layer, solver,
full_balanced MRBI profile, antithetic detector probe RNG, start and stage
iteration limits are identical to the completed full campaign. The two arms
are continuation (0.70, 0.25, 0.08, 0.02, 0.02) and repeated final scale
(0.02 at all five stages).

The audit instruments the **existing** SciPy optimizer and records candidate
coordinates, step distances, sigma, residual and Newton terms, detector
ratio and weighted contribution, regularization, Jacobian minimum singular
value and condition number, optimizer convergence, actual optimizer objective
calls and a diagnostic root solve from each intermediate candidate. The
last stage's root result is compared using the existing forced-acceptance
rule. The repeated-final-sigma control uses matching RNG draws.

**Diagnostic work is extra:** computing the term decomposition and running
extra intermediate root solves costs computation. It does *not* count as
optimizing objective calls or feed back into the optimization trajectory.
The test verifies the audited and untouched solver give identical
candidates, objective call counts and subsequent RNG draws. Do not use
instrumented wall time as the runtime comparison from the primary ablation.

## Run on the Ryzen workstation (first job)

Start from a separate checkout of
\`experiment/multiscale-stage-audit-20260928\` and activate the full campaign's
original Python environment:

\`\`\`bash
cd ~
git clone --branch experiment/multiscale-stage-audit-20260928 \\
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git \\
  mrbi-qnn-stage-audit
cd ~/mrbi-qnn-stage-audit
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate
python scripts/test_multiscale_stage_audit.py

FULL="$HOME/mrbi-qnn-full-ryzen/outputs/full_ryzen"
REF="$HOME/mrbi-qnn-full-ryzen/outputs/continuation_v1/main_raw.csv"
python -u scripts/multiscale_stage_audit.py \\
  --reference-raw "$REF" \\
  --reference-env "$FULL/shard1/environment.json" \\
                  "$FULL/shard2/environment.json" \\
                  "$FULL/shard3/environment.json" \\
  --samples-per-job 8 --max-new-jobs 1
\`\`\`

The runner checks the source's 97-method/32-QNN completeness, source shard
Git commits, input environment package versions, audit Git commit and source
CSV SHA-256. Complete jobs are skipped if you rerun the **same command**.
Do not change the audit Git commit or environment mid-campaign.

After reviewing the first job, run the same command without
\`--max-new-jobs 1\` for all 45 jobs. It writes
\`outputs/multiscale_stage_audit_v1/stage_raw.csv\`,
\`paired_sample_results.csv\` and \`stage_summary.csv\` as well as
individual per-job stages/pairs/config files. Use \`--collect-only\` to
rebuild aggregates without new optimizations.

## Interpretation

First check how often earlier scales materially displace the candidate,
the weighted detector contribution relative to the residual term, the
frequency of distinct converged roots, and actual objective calls. This
small deterministic training subset is for diagnostics. Do not treat its
results as a held-out hypothesis test, infer population-wide QNN accuracy,
or select an improved schedule by its test-label performance.

Only then choose one development change (e.g. common antithetic probes across
scales or a genuinely smoothed early objective) and test it with an
**equal-realized-evaluation-budget** final-sigma control. The existing full
and paired-ablation results remain the reference.
