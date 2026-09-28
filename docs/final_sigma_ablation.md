# Full-campaign final-sigma ablation (fixed profile)

The canonical paired comparison uses the completed full corrected-continuation
97-method campaign (9 tasks × 5 seeds) as the source and runs only one new
control for each dataset/seed. It does not retrain the continuation QNN. The
fixed method is `forced_full_balanced_qnn`; the control is
`final_sigma_repeated_full_balanced_qnn`.

## Controlled difference

Both arms share the same data split, implicit operator, root solver, forced
acceptance rule, antithetic detector probes, QNN architecture and training
seed. The continuation arm optimizes the MRBI objective at
`(0.70, 0.25, 0.08, 0.02)`; the control runs four stages at `0.02`.
Both have a fifth final-scale refinement stage. All five L-BFGS-B stages
have the same respective maximum iteration limits in both arms. The control
pre-generates the nominal scale probe sets to preserve RNG ordering and uses
the same fixed final-scale probe set as the continuation arm.

Equal stage limits do **not** enforce equal objective evaluation counts.
Report the measured objective calls and feature computation time alongside
root success and QNN balanced accuracy. The comparison was designed after
the earlier campaign and remains exploratory. No claims of quantum advantage
or improvement across every MRBI profile follow from it.

## Observed outcome (completed 45 pairs)

The mean per-task QNN balanced-accuracy difference, continuation minus
repeated final scale, is +0.0023: four positive tasks, one tie, four
negative tasks; exploratory two-sided Wilcoxon p=0.5703. The mean root
success difference is +0.00083. Continuation uses more objective calls:
2729 versus 2175 per test sample on average over tasks. The stage schedule
does not show an established independent benefit for this fixed profile.

## Reproduce with the full Ryzen campaign

Use a separate checkout of branch
`experiment/continuation-final-sigma-ablation-20260928` and activate the
*same venv* used to compute the full campaign. Keep the original full
checkout and results unchanged.

```bash
cd ~
git clone --branch experiment/continuation-final-sigma-ablation-20260928 \
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git mrbi-qnn-final-ablation
cd ~/mrbi-qnn-final-ablation
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate
python scripts/test_final_sigma_ablation.py

FULL="$HOME/mrbi-qnn-full-ryzen/outputs/full_ryzen"
REF="$HOME/mrbi-qnn-full-ryzen/outputs/continuation_v1/main_raw.csv"
python scripts/run_final_sigma_ablation.py \
  --reference-raw "$REF" \
  --reference-env "$FULL/shard1/environment.json" \
                  "$FULL/shard2/environment.json" \
                  "$FULL/shard3/environment.json" \
  --out-dir outputs/final_sigma_ablation_full_ryzen_v1 \
  --summarize
```

The runner verifies all 97 methods and 32 QNN rows in each requested source
job, the source shard manifests and package versions, the source raw file
SHA-256 and the source and control Git commits. It skips completed controls.
It writes each control's raw/config/log, then aggregated `main_raw.csv`,
`paired_seed_results.csv`, `dataset_comparison.csv` and
`ablation_statistics.json`.

Pass `--max-new-jobs 1` to run one control first, or
`--max-wall-hours 8` for an approximately eight-hour run. Jobs already
saved are not rerun. For the completed campaign, the three summary outputs
and 45-row control raw should be retained with the full benchmark provenance.
