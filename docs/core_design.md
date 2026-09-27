# Fixed-after-pilot core comparison: corrected MRBI continuation

The full continuation campaign remains available as `scripts/run_continuation_campaign.py`. It evaluates 97 readout-method combinations per dataset/seed. The core campaign reduces this to 25 rows (eight simulated-QNN variants), while retaining the same implicit operator, root solver, stage sequence, datasets, sample cap, split seeds, four-qubit architecture, 60 QNN epochs, and QNN initialization seed.

## Fixed experimental design

The methods are fixed for the new campaign using their numerical roles. This narrower design was developed after inspecting the earlier full-sweep results and the first corrected-continuation pilot, so it is **pilot-informed, not prospectively preregistered**. The reported method-level comparisons must retain that selection limitation. Every dataset is evaluated at seeds 0–4. Each representation uses the same corresponding QNN training protocol and the same classical control readouts as the full benchmark.

| Representation | QNN | Logistic regression | SVM-RBF | MLP |
| --- | :---: | :---: | :---: | :---: |
| Direct PCA | Yes | Yes | Yes | Yes |
| Zero-initialized implicit | Yes | Yes | Yes | No |
| Forced full_balanced | Yes | Yes | Yes | No |
| Standard-hybrid full_balanced | Yes | Yes | Yes | No |
| Forced qnn_oriented | Yes | Yes | Yes | No |
| Standard-hybrid qnn_oriented | Yes | Yes | Yes | No |
| Forced no_detector | Yes | Yes | Yes | No |
| Standard-hybrid no_detector | Yes | Yes | Yes | No |

`full_balanced` and `no_detector` share the same stage schedule and Newton weight. The latter sets the detector weight to zero and acts as the matched detector ablation. `qnn_oriented` is an additional predefined, detector-enabled profile. The forced mode probes the usefulness of an MRBI candidate. Standard hybrid reports the original trigger-and-accept policy.

The core comparison estimates the effects of fixed solver-aware initializers and the detector term. It does not claim that staged continuation itself outperforms a computationally matched final-scale-only optimizer. Such a claim needs a separate continuation-versus-single-scale experiment, preferably with matched objective-evaluation budgets.

## Execution

Use a separate checkout of branch `experiment/continuation-core-20260927`. Do not run this in the directory of an active 97-method campaign: its environment manifest records a different Git commit and the output files belong to a different experimental design.

```bash
python scripts/test_core_plan.py
python scripts/run_core_campaign.py --max-new-jobs 1
python scripts/run_core_campaign.py
python scripts/summarize_core_campaign.py --require-complete
```

The default output path is `outputs/core_continuation_v1/`. The driver records a Git commit and Python package versions, rejects mixing runs from another commit or design, and writes raw, summary, configuration and logs for each completed dataset/seed job. It resumes by skipping verified 25-row jobs. The corrected implementation has the unchanged label `mrbi_continuation_v1`, and every new row carries `campaign_design=core_fixed_after_pilot_v1`.

The first full 97-method `breast_cancer` seed 0 was performed on an earlier checkout. Do not automatically copy it into this campaign. It can serve as an independent cross-check of the corresponding eight methods after confirming source and environment compatibility.

## Computational scope

In the recorded full `breast_cancer` seed 0, 32 QNN trainings used about 29 minutes and distinct implicit-feature computations about 30 minutes. The eight predefined QNN variants and their corresponding representations occupied about 13 minutes of those measured components. This is an estimate for the core campaign, not a benchmarked promise for every dataset.

The historical nine-task table and its 97-method upper envelope cannot be relabeled as a result of this eight-QNN design. The current paper must report new fixed-method means across seeds, alongside any descriptive best-method envelope, with its selection limitation. The full 97-method campaign remains independently available for a broader comparison.
