# Certified QNN numerical gate — full seed-15–19 report (29 September 2026)

## Provenance, scope and audit

The user-supplied `qnn_raw.csv`, `feature_raw.csv`,
`paired_job_results.csv`, `dataset_summary.csv`,
`qnn_summary.json`, and `environment.json` correspond to
`certified_gate_qnn_seeds15to19_v1`, repository commit
`04c62e85fdcd8e6aa0893ae943fc5c4f4d8f874f`. The stated
environment is WSL2/Python 3.14.4, NumPy 2.5.3, SciPy 1.18.1,
pandas 3.0.6, scikit-learn 1.9.1, torch 2.14.0+cpu and
PennyLane 0.45.1. The experiment covers nine previously studied
datasets × five **fresh seeds 15–19**, 45 complete dataset–seed
jobs, **270 classically simulated QNN readouts**, 4,600 training
inputs and 1,980 test inputs per implicit method, yielding
**32,900 feature records** across the five implicit methods.

Aggregates were independently recomputed from the uploaded CSVs
and match the provided JSON/paired/dataset summary. No duplicate
dataset–seed/method/input keys were found. Paired numerical methods
share exact per-input antithetic probe hashes. The certified coarse
and ungated coarse variants have exactly equal observed per-input
F/J/optimizer/root work; the same holds for certified versus
ungated fine. Certified-feature acceptance follows the strict root
success flag for each observed input. The input arrays, QNN
predictions/logits and per-job SHA manifests were **not uploaded**;
their exact contents and job-level hashes are therefore not
independently checked here.

## Frozen primary result: no QNN balanced-accuracy gain

The prespecified primary outcome was the *paired* mean of
`certified_coarse_qnn − zero_qnn` balanced accuracy over 45
jobs. It is **−0.00037478** (−0.0375 percentage point), not
a positive effect. At the job level, the contrast is positive
in 16 jobs, negative in 18 and tied in 11. Averaged within
each of the nine datasets, it is positive in three and negative
in six. An exploratory, dataset-level two-sided Wilcoxon signed
rank test is `p=0.91015625`; the dataset families include
related digits and wine tasks and should not be treated as
nine unrelated external benchmarks.

| QNN method | Mean BA across 45 jobs |
|---|---:|
| PCA-QNN | 0.97593474 |
| Zero-QNN | 0.90054674 |
| Smoothed coarse, ungated | 0.89373898 |
| Smoothed fine, ungated | 0.89985450 |
| Certified smoothed coarse | 0.90017196 |
| Certified smoothed fine | 0.89580247 |

Secondary descriptive contrasts for certified coarse:
+0.00643298 against ungated coarse,
+0.00436949 against certified fine,
and −0.07576279 against PCA-QNN. The certified versus
ungated coarse difference is positive in 16 jobs, negative
in 17 and tied in 12, and in five of nine per-dataset means
(negative in four). It is not a consistent improvement.
Neither the previous seed-10–14 experiment nor the
new seed-15–19 experiment supports improved mean
QNN balanced accuracy over Zero-QNN.

Per-dataset certified-coarse-minus-zero mean BA:

| Dataset | BA difference |
|---|---:|
| breast_cancer | −0.02500 |
| digits_1_vs_7 | −0.00833 |
| digits_2_vs_7 | +0.03333 |
| digits_3_vs_8 | −0.00833 |
| digits_4_vs_9 | +0.02500 |
| digits_5_vs_6 | −0.02083 |
| wine_0_vs_2 | −0.02889 |
| wine_1_vs_2 | +0.06381 |
| wine_binary | −0.03413 |

These values should be reported together without choosing
only favorable tasks.

## Numerical effect and compute trade-off

On the same **1,980 test inputs**, zero-start root solving
strictly succeeds **1,238** times (62.53%); the fine stage
strictly rescues **77** additional roots, yielding **1,315**
(66.41%); the coarse stage rescues **217**, yielding
**1,455** (73.48%). For paired checkpoint root successes
among the 742 zero-start failures, 60 are successful
in *both* coarse and fine, 157 coarse-only, 17 fine-only,
and 508 neither. The coarse-stage root-rescue effect is
preserved under certification. On the 4,600 training
inputs, the corresponding accepted strict root counts
are zero 2,922, fine 3,095 and coarse 3,421.

However, deployable primitive residual costs on test
inputs are approximately **41.2 F/input** for zero
against **2,216.7 F/input** for coarse and
**2,215.8 F/input** for fine: about **53.8×**
the zero-start cost. The certified gate has *no*
F/J cost advantage against its ungated source
because the full failed candidate must already
be computed to decide on its acceptance.
Its purpose is to prevent uncertified feature
replacement. In this run, the ungated coarse arm
selected unsuccessful candidate features on
**389 test inputs**, while certified coarse retained
the existing zero-root feature on all **525**
zero-start failures where its checkpoint failed.
A zero-root fallback itself need not be a converged
root; the certificate applies only to retained
MRBI substitutions.

These data establish a measured solver-convergence
difference at a large additional cost, not a
cost-effective solver win relative to zero-init,
and not a downstream classification gain.

## Interpretation and publication gate

The prior seed-10–14 failure audit motivated a
fixed, label-free strict-success acceptance rule,
which was then frozen for new seeds 15–19. Its
QNN benefit **did not emerge on the primary
prespecified comparison**. Do not reinterpret
a subgroup gain, a numerically larger root count,
or the exploratory secondary mean as a positive
primary classification result. Do not tune
weights/sigma/seeds or repeat the same nine
datasets until a favorable outcome appears.

For a solver-focused publication, the reproducible
coarse-scale/checkpoint root-rescue mechanism can
be developed with stronger numerical baselines,
including random/multistart, damped Newton, Broyden/
Anderson and *matched observed* F/J and CPU costs,
and validated on genuinely different equilibrium
problem families. Only make a solver efficiency
claim if the method demonstrably improves a
success-versus-compute Pareto comparison.

For a QNN-centered publication, an additional
algorithmic mechanism connecting solver reliability
to discriminative representation quality would
be required, designed using training-only
objectives and evaluated on independent datasets;
the current evidence does not justify such an
improvement claim. The existing original full
QNN benchmark and all negative follow-ups must
remain in the research record. No quantum
advantage is implied: all circuits were
classically simulated.
