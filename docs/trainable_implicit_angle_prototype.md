# Prototype: trainable implicit equilibrium and certified quantum-angle correction

## Problem identified by the previous campaigns

The initial frozen implicit layer has random fixed W, U and b.
MRBI increases strict root recovery, but the corresponding
implicit representation need not carry useful label information.
Implicit-only QNN did not gain average balanced accuracy.
Static PCA+certified-MRBI fusion gained **+1.23 percentage
points** on new seeds of the previously used nine tasks,
but failed to improve upon PCA+zero fusion on **three
genuinely new UCI source families**. Both outcomes must
remain visible in the manuscript; do not tune them away.

This new *development prototype* tests a different
mechanism. Train W, U and b from the supervised QNN
loss using implicit-function gradients and encode the
**certified state** as an additive correction to the
**original PCA-QNN's four quantum angles**. It is
not a new benchmark or validation of QNN accuracy.

## Exact mathematical object and differentiation

Original model (unchanged):

\`\`\`text
F(z; x, W, U, b) = z − tanh(W z + U x + b) = 0
J_z = I − D W,   D = diag(1 − tanh(W z + U x + b)^2)
\`\`\`

Here x is the original four-dimensional, training-fitted
PCA representation. The latent z has dimension 16 and
the hard W is initialized in the original non-contractive
spectral-radius-2 regime. W, U and b now become trainable
\`torch.float64\` parameters.

For a scalar loss L and a **strictly successful**
root, write \`J_z^T lambda = dL/dz\` and \`q = D lambda\`.
The custom backward pass implements:

\`\`\`text
dL/dW = q z^T
dL/dU = q x^T
dL/db = q
dL/dx = U^T q
\`\`\`

These are derivatives of the selected **local equilibrium
branch**, not of the root solver, MRBI optimization,
discrete branch selection, or a nonconverged root.
The implicit gradient is allowed only if the original
root solver returns strict success on **the current
learned F** and the minimum singular value of
\`J_z\` is at least **1e-5**. Failed/ill-conditioned
roots contribute zero implicit gradients and cannot
alter QNN angles. A root can change branch across
parameter updates; the local IFT is not a guarantee
of globally differentiable branch selection.

The original \`TorchQNN(4,4,2)\` retains its exact
\`pre\`, PennyLane quantum circuit, quantum weights
and \`post\` readout. The only new pathway is:

\`\`\`text
a(x) = pi * tanh(pre(x)) + alpha * gate(x) * B z*(x)
\`\`\`

\`B\` is a trainable 16→4 linear map without bias;
\`alpha\` is a trainable scalar initialized **exactly 0**.
At \`alpha=0\`, the model must produce **bitwise
identical outputs** to the unmodified PCA-QNN
under the same input, seed and parameters. Nonzero
\`B\` initialization permits alpha to start learning
on the first gradient step; gradients into W/U/b
start once alpha is nonzero and roots pass the
strict numeric gate.

Do **not** standardize z across a test batch or
add 16 extra coordinates to the QNN input. The
original four-feature classical projection and
quantum circuit remain fixed in shape.

## Two matched numerical arms

The prototype uses the identical original F, root
solver, initial W/U/b/QNN/B/alpha and paired
minibatch order in both arms:

- \`zero\`: strict root solve from z0=0.
- \`mrbi\`: the same zero solve, then on its
  failure the previously frozen **one-stage
  smoothed-coarse MRBI** candidate (sigma=0.70,
  smoothing weight=0.75, 10 antithetic probes,
  480 objective/5,760 optimizer-F caps),
  followed by a strict solve of the original F.

No gradients flow through candidate optimization.
Unsuccessful checkpoint candidates are *never*
substituted in the QNN angle. An unsuccessful
zero root is also gated out; its numerical
state may remain in an audit but cannot
influence the QNN output or receive an IFT
gradient. The same sample ID seeds the
same deterministic MRBI probe stream.
Re-solve after **each** parameter update;
never reuse roots computed under earlier
W/U/b as if they were current equilibria.

Matched **initial** parameters isolate the
intervention. Once training begins, the
two arms can learn different W/U/b and
quantum parameters; this is the intended
consequence of different root initialization,
not a second architectural difference.

## Tests and safe first run

The code first checks the original analytic
Jacobian by central differences, verifies
that MRBI and zero modes return exactly
the same zero result when zero succeeds,
and compares all four custom IFT gradients
(W, U, b, x) against central differences
of actual re-solved local roots. It checks
that an ineligible root produces **zero**
implicit-function gradients. With Torch
and PennyLane present it also verifies
bitwise PCA-QNN equality at alpha=0 and
identical initial paired model parameters.

GitHub CI installs CPU Torch and PennyLane
and runs those **synthetic tests only**.
It then dry-runs the pilot without creating
or inspecting any scientific dataset output.

In Ubuntu/WSL:

\`\`\`bash
cd ~
git clone --branch experiment/trainable-implicit-angle-prototype-20260929 \\
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git \\
  mrbi-qnn-trainable-prototype
cd ~/mrbi-qnn-trainable-prototype
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate

python -u scripts/test_trainable_implicit_angle.py
python -u scripts/run_trainable_implicit_angle_pilot.py --dry-run
python -u scripts/run_trainable_implicit_angle_pilot.py --pilot
\`\`\`

The tiny pilot trains **two paired steps**
per arm on eight deterministic *synthetic*
four-feature vectors with toy labels. This
checks real QNN forward/backprop under
changing equilibrium parameters. It writes
\`outputs/trainable_implicit_angle_prototype_v1/
synthetic_pilot.json\` containing per-step
certification and parameter-change
diagnostics. Its toy losses, labels,
strict-root counts and learned parameters
**are not scientific performance results**.
There is intentionally no independent
test accuracy or exploratory sweep.

## Go/no-go before any new classification study

Proceed to a prespecified development-data
experiment only if: the four finite-difference
IFT checks pass; alpha=0 reproduces
PCA-QNN bitwise; both modes start with
identical QNN and W/U/b parameters;
successful roots satisfy F of the
current parameters; IFT masking is
correct on failures; and two optimization
steps complete with finite parameters
and logged root diagnostics. Record
a lack of strict roots or an ineffective
gradient as a prototype limitation,
not an invitation to re-label failed
equilibria as successful.

Only **after** numerical checks pass
should we design a train/dev split for
actual model training, fix its architecture,
stopping rules, controls and genuinely
unseen source-family evaluation. The
earlier Banknote/Ionosphere/Sonar external
results cannot be represented as unseen
validation for this new mechanism. No
quantum advantage or demonstrated
classification improvement is claimed.
