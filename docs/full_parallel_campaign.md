# Full corrected-continuation QNN campaign: 12 independent CPU jobs

This branch adds a parallel *driver only*. It calls the original
`experiments/main_qnn_benchmark.py` once for each of the nine datasets
and seeds 0–4 with **exactly the same arguments** as
`scripts/run_continuation_campaign.py`: 97 methods (32 QNN), four qubits,
latent dimension 16, spectral radius 2.0, input scale 1.1, at most 80 samples
per class, 60 QNN epochs, two layers and corrected MRBI continuation.

Each worker is a **separate process** and receives
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=BLIS_NUM_THREADS=
NUMEXPR_NUM_THREADS=VECLIB_MAXIMUM_THREADS=RAYON_NUM_THREADS=
TORCH_NUM_THREADS=1`. The benchmark itself uses `--n-workers 1`.
Thus `--workers 12` launches up to 12 independent single-threaded jobs.
The operating system chooses which physical CPU executes each process.
On Windows/WSL, logical CPU topology may not correspond one-to-one to
physical cores; 12 workers is concurrency, not a guaranteed CPU affinity.

## Installation (WSL Ubuntu)

From a fresh directory, use Python 3.11 if available:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip python3-dev build-essential
cd ~
git clone --branch experiment/full-parallel-12core-20260928 --single-branch \
    https://github.com/dvlahek/mrbi-quantumNN.git mrbi-qnn-full-ryzen
cd ~/mrbi-qnn-full-ryzen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-qnn.txt
python scripts/check_continuation.py
python scripts/test_parallel_continuation.py
```

If `~/mrbi-qnn-full-ryzen` already contains the repository and venv
installed from `main`, fetch and switch to this branch without recloning:

```bash
cd ~/mrbi-qnn-full-ryzen
git fetch origin experiment/full-parallel-12core-20260928
git switch --track origin/experiment/full-parallel-12core-20260928
source .venv/bin/activate
python scripts/check_continuation.py
python scripts/test_parallel_continuation.py
```

If the branch already exists locally, use `git switch
experiment/full-parallel-12core-20260928` instead of `git switch --track`.
Do **not** switch commits or upgrade packages while an output campaign
is in progress; the driver refuses to combine changed environments.

## Start all 45 jobs (no eight-hour cutoff)

```bash
cd ~/mrbi-qnn-full-ryzen
source .venv/bin/activate
mkdir -p outputs/full_parallel_v1
nohup python -u scripts/run_parallel_continuation_campaign.py \
    --workers 12 --summarize \
    > outputs/full_parallel_v1/campaign.log 2>&1 < /dev/null &
echo $! > outputs/full_parallel_v1/campaign.pid
disown
```

Leave Windows and WSL awake. The driver prints start/completion lines to
`campaign.log`, with detailed per-job logs at `outputs/full_parallel_v1/jobs/`.
You can close the terminal after `nohup` and `disown`, but not put
Windows to Sleep. Monitor without stopping the campaign:

```bash
ps -p "$(cat outputs/full_parallel_v1/campaign.pid)" -o pid,etime,stat,cmd
tail -n 30 outputs/full_parallel_v1/campaign.log
```

After a normal stop, the driver collects per-job raw CSVs to
`outputs/full_parallel_v1/main_raw.csv`. With `--summarize` it
creates `main_qnn_results.csv`, `selected_profiles.csv` and
`main_statistics.json` for completed five-seed datasets. At completion
all 45 jobs are available; each job has 97 rows.

To continue, run the **same** command again after the old process
has stopped. Completed valid jobs are skipped. An invalid partial
raw CSV is not silently overwritten; inspect/move it first, then resume.
Use `--collect-only --summarize` to rebuild aggregated outputs after
a crash without starting training again.

The recorded manifest includes Git SHA, Python version, package versions,
full `pip freeze`, worker count and one-thread-per-process settings.
Actual source-data, package, hardware and OS differences between this
new computer and the earlier core campaign should be disclosed when
comparing timings or exact numerical results.

**Resource note:** 12 simultaneous 97-method QNN jobs can require substantial
RAM; if WSL starts swapping or Windows becomes unresponsive, stop the
campaign, then resume using `--workers 6`. Worker-count changes alone
do not change the experiment's per-job configuration.
