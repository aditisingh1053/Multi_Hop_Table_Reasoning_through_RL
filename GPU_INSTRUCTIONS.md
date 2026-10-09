# Running the experiments on a GPU

Everything (data preparation, baselines, training, and evaluation on the dev set) runs on the GPU
machine with one command. The laptop only turns the returned results into tables, figures and
the PDF, which takes a few minutes.

Why a GPU helps here: the policy itself is tiny (about 0.76M parameters), but every training
episode is scored by a frozen question-answering reader and all text is embedded by a frozen
encoder. On the laptop CPU one reader call takes about 120 ms; on a T4 GPU it takes a few ms.

**What you upload:** `gpu_bundle.zip`, in the repository root. It contains only code and
configs, under 100 KB. The data (HybridQA plus about 1.2 GB of prepared files) is downloaded and
rebuilt on the GPU machine automatically.

**What you get back:** `results_bundle.zip` (a few hundred MB at most). Put it in the repository
root on the laptop. The last section below explains how to use it.

**Expected time:** roughly 4–8 hours on a T4 for the full suite (within Kaggle's 12-hour limit). That is an estimate;
`results/logs/suite.log` prints the real time of every step. The full suite is:

- 5 seeds of BC, PPO, GRPO and REINFORCE
- 3 seeds of each of the 9 ablations
- evaluation on all 3,466 dev questions
- the second-reader test

To shorten it, set `SEEDS_MAIN=3 SEEDS_ABL=2` before the command (and tell me, so the report
says so).

---

## Option A — Kaggle (recommended: runs in the background, no disconnects)

1. Log in at kaggle.com. GPU access requires a phone-verified account.
2. **Upload the code.** Go to *Create → New Dataset*, upload `gpu_bundle.zip`, name it
   `mhtrl-code` and keep it private.
3. **Create a notebook.** Go to *Create → New Notebook*, then:
   - In the right panel, *Session options*, set *Accelerator* to **GPU T4 x2** (or P100).
   - Set *Internet* to **On**. This is needed to download HybridQA and the two models.
   - Use *Add Input* to add your `mhtrl-code` dataset.
4. Put these four cells in the notebook:

   ```python
   # Cell 1: check the GPU
   !nvidia-smi
   ```

   ```python
   # Cell 2: copy the code into the writable working directory
   import glob, shutil, zipfile, os
   zips = glob.glob('/kaggle/input/*/gpu_bundle.zip')
   if zips:
       zipfile.ZipFile(zips[0]).extractall('/kaggle/working')
   else:   # Kaggle extracted the zip on upload
       shutil.copytree(glob.glob('/kaggle/input/*/mhtrl')[0], '/kaggle/working/mhtrl', dirs_exist_ok=True)
   os.chdir('/kaggle/working/mhtrl')
   !pip install -q pyyaml tqdm
   ```

   ```python
   # Cell 3: 2–3 minute end-to-end check (data preparation runs first, about 10 minutes)
   !SMOKE=1 bash scripts/run_all_gpu.sh
   ```

   ```python
   # Cell 4: full suite
   !bash scripts/run_all_gpu.sh 2>&1 | tail -n 200
   !rm -rf data      # keep the saved output small; results/ has everything needed
   ```

5. Run Cells 1–3 interactively. If Cell 3 prints `SMOKE TEST PASSED`, click
   **Save Version → Save & Run All (Commit)**. Kaggle then runs the whole notebook in the
   background, for up to 12 hours, and you can close the browser.
6. When the version finishes, open it, go to *Output*, and download
   `mhtrl/results_bundle.zip`.

## Option B — Google Colab

Colab can disconnect, so work inside Google Drive. That way a restart resumes where it stopped:
the script skips finished steps.

1. Upload `gpu_bundle.zip` to `My Drive/mhtrl_run/`.
2. Choose *Runtime → Change runtime type → T4 GPU*.
3. Run these cells:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/mhtrl_run
   !unzip -n -q gpu_bundle.zip
   %cd mhtrl
   !pip install -q pyyaml tqdm
   !nvidia-smi
   ```

   ```python
   !SMOKE=1 bash scripts/run_all_gpu.sh
   ```

   ```python
   !bash scripts/run_all_gpu.sh 2>&1 | tail -n 200
   ```

4. Keep the tab open. If the session drops, reconnect and run the three cells again; it
   continues from where it stopped.
5. Download `My Drive/mhtrl_run/mhtrl/results_bundle.zip` when it prints `ALL DONE`.

## Option C — a lab GPU server (SSH)

```bash
unzip gpu_bundle.zip && cd mhtrl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch 2.3.1 from PyPI includes CUDA on Linux
python -c "import torch; print(torch.cuda.is_available())"   # must print True
tmux new -s mhtrl                        # so the run survives a dropped SSH connection
SMOKE=1 bash scripts/run_all_gpu.sh
bash scripts/run_all_gpu.sh 2>&1 | tee run.log
```

Copy `results_bundle.zip` back with `scp`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `git clone` or model download fails on Kaggle | Internet is off: *Session options → Internet → On* |
| `torch.cuda.is_available()` is False | GPU accelerator not selected; on a server, install a CUDA build of torch |
| Session ended halfway | Run the same command again; finished steps are skipped |
| CUDA out of memory (unlikely; models are small) | Run `export MHTRL_READER_BATCH=32` before the command, and tell me |
| Anything else | Send me the last 50 lines of `results/logs/suite.log` |

## After the run: on the laptop

1. Copy `results_bundle.zip` into the repository root
   (`C:\Users\aksgu\Desktop\Multi_Hop_Table_Reasoning_through_RL`).
2. Tell me it is there. I will:
   - replace the laptop's partial `results/` with it;
   - run `python scripts/make_assets.py`, which generates all tables and figures in seconds;
   - write the results, discussion and conclusion chapters;
   - compile `report/main.pdf`.
