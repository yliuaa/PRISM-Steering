# Mitigating the Fairness Steering Tax by Complementing Prompting with Adaptive Activation Steering (PRISM)

*Findings of the Association for Computational Linguistics: EACL 2027*.

**PRISM** (Probe-adaptive Residual Intervention via Subspace Mappings) combines fairness prompting with adaptive activation steering. The prompt helps the model avoid stereotypes, but can also cause it to overlook relevant facts. PRISM uses activation steering to mitigate this loss of accuracy, which we call the **fairness steering tax**.

During training, we compare the model's responses to the same question with and without the fairness instruction. A sparse subspace probe learns to flag examples at risk of a prompted error. On examples where prompting spoiled a correct answer, a low-rank mapping learns a correction from the paired residual states. At inference time, the model reads the fairness-prompted question and applies the mapping only when the probe fires.

We evaluate both sides of this trade-off. BBQ tests whether the model uses the evidence in a question without relying on stereotypes. The descriptive subset of DiffAware tests whether it can still recognize factual demographic differences when they matter to the answer. Across three backbone models, PRISM improves demographic factuality by up to 13.6 points and BBQ accuracy by up to 9.3 points over prompting alone.

## Data splits

The following counts describe the files included in this release.

| Benchmark           | Training sample                                                       | Held-out evaluation                                                                   |
| ------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| BBQ (11 categories) | 100 ambiguous + 100 disambiguated questions per category, 2,200 total | Remaining questions from the same 11 source files (evaluation file stored separately) |
| DiffAware (D1–D4)  | 200 `diffs` + 200 `equals` per dataset, 1,600 total               | 800 `diffs` + 800 `equals` per dataset, 6,400 total                               |

In [training_data](./training_data/), `bbq_samples.pkl` contains the selected BBQ questions. For DiffAware, each `sampled_D*.pkl` has a corresponding `heldout_D*.pkl`. The 70/30 split used for probe training and validation happens later, within the training pool.

## Train and evaluate on DiffAware

Run the commands below from the repository root. The default configuration uses CUDA. You will need a CUDA-capable PyTorch installation and access to the selected model weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp config.yml config.local.yml
```

In `config.local.yml`, replace the following five path values. The two input paths point to files included in the repository. Training creates the profile and residual caches on the first run.

```yaml
training_cache_dir: ./training_data
diffaware_data_root: ./training_data
profile_cache_dir: ./results/cache/profiles
resid_cache_dir: ./results/cache/residuals
results_root: ./results
```

Train the Qwen2.5-7B preset, then evaluate it on the four held-out DiffAware splits:

```bash
python main.py --config config.local.yml --model-name qwen-2.5-7b \
  --run-id diffaware_01 --mode all --benchmarks diffaware
```

Use `--mode train` to stop after fitting the probe and projection. To evaluate an existing run, use `--mode eval` with the same config, model name, and run ID. Specify `--benchmarks diffaware` for DiffAware alone. The default runs both benchmarks.

## Models and experiment settings

| CLI model name    | Model weights                 |
| ----------------- | ----------------------------- |
| `qwen-2.5-7b`   | `Qwen/Qwen2.5-7B-Instruct`  |
| `qwen-2.5-14b`  | `Qwen/Qwen2.5-14B-Instruct` |
| `gemma-2-9b-it` | `google/gemma-2-9b-it`      |

The presets use seed 2026, a 70% training split for fitting the probe and projection, and automatic projection-layer selection on the validation split. Preset values take precedence over matching YAML fields, including the training batch size. The CLI flag `--batch-size` controls benchmark evaluation. See [model presets](./presets.py) for the settings used by each backbone.

Use a distinct run ID for each experiment. When changing the training data or `profile_max_samples`, also choose fresh `profile_cache_dir` and `resid_cache_dir` paths. These caches are shared across run IDs.

## Evaluate on BBQ

We keep the BBQ evaluation file, `bbq_eval.pkl`, outside the repository to limit the download size. If you have a copy, set `bbq_benchmark_path` to its parent directory and run the evaluation with the trained model:

```bash
python main.py --config config.local.yml --model-name qwen-2.5-7b \
  --run-id diffaware_01 --mode eval --benchmarks bbq
```

If only the source JSONL files are present, the evaluator generates a fresh split and writes its training sample to `act_data_root/training_data/bbq_samples.pkl`.

## Results

With the paths above, look under `results/` after a run. The run name combines the run ID, intervention, model name, and method.

| Location under `results/`                           | Contents                                                                                        |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `train_cache/<run_name>/`                           | Trained probe (`probe.pt`), residual projection (`scheme.json`), runtime settings, and logs |
| `train_cache/<run_name>/artifacts/run_summary.json` | Sample counts, probe threshold and detection layer, and training and validation summaries       |
| `benchmark_results/diffaware/<run_name>/`           | Per-split answers and metrics, plus a combined DiffAware summary                                |
| `benchmark_results/bbq/<run_id>/<run_name>.json`    | Per-question BBQ records with predictions, labels, and trigger status                           |

The repository also includes [PCA figures and layer selections](./figures/additional_pca_by_model/figures.md) and [UMAP figures](./figures/umap_by_model/) for seven backbones. The three models listed above have training and evaluation presets in this release.
