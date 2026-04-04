# PRISM

## Data paths

The default paths live in [`config.yml`](./config.yml). Update them to match your local environment before running.

The minimal release expects:

- training caches
- profile caches
- residual caches
- BBQ benchmark files
- DiffAware held-out files

## Usage

Train and evaluate a preset:

```bash
cd /u/yifan40/PRISM
python main.py --mode all --model-name qwen-2.5-7b --run-id your_id
```

Train only:

```bash
python main.py --mode train --model-name qwen-2.5-14b --run-id your_id
```

Evaluate existing artifacts:

```bash
python main.py --mode eval --model-name gemma-2-9b-it --run-id your_id
```

Restrict to one benchmark:

```bash
python main.py --mode all --model-name qwen-2.5-7b --benchmarks bbq --run-id your_id
```
