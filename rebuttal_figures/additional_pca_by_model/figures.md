# PCA visualizations by backbone

Layer panels are selected per model as `early / middle / last` based on each residual cache depth.

| Model | Layers | PDF |
|---|---:|---|
| Qwen2.5-7B | 5/14/27 | `qwen-2.5-7b.pdf` |
| Qwen2.5-14B | 5/24/47 | `qwen-2.5-14b.pdf` |
| Qwen3-4B | 5/18/35 | `qwen-3-4b.pdf` |
| Qwen3-8B | 5/18/35 | `qwen-3-8b.pdf` |
| Gemma2-2B | 5/12/25 | `gemma-2-2b-it.pdf` |
| Gemma2-9B | 5/20/41 | `gemma-2-9b-it.pdf` |
| Llama3.1-8B | 5/16/31 | `llama-3.1-8b.pdf` |

The additional directional conflicts are observed in the following table. `FF cos` indicates the cosine similarity between the fairness and factuality vectors. `Fairn norm` and `Fact norm` indicate the norm of the obtained difference-of-mean vectors.

| Backbone | Layers | FF cos early | FF cos mid | FF cos last | Fair norm early/mid/last | Fact norm early/mid/last |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-14B | 5/24/47 | -0.973 | -0.437 | -0.249 | 0.55/7.78/182.89 | 2.55/7.46/67.43 |
| Qwen3-4B | 5/18/35 | -0.772 | -0.483 | -0.266 | 0.18/7.09/65.81 | 0.74/4.79/39.15 |
| Qwen3-8B | 5/18/35 | -0.860 | -0.495 | -0.293 | 0.23/14.59/127.67 | 1.00/9.25/58.39 |
| Gemma2-2B | 5/12/25 | -0.900 | -0.705 | -0.372 | 2.02/26.49/108.57 | 9.01/29.58/44.43 |
| Gemma2-9B | 5/20/41 | -0.954 | -0.599 | -0.185 | 1.89/75.48/207.21 | 8.30/39.04/68.26 |
| Llama3.1-8B | 5/16/31 | -0.771 | -0.553 | -0.438 | 0.11/2.56/8.08 | 0.43/1.36/4.04 |
