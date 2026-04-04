import logging
import json
import pickle as pkl
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import time
from tqdm.auto import tqdm


LOGGER = logging.getLogger(__name__)


def build_cache_signature(cfg: Mapping[str, Any]) -> str:
    model_kwargs = dict(cfg.get("model_kwargs", {}))
    answer_format = str(cfg.get("answer_format", "json_upper_v1"))
    load_mode = str(cfg.get("model_load_mode", "legacy"))
    parts = [
        f"ans-{answer_format}",
        f"load-{load_mode}",
        f"foldln-{int(bool(model_kwargs.get('fold_ln', False)))}",
        f"centerw-{int(bool(model_kwargs.get('center_writing_weights', False)))}",
        f"centeru-{int(bool(model_kwargs.get('center_unembed', False)))}",
    ]
    return re.sub(r"[^\w.-]+", "_", "__".join(parts)).strip("._")


def render_chat_template_with_fallback(
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
    add_generation_prompt: bool = True,
) -> str:
    system_text = str(system_prompt or "")
    user_text = str(user_prompt)
    cached_support = getattr(tokenizer, "_normmod_system_role_supported", None)

    def render_system_user() -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def render_merged_user() -> str:
        merged_user = user_text if not system_text.strip() else f"{system_text.strip()}\n\n{user_text}"
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": merged_user}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    if cached_support is True:
        return render_system_user()
    if cached_support is False:
        return render_merged_user()

    try:
        rendered = render_system_user()
        setattr(tokenizer, "_normmod_system_role_supported", True)
        return rendered
    except Exception as exc:
        if "System role not supported" not in str(exc):
            raise
        setattr(tokenizer, "_normmod_system_role_supported", False)
        return render_merged_user()


def init_logger(log_dir: Path, run_id: str, level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M")
    log_path = log_dir / f"{run_id}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)

def char_to_token_idx(model, text: str, char_idx: int) -> int:
    char_idx = max(0, min(int(char_idx), len(text)))
    return int(model.to_tokens(text[:char_idx]).shape[-1])


def token_labels_from_text(model, text: str, start_tok: int, length: int) -> list[str]:
    token_ids = model.to_tokens(text)[0][start_tok : start_tok + length]
    try:
        return [str(tok) for tok in model.to_str_tokens(token_ids)]
    except Exception:
        return [str(i) for i in range(length)]


def flatten_profile_data(profile_data: Mapping[str, Mapping[str, dict]]) -> Dict[str, dict]:
    flat: Dict[str, dict] = {}
    for bucket in profile_data.values():
        for sid, row in bucket.items():
            flat[str(sid)] = row
    return flat


def extract_options(question_text: str) -> Optional[List[str]]:
    m = re.search(
        r"\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*\(c\)\s*(.*)$",
        question_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m is None:
        return None
    return [m.group(1).strip(), m.group(2).strip(), m.group(3).strip()]



def fetch_ans_toks(model, inputs: Sequence[dict], cand_map: Sequence[str]) -> torch.Tensor:
    ans_tok_list = []
    device = model.cfg.device if hasattr(model, "cfg") else "cuda"
    for row in inputs:
        base_ans_tok = model.to_single_token(cand_map[row["base_pred"]])
        steer_ans_tok = model.to_single_token(cand_map[row["steer_pred"]])
        ans_tok_list.append((base_ans_tok, steer_ans_tok))

    return torch.tensor(ans_tok_list, device=device)


def logit_diff(model, inputs, logits, cand_map: Sequence[str]):
    ans_toks = fetch_ans_toks(model, inputs, cand_map)
    last_token_logit = logits[:, -1, :]
    base_tok, steer_tok = ans_toks[:, 0], ans_toks[:, 1]
    base_logits = last_token_logit.gather(1, base_tok.unsqueeze(1)).squeeze(1)
    steer_logits = last_token_logit.gather(1, steer_tok.unsqueeze(1)).squeeze(1)
    return steer_logits - base_logits


def expand_inputs_by_alpha(inputs, n_intervention_steps: int):
    """
    sample-major ordering:
    [x0@a0, x0@a1, ..., x0@aS, x1@a0, x1@a1, ..., xB@aS]
    """
    expanded_inputs = []
    for row in inputs:
        for _ in range(n_intervention_steps):
            expanded_inputs.append(row)
    return expanded_inputs


def make_alpha_vector(batch_size: int, n_intervention_steps: int, device):
    """
    Returns alpha of shape [batch_size * n_intervention_steps]
    with sample-major ordering.
    """
    alpha_steps = torch.linspace(0.0, 1.0, steps=n_intervention_steps, device=device)
    alpha = alpha_steps.repeat(batch_size)
    return alpha


def build_intervention_hooks(model, component, alpha):
    """
    component examples:
        {"type": "resid_pre", "layer": L0, "hook_name": f"blocks.{L0}.hook_resid_pre"}
        {"type": "attn_head", "layer": l, "head": h, "hook_name": f"blocks.{l}.attn.hook_z"}
        {"type": "mlp_out", "layer": l, "hook_name": f"blocks.{l}.hook_mlp_out"}

    alpha:
        shape [expanded_batch] or [expanded_batch, 1]
        values in [0, 1]
    """
    if not torch.is_tensor(alpha):
        device = model.cfg.device if hasattr(model, "cfg") else "cuda"
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)

    alpha = alpha.flatten()

    comp_type = component["type"]
    hook_name = component["hook_name"]

    def resid_or_mlp_hook(act, hook):
        # act: [batch, seq, d_model]
        assert act.shape[0] == alpha.shape[0], (
            f"Batch mismatch: act batch = {act.shape[0]}, alpha batch = {alpha.shape[0]}"
        )
        out = act.clone()
        out[:, -1, :] = out[:, -1, :] * (1 - alpha[:, None])
        return out

    def attn_head_hook(act, hook):
        # act: [batch, seq, n_heads, d_head] or [batch, seq, n_heads, d_model]
        assert act.shape[0] == alpha.shape[0], (
            f"Batch mismatch: act batch = {act.shape[0]}, alpha batch = {alpha.shape[0]}"
        )
        head_idx = component["head"]
        out = act.clone()
        out[:, -1, head_idx, :] = out[:, -1, head_idx, :] * (1 - alpha[:, None])
        return out

    if comp_type in ("resid_pre", "mlp_out"):
        return [(hook_name, resid_or_mlp_hook)]
    if comp_type == "attn_head":
        return [(hook_name, attn_head_hook)]
    raise ValueError(f"Unsupported component type: {comp_type}")


def component_to_key(component: Dict[str, Any]) -> str:
    ctype = component["type"]
    layer = component["layer"]

    if ctype == "resid_pre":
        return f"L{layer}.resid_pre"
    if ctype == "attn_head":
        return f"L{layer}.H{component['head']}.attn_head"
    if ctype == "attn_out":
        return f"L{layer}.attn_out"
    if ctype == "mlp_out":
        return f"L{layer}.mlp_out"
    raise ValueError(f"Unknown component type: {ctype}")


def build_candidate_components(
    model,
    L0: int,
    split_attn_heads: bool = True,
    attn_hook_name: Optional[str] = None,
):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if attn_hook_name is None:
        attn_hook_name = "hook_z" if split_attn_heads else "hook_attn_out"

    components = []
    components.append({
        "type": "resid_pre",
        "layer": L0,
        "hook_name": f"blocks.{L0}.hook_resid_pre",
    })

    for l in range(L0, n_layers):
        if split_attn_heads:
            for h in range(n_heads):
                components.append({
                    "type": "attn_head",
                    "layer": l,
                    "head": h,
                    "hook_name": f"blocks.{l}.attn.{attn_hook_name}",
                })
        else:
            components.append({
                "type": "attn_out",
                "layer": l,
                "hook_name": f"blocks.{l}.{attn_hook_name}",
            })

        components.append({
            "type": "mlp_out",
            "layer": l,
            "hook_name": f"blocks.{l}.hook_mlp_out",
        })

    return components


def evaluate_intervention(intervention_results):
    """
    h' = (1 - alpha) * h

    alpha = 0 -> no intervention (baseline)
    alpha = 1 -> strongest intervention
    """
    alpha_steps = intervention_results["alpha_steps"]
    component_results = intervention_results["component_results"]

    summary = {}
    ranking = []

    comp_items = list(component_results.items())
    comp_iter = tqdm(comp_items, desc="Evaluating")

    baseline_idx = 0
    intervention_mask = alpha_steps > 0

    for comp_key, grid in comp_iter:
        comp_iter.set_description(f"Evaluating: {comp_key}")

        # grid: [n_samples, n_alpha_steps]
        mean_curve = grid.mean(dim=0)
        std_curve = grid.std(dim=0, unbiased=False)

        # baseline = no intervention = alpha = 0 = first column
        baseline = grid[:, baseline_idx]  # [n_samples]

        # positive means intervention reduced steer-base margin
        reduction_grid = baseline[:, None] - grid
        mean_reduction_curve = reduction_grid.mean(dim=0)

        flip_rate_curve = (grid < 0).float().mean(dim=0)
        flip_gain_curve = flip_rate_curve - flip_rate_curve[baseline_idx]
        score_curve = mean_reduction_curve + flip_gain_curve

        # only choose best alpha among actual interventions
        valid_idxs = torch.where(intervention_mask)[0]
        valid_score_curve = score_curve[valid_idxs]

        best_local_idx = torch.argmax(valid_score_curve).item()
        best_alpha_idx = valid_idxs[best_local_idx].item()

        best_alpha = alpha_steps[best_alpha_idx].item()
        best_mean_delta = mean_curve[best_alpha_idx].item()
        best_flip_rate = flip_rate_curve[best_alpha_idx].item()
        best_mean_reduction = mean_reduction_curve[best_alpha_idx].item()

        # simple ranking score
        score = score_curve[best_alpha_idx].item()

        summary[comp_key] = {
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "mean_reduction_curve": mean_reduction_curve,
            "flip_rate_curve": flip_rate_curve,
            "baseline_alpha": alpha_steps[baseline_idx].item(),
            "baseline_mean_delta": mean_curve[baseline_idx].item(),
            "best_alpha_idx": best_alpha_idx,
            "best_alpha": best_alpha,
            "best_mean_delta": best_mean_delta,
            "best_flip_rate": best_flip_rate,
            "best_mean_reduction": best_mean_reduction,
            "score": score,
        }

        ranking.append({
            "component": comp_key,
            "baseline_mean_delta": mean_curve[baseline_idx].item(),
            "best_alpha": best_alpha,
            "best_mean_delta": best_mean_delta,
            "best_flip_rate": best_flip_rate,
            "best_mean_reduction": best_mean_reduction,
            "score": score,
        })

    ranking = sorted(ranking, key=lambda x: x["score"], reverse=True)

    return {
        "alpha_steps": alpha_steps,
        "summary": summary,
        "ranking": ranking,
    }


def print_top_components(eval_results, top_k: int = 20):
    LOGGER.info("Top %d components", top_k)
    LOGGER.info("%s", "-" * 80)
    for row in eval_results["ranking"][:top_k]:
        LOGGER.info(
            "%-25s | best_alpha=%.3f | best_mean_delta=%.4f | best_flip_rate=%.3f | "
            "best_mean_reduction=%.4f | score=%.4f",
            row["component"],
            row["best_alpha"],
            row["best_mean_delta"],
            row["best_flip_rate"],
            row["best_mean_reduction"],
            row["score"],
        )


def set_random_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return rng


def split_profile_rows(
    inputs: list[dict],
    train_ratio: float,
    rng: np.random.Generator,
    pos_labels: tuple[str, ...] = ("taxed",),
    neg_labels: tuple[str, ...] = ("fixed",),
) -> tuple[list[dict], list[dict], list[str], list[str]]:
    valid_pos_labels = tuple(pos_labels)
    valid_neg_labels = tuple(neg_labels)
    rows = [r for r in inputs if r.get("status") in valid_pos_labels or r.get("status") in valid_neg_labels]
    if not rows:
        raise ValueError("No positive/negative samples found for probe split.")

    train_data, val_data = [], []
    pos_rows = [r for r in rows if r["status"] in valid_pos_labels]
    rng.shuffle(pos_rows)
    n_train_pos = int(len(pos_rows) * train_ratio)
    train_data.extend(pos_rows[:n_train_pos])
    val_data.extend(pos_rows[n_train_pos:])

    neg_rows = [r for r in rows if r["status"] in valid_neg_labels]
    rng.shuffle(neg_rows)
    n_train_neg = int(len(neg_rows) * train_ratio)
    train_data.extend(neg_rows[:n_train_neg])
    val_data.extend(neg_rows[n_train_neg:])

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    train_sample_ids = [r["sample_id"] for r in train_data]
    val_sample_ids = [r["sample_id"] for r in val_data]
    return train_data, val_data, train_sample_ids, val_sample_ids


def collect_residual_streams(
    inputs: list[dict],
    model,
    resid_cache_dir: Path,
    intervention: str,
    model_name: str,
    batch_size: int,
    cache_signature: str,
    pos_labels: tuple[str, ...] = ("taxed",),
):
    resid_cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_name = str(getattr(model, "name", model_name))
    safe_model_name = resolved_model_name.replace("/", "__")
    cache_file = resid_cache_dir / f"{safe_model_name}_{intervention}_{cache_signature}_steered_last_resids.pkl"
    if cache_file.exists():
        LOGGER.info("loading residual streams from cache: %s", cache_file)
        with cache_file.open("rb") as f:
            return pkl.load(f), cache_file

    LOGGER.info("collecting residual streams and caching to: %s", cache_file)
    n_layers = model.cfg.n_layers
    names_filter = lambda name: name.endswith("hook_resid_pre")
    resids = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Collect steered last resid"):
            batch = inputs[i : i + batch_size]
            texts = [x["steered_input"] for x in batch]
            toks = model.to_tokens(texts)
            _, cache = model.run_with_cache(toks, names_filter=names_filter)
            for j, sample in enumerate(batch):
                stack = []
                for layer in range(n_layers):
                    h = cache[f"blocks.{layer}.hook_resid_pre"][j]
                    stack.append(h[-1].detach().to("cpu", dtype=torch.float16))
                resids.append({
                    "sample_id": sample["sample_id"],
                    "label": 1 if sample["status"] in pos_labels else 0,
                    "residual": torch.stack(stack, dim=0),
                })

    with cache_file.open("wb") as f:
        pkl.dump(resids, f)
    return resids, cache_file


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
