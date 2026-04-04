from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

import torch
from transformer_lens import HookedTransformer
import yaml
from dataset import AttributionDataset, JSON_ANSWER_EXAMPLE
from probe import SparseSubspaceProbe
from utils import (
    build_cache_signature,
    collect_residual_streams,
    flatten_profile_data,
    init_logger,
    save_json,
    set_random_seed,
    split_profile_rows,
    char_to_token_idx,
)


CAND_MAP = ("A", "B", "C")
METHOD_RESID_PROJECTION = "resid_projection"
TRAINING_METHODS = {
    METHOD_RESID_PROJECTION,
}
SCHEME_BUILDER_RESID_PROJECTION = "resid_projection"
PROBE_POS_LABELS = ("taxed", "broken")
PROBE_NEG_LABELS = ("fixed", "robust")
RESID_PROJECTION_TRAIN_LABELS = ("taxed",)
RESID_PROJECTION_TRAIN_SUBSET_TAXED = "taxed"
RESID_PROJECTION_TRAIN_SUBSET_WRONG = "wrong"
RESID_PROJECTION_TRAIN_SUBSET_ALL = "all"
LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def resolve_scheme_builder(cfg: dict[str, Any]) -> str:
    return SCHEME_BUILDER_RESID_PROJECTION


def build_training_cfg(
    base_cfg: dict[str, Any],
    *,
    run_id: str,
    intervention: str | None = None,
    model_name: str | None = None,
    model_id: str | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["run_id"] = run_id
    if intervention is not None:
        cfg["intervention"] = intervention
    if model_name is not None:
        cfg["model_name"] = model_name
    if model_id is not None:
        cfg["model_id"] = model_id
    if method is not None:
        cfg["method"] = method
    cfg.setdefault("method", METHOD_RESID_PROJECTION)
    cfg.setdefault("model_load_mode", "no_processing")
    cfg.setdefault("answer_format", "json_upper_v1")
    cfg["scheme_builder"] = resolve_scheme_builder(cfg)
    cfg.setdefault("resid_projection_layer", 22)
    cfg.setdefault("resid_projection_layer_offset", None)
    cfg.setdefault("resid_projection_auto_layer", False)
    cfg.setdefault("resid_projection_layer_select_split", "validation")
    cfg.setdefault("resid_projection_layer_plateau_frac", 0.98)
    cfg.setdefault("resid_projection_rank", 16)
    cfg.setdefault("resid_projection_scale", 1.5)
    cfg.setdefault("resid_projection_apply_mode", "last_token")
    cfg.setdefault("resid_projection_tail_window_radius", 1)
    cfg.setdefault("resid_projection_train_subset", RESID_PROJECTION_TRAIN_SUBSET_TAXED)
    cfg.setdefault("resid_projection_train_steps", 400)
    cfg.setdefault("resid_projection_lr", 1e-2)
    cfg.setdefault("resid_projection_weight_decay", 1e-4)
    cfg.setdefault("resid_projection_cka_weight", 0.0)
    return cfg


def sanitize_name_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w.-]+", "_", text)
    return text.strip("._") or "unknown"


def build_run_name(cfg: dict[str, Any]) -> str:
    return "_".join(
        [
            sanitize_name_part(cfg["run_id"]),
            sanitize_name_part(cfg["intervention"]),
            sanitize_name_part(cfg["model_name"]),
            sanitize_name_part(cfg.get("method", METHOD_RESID_PROJECTION)),
        ]
    )


def build_run_paths(cfg: dict[str, Any]) -> dict[str, Path | str]:
    run_name = build_run_name(cfg)
    run_dir = Path(cfg["results_root"]) / "train_cache" / run_name
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "logs_dir": run_dir / "logs",
        "artifacts_dir": run_dir / "artifacts",
        "settings_path": run_dir / "prism_settings.json",
        "probe_artifact_path": run_dir / "probe.pt",
        "scheme_path": run_dir / "scheme.json",
    }


def init_model(cfg: dict[str, Any]):
    model_kwargs = dict(cfg.get("model_kwargs", {}))
    device = model_kwargs.pop("device", "cuda")
    load_mode = str(cfg.get("model_load_mode", "no_processing")).lower()
    for key in ("center_unembed", "center_writing_weights", "fold_ln"):
        model_kwargs.pop(key, None)

    LOGGER.info("loading model: %s", cfg["model_id"])
    if load_mode == "no_processing":
        model = HookedTransformer.from_pretrained_no_processing(
            cfg["model_id"],
            dtype=torch.bfloat16,
            device=device,
            **model_kwargs,
        )
    else:
        model = HookedTransformer.from_pretrained(
            cfg["model_id"],
            dtype=torch.bfloat16,
            device=device,
            **model_kwargs,
        )
    model.tokenizer.padding_side = "left"
    # GQA
    if getattr(model.cfg, "n_key_value_heads", None) is not None:
        model.cfg.ungroup_grouped_query_attention = True
    return model


def build_dataset_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "intervention": cfg["intervention"],
        "model_name": cfg["model_name"],
        "model_id": cfg["model_id"],
        "training_cache_dir": cfg["training_cache_dir"],
        "model_profile_cache_dir": str(cfg["profile_cache_dir"]),
        "profile_batch_size": int(cfg["profile_batch_size"]),
        "profile_max_samples": cfg.get("profile_max_samples"),
        "answer_format": cfg.get("answer_format", "json_upper_v1"),
        "cache_signature": build_cache_signature(cfg),
    }


def build_probe_label_signature() -> str:
    pos = "-".join(PROBE_POS_LABELS)
    neg = "-".join(PROBE_NEG_LABELS)
    return f"probepos-{pos}__probeneg-{neg}"


def prepare_run_context(cfg: dict[str, Any], model: Any) -> dict[str, Any]:
    cfg = dict(cfg)
    run_paths = build_run_paths(cfg)
    cfg["run_name"] = str(run_paths["run_name"])
    cfg["run_dir"] = str(run_paths["run_dir"])

    run_dir = Path(run_paths["run_dir"])
    artifacts_dir = Path(run_paths["artifacts_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    init_logger(Path(run_paths["logs_dir"]), cfg["run_id"])
    logger = LOGGER

    rng = set_random_seed(int(cfg["seed"]))
    logger.info("starting run with config: %s", cfg)

    return {
        "cfg": cfg,
        "model": model,
        "run_paths": run_paths,
        "run_dir": run_dir,
        "artifacts_dir": artifacts_dir,
        "logger": logger,
        "rng": rng,
    }


def load_and_profile_data(context: dict[str, Any]) -> dict[str, Any]:
    cfg = context["cfg"]
    logger = context["logger"]
    model = context["model"]
    dataset = AttributionDataset(
        model=model,
        config=build_dataset_cfg(cfg),
        cand_map=CAND_MAP,
    )

    logger.info("loading training data...")
    raw_train_data = dataset.load_training_data()
    flatten_train_data = dataset.flatten_training_data(raw_train_data)
    logger.info("profile samples: %d", len(flatten_train_data))

    profile_data, cache_path = dataset.profile_model_intervention(
        flatten_train_data,
        batch_size=int(cfg["profile_batch_size"]),
        max_samples=cfg.get("profile_max_samples"),
    )
    logger.info("profile data cached at: %s", cache_path)
    flattened_raw_profile = flatten_profile_data(profile_data)
    inputs = dataset.build_input_intervention(flattened_raw_profile)
    logger.info("data sample number: %d", len(inputs))

    return {
        "model": model,
        "dataset": dataset,
        "raw_train_data": raw_train_data,
        "flatten_train_data": flatten_train_data,
        "profile_data": profile_data,
        "cache_path": cache_path,
        "inputs": inputs,
    }
def split_and_collect_residuals(context: dict[str, Any], data_state: dict[str, Any]) -> dict[str, Any]:
    cfg = context["cfg"]
    logger = context["logger"]
    rng = context["rng"]
    inputs = data_state["inputs"]
    model = data_state["model"]

    train_data, val_data, train_sample_ids, val_sample_ids = split_profile_rows(
        inputs=inputs,
        train_ratio=float(cfg["train_ratio"]),
        rng=rng,
        pos_labels=PROBE_POS_LABELS,
        neg_labels=PROBE_NEG_LABELS,
    )
    train_pos = sum(int(r["status"] in PROBE_POS_LABELS) for r in train_data)
    train_neg = sum(int(r["status"] in PROBE_NEG_LABELS) for r in train_data)
    val_pos = sum(int(r["status"] in PROBE_POS_LABELS) for r in val_data)
    val_neg = sum(int(r["status"] in PROBE_NEG_LABELS) for r in val_data)
    logger.info("[train] total=%d pos=%d neg=%d", len(train_data), train_pos, train_neg)
    logger.info("[val] total=%d pos=%d neg=%d", len(val_data), val_pos, val_neg)

    resids, resid_cache_path = collect_residual_streams(
        inputs=train_data + val_data,
        model=model,
        resid_cache_dir=Path(cfg["resid_cache_dir"]),
        intervention=cfg["intervention"],
        model_name=cfg["model_name"],
        batch_size=int(cfg["batch_size"]),
        cache_signature=f"{build_cache_signature(cfg)}__{build_probe_label_signature()}",
        pos_labels=PROBE_POS_LABELS,
    )
    train_sample_id_set = set(train_sample_ids)
    val_sample_id_set = set(val_sample_ids)
    train_resids = [r for r in resids if r["sample_id"] in train_sample_id_set]
    val_resids = [r for r in resids if r["sample_id"] in val_sample_id_set]
    logger.info("[resid split] train=%d val=%d", len(train_resids), len(val_resids))

    return {
        "train_data": train_data,
        "val_data": val_data,
        "train_sample_ids": train_sample_ids,
        "val_sample_ids": val_sample_ids,
        "resids": resids,
        "resid_cache_path": resid_cache_path,
        "train_resids": train_resids,
        "val_resids": val_resids,
    }

def fit_and_calibrate_probe(
    context: dict[str, Any],
    data_state: dict[str, Any],
    split_state: dict[str, Any],
) -> dict[str, Any]:
    cfg = context["cfg"]
    artifacts_dir = context["artifacts_dir"]
    train_resids = split_state["train_resids"]
    val_resids = split_state["val_resids"]
    train_rows = split_state["train_data"]
    val_rows = split_state["val_data"]

    def attach_row_metadata(resids, rows):
        row_map = {str(r["sample_id"]): r for r in rows}
        out = []
        for resid in resids:
            merged = dict(resid)
            row = row_map.get(str(resid["sample_id"]))
            if row is not None:
                for key in ("status", "domain_flag", "dataset_flag", "source_flag", "subset_flag"):
                    if key in row:
                        merged[key] = row.get(key)
            out.append(merged)
        return out

    probe_type = str(cfg.get("probe_type", "sparse_subspace")).lower().strip()
    if probe_type != "sparse_subspace":
        raise ValueError(f"Unsupported probe_type for PRISM release: {probe_type}")
    train_resids = attach_row_metadata(train_resids, train_rows)
    val_resids = attach_row_metadata(val_resids, val_rows)
    if len(val_resids) < 2:
        raise ValueError("SparseSubspaceProbe requires at least 2 validation residuals for agg/cal split.")
    split_idx = max(1, len(val_resids) // 2)
    if split_idx >= len(val_resids):
        split_idx = len(val_resids) - 1
    val_agg_resids = val_resids[:split_idx]
    val_cal_resids = val_resids[split_idx:]
    logger = context["logger"]
    logger.info(
        "[sparse probe val split] agg=%d cal=%d (from total=%d)",
        len(val_agg_resids),
        len(val_cal_resids),
        len(val_resids),
    )
    probe = SparseSubspaceProbe(
        train_data=train_resids,
        val_data=val_agg_resids,
        val_cal_data=val_cal_resids,
        pos_weight=float(cfg.get("probe_pos_weight", 0.25)),
        num_subspaces=int(cfg.get("probe_num_subspaces", 2)),
        subspace_rank=int(cfg.get("probe_subspace_rank", 32)),
        orthogonality_weight=float(cfg.get("probe_orthogonality_weight", 1e-2)),
        aggregator_l1_weight=float(cfg.get("probe_aggregator_l1_weight", 1e-3)),
        epochs=int(cfg.get("probe_sparse_epochs", 80)),
        lr=float(cfg.get("probe_sparse_lr", 1e-3)),
        batch_size=int(cfg.get("probe_sparse_batch_size", 256)),
        weight_decay=float(cfg.get("probe_sparse_weight_decay", 1e-4)),
    )
    n_layers = int(train_resids[0]["residual"].shape[0])
    probe.fit_scan_layers(
        layer_range=range(n_layers),
        metric=str(cfg["probe_metric"]),
        backend="torch",
    )
    probe_calibration = probe.calibrate_threshold(
        metric=str(cfg.get("probe_threshold_metric", "f1")),
        split="val_cal",
    )
    save_json(
        artifacts_dir / "probe_results.json",
        {
            "detection_layer": probe.detection_layer,
            "probe_type": probe_type,
            "probe_results": probe.probe_results,
            "calibration": probe_calibration,
        },
    )

    return {
        "probe": probe,
        "probe_calibration": probe_calibration,
    }
def collect_last_token_features_for_rows(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    layer: int,
    text_key: str,
    batch_size: int,
) -> torch.Tensor:
    texts = [str(row[text_key]) for row in rows]
    return collect_last_token_features_for_texts(
        model,
        texts,
        layer=layer,
        batch_size=batch_size,
    )


def collect_last_token_features_for_texts(
    model: Any,
    texts: list[str],
    *,
    layer: int,
    batch_size: int,
) -> torch.Tensor:
    hook_name = f"blocks.{layer}.hook_resid_pre"
    feats: list[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        cache: dict[str, torch.Tensor] = {}

        def capture_hook(act, hook):
            cache["act"] = act.detach()
            return act

        with torch.inference_mode():
            with model.hooks(fwd_hooks=[(hook_name, capture_hook)]):
                _ = model(batch_texts)
        feats.append(cache["act"][:, -1, :].cpu())

    return torch.cat(feats, dim=0)


def locate_tail_bundle_positions(
    model: Any,
    text: str,
    *,
    window_radius: int,
) -> list[int]:
    n_tokens = int(model.to_tokens(text).shape[-1])
    if n_tokens <= 0:
        return []
    format_start_char = text.find(JSON_ANSWER_EXAMPLE)
    if format_start_char == -1:
        return [n_tokens - 1]
    format_start_tok = int(char_to_token_idx(model, text, format_start_char))
    start_tok = max(0, format_start_tok - max(0, int(window_radius)))
    return list(range(start_tok, n_tokens))


def collect_tail_bundle_features_for_rows(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    layer: int,
    text_key: str,
    batch_size: int,
    window_radius: int,
) -> torch.Tensor:
    texts = [str(row[text_key]) for row in rows]
    return collect_tail_bundle_features_for_texts(
        model,
        texts,
        layer=layer,
        batch_size=batch_size,
        window_radius=window_radius,
    )


def collect_tail_bundle_features_for_texts(
    model: Any,
    texts: list[str],
    *,
    layer: int,
    batch_size: int,
    window_radius: int,
) -> torch.Tensor:
    hook_name = f"blocks.{layer}.hook_resid_pre"
    feats: list[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        cache: dict[str, torch.Tensor] = {}

        def capture_hook(act, hook):
            cache["act"] = act.detach()
            return act

        with torch.inference_mode():
            with model.hooks(fwd_hooks=[(hook_name, capture_hook)]):
                _ = model(batch_texts)
        batch_act = cache["act"].cpu()
        for i, text in enumerate(batch_texts):
            token_positions = locate_tail_bundle_positions(
                model,
                str(text),
                window_radius=window_radius,
            )
            if token_positions:
                feats.append(batch_act[i, token_positions, :])

    if not feats:
        return torch.empty((0, int(model.cfg.d_model)), dtype=torch.float32)
    return torch.cat(feats, dim=0)
def collect_all_last_token_features_for_rows(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    text_key: str,
    batch_size: int,
) -> torch.Tensor:
    texts = [str(row[text_key]) for row in rows]
    n_layers = int(model.cfg.n_layers)
    names_filter = lambda name: name.endswith("hook_resid_pre")
    feats: list[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        toks = model.to_tokens(batch_texts)
        with torch.inference_mode():
            _, cache = model.run_with_cache(toks, names_filter=names_filter)
        stack = []
        for layer in range(n_layers):
            stack.append(cache[f"blocks.{layer}.hook_resid_pre"][:, -1, :].detach().to("cpu", dtype=torch.float32))
        feats.append(torch.stack(stack, dim=1))

    return torch.cat(feats, dim=0)


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean(dim=0, keepdim=True)
    x_tx = x_c.T @ x_c
    y_ty = y_c.T @ y_c
    x_ty = x_c.T @ y_c
    numerator = torch.sum(x_ty * x_ty)
    denom = torch.linalg.norm(x_tx, ord="fro") * torch.linalg.norm(y_ty, ord="fro")
    return numerator / denom.clamp_min(float(eps))


def fit_low_rank_resid_projection(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    rank: int,
    steps: int,
    lr: float,
    weight_decay: float,
    cka_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    x = x_train.float()
    y = y_train.float()
    _, d = x.shape
    rank = max(1, min(int(rank), d))
    cka_weight = float(cka_weight)

    u = torch.zeros(d, rank, dtype=torch.float32, requires_grad=True)
    v = torch.zeros(d, rank, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(d, dtype=torch.float32, requires_grad=True)
    torch.nn.init.normal_(u, std=0.02)
    torch.nn.init.normal_(v, std=0.02)

    optimizer = torch.optim.Adam([u, v, b], lr=float(lr), weight_decay=float(weight_decay))
    mse_first = None
    mse_last = None
    cka_first = None
    cka_last = None
    total_first = None
    total_last = None
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        correction = (x @ v) @ u.T + b
        pred = x - correction
        mse_loss = torch.mean((pred - y) ** 2)
        cka_loss = 1.0 - linear_cka(pred, y)
        loss = mse_loss
        if cka_weight > 0.0:
            loss = loss + cka_weight * cka_loss
        mse_val = float(mse_loss.detach().cpu())
        cka_val = float(cka_loss.detach().cpu())
        total_val = float(loss.detach().cpu())
        if mse_first is None:
            mse_first = mse_val
            cka_first = cka_val
            total_first = total_val
        mse_last = mse_val
        cka_last = cka_val
        total_last = total_val
        loss.backward()
        optimizer.step()

    return {
        "u": u.detach().cpu(),
        "v": v.detach().cpu(),
        "b": b.detach().cpu(),
        "train_mse_first": mse_first,
        "train_mse_last": mse_last,
        "train_cka_loss_first": cka_first,
        "train_cka_loss_last": cka_last,
        "train_total_loss_first": total_first,
        "train_total_loss_last": total_last,
    }


def apply_low_rank_resid_projection(
    x: torch.Tensor,
    params: dict[str, Any],
    scale: float,
) -> torch.Tensor:
    orig_dtype = x.dtype
    u = torch.as_tensor(params["u"], device=x.device, dtype=torch.float32)
    v = torch.as_tensor(params["v"], device=x.device, dtype=torch.float32)
    b = torch.as_tensor(params["b"], device=x.device, dtype=torch.float32)
    x_f = x.to(torch.float32)
    correction = (x_f @ v) @ u.T + b
    out = x_f - float(scale) * correction
    return out.to(orig_dtype)


def evaluate_resid_projection(
    model: Any,
    cand_map: tuple[str, ...],
    rows: list[dict[str, Any]],
    *,
    layer: int,
    params: dict[str, Any],
    scale: float,
    batch_size: int,
    split_name: str,
    apply_mode: str = "last_token",
    tail_window_radius: int = 1,
) -> dict[str, Any]:
    texts = [str(row["steered_input"]) for row in rows]
    cand_tokens = [model.to_single_token(c) for c in cand_map]
    probs_base: list[float] = []
    probs_steer: list[float] = []
    margins: list[float] = []
    pred_is_base = 0
    pred_is_correct = 0
    hook_name = f"blocks.{layer}.hook_resid_pre"

    for start in range(0, len(texts), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]
        batch_tail_positions = None
        if str(apply_mode).lower() == "tail_bundle":
            batch_tail_positions = [
                locate_tail_bundle_positions(
                    model,
                    str(row["steered_input"]),
                    window_radius=tail_window_radius,
                )
                for row in batch_rows
            ]

        def hook_fn(act, hook):
            out = act.clone()
            if str(apply_mode).lower() == "tail_bundle":
                for i, token_positions in enumerate(batch_tail_positions or []):
                    if token_positions:
                        out[i, token_positions, :] = apply_low_rank_resid_projection(
                            out[i, token_positions, :],
                            params,
                            scale,
                        )
            else:
                out[:, -1, :] = apply_low_rank_resid_projection(out[:, -1, :], params, scale)
            return out

        with torch.inference_mode():
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                logits = model(batch_texts)
        cand_logits = logits[:, -1, cand_tokens]
        probs = cand_logits.softmax(dim=-1)
        preds = cand_logits.argmax(dim=-1)
        for local_idx, row in enumerate(batch_rows):
            base_idx = int(row["base_pred"])
            steer_idx = int(row["steer_pred"])
            probs_base.append(float(probs[local_idx, base_idx].item()))
            probs_steer.append(float(probs[local_idx, steer_idx].item()))
            margins.append(float((cand_logits[local_idx, base_idx] - cand_logits[local_idx, steer_idx]).item()))
            pred_is_base += int(int(preds[local_idx].item()) == base_idx)
            pred_is_correct += int(int(preds[local_idx].item()) == int(row["true_idx"]))

    n = max(1, len(rows))
    return {
        "split": split_name,
        "layer": int(layer),
        "scale": float(scale),
        "mean_p_base": sum(probs_base) / n,
        "mean_p_steer": sum(probs_steer) / n,
        "mean_base_minus_steer_margin": sum(margins) / n,
        "flip_to_base_rate": pred_is_base / n,
        "flip_to_correct_rate": pred_is_correct / n,
    }


def select_resid_projection_layer_by_spectrum(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    detection_layer: int,
    n_layers: int,
    batch_size: int,
    split_name: str,
    rank: int,
    plateau_frac: float,
) -> tuple[int, list[dict[str, Any]]]:
    x_steer = collect_all_last_token_features_for_rows(
        model,
        rows,
        text_key="steered_input",
        batch_size=batch_size,
    )
    x_base = collect_all_last_token_features_for_rows(
        model,
        rows,
        text_key="base_input",
        batch_size=batch_size,
    )
    layer_range = range(int(detection_layer), int(n_layers))
    sweep: list[dict[str, Any]] = []

    for layer in layer_range:
        diff = (x_steer[:, layer, :] - x_base[:, layer, :]).float()
        fro_norm = float(torch.linalg.matrix_norm(diff, ord="fro").item())
        if diff.shape[0] == 0 or diff.shape[1] == 0:
            singular_values = torch.empty(0, dtype=torch.float32)
        else:
            singular_values = torch.linalg.svdvals(diff)
        energy = singular_values.square()
        total_energy = float(energy.sum().item())
        top_r_energy = float(energy[: max(1, min(int(rank), energy.numel()))].sum().item()) if energy.numel() > 0 else 0.0
        explained_energy_rank_r = (top_r_energy / total_energy) if total_energy > 0 else 0.0
        if total_energy > 0:
            probs = energy / energy.sum()
            entropy = float((-(probs * (probs + 1e-12).log())).sum().item())
            effective_rank = float(torch.exp(torch.tensor(entropy)).item())
        else:
            effective_rank = 0.0
        sweep.append(
            {
                "split": split_name,
                "layer": int(layer),
                "fro_norm": fro_norm,
                "top_r_explained_energy": explained_energy_rank_r,
                "effective_rank": effective_rank,
                "n_samples": int(diff.shape[0]),
                "hidden_dim": int(diff.shape[1]),
            }
        )

    max_fro = max((row["fro_norm"] for row in sweep), default=0.0)
    threshold = float(plateau_frac) * max_fro
    selected = None
    for row in sweep:
        if row["fro_norm"] >= threshold:
            selected = row
            break
    if selected is None:
        selected = max(sweep, key=lambda row: row["fro_norm"])
    return int(selected["layer"]), sweep


def build_resid_projection_state(
    context: dict[str, Any],
    data_state: dict[str, Any],
    split_state: dict[str, Any],
    probe_state: dict[str, Any],
) -> dict[str, Any]:
    cfg = context["cfg"]
    logger = context["logger"]
    model = data_state["model"]
    train_subset = str(cfg.get("resid_projection_train_subset", RESID_PROJECTION_TRAIN_SUBSET_TAXED)).lower()
    if train_subset == RESID_PROJECTION_TRAIN_SUBSET_TAXED:
        train_rows = [r for r in split_state["train_data"] if r["status"] in RESID_PROJECTION_TRAIN_LABELS]
        val_rows = [r for r in split_state["val_data"] if r["status"] in RESID_PROJECTION_TRAIN_LABELS]
    elif train_subset == RESID_PROJECTION_TRAIN_SUBSET_WRONG:
        train_rows = [r for r in split_state["train_data"] if r["status"] in PROBE_POS_LABELS]
        val_rows = [r for r in split_state["val_data"] if r["status"] in PROBE_POS_LABELS]
    elif train_subset == RESID_PROJECTION_TRAIN_SUBSET_ALL:
        train_rows = list(split_state["train_data"])
        val_rows = list(split_state["val_data"])
    else:
        raise ValueError(f"Unsupported resid_projection_train_subset: {train_subset}")
    n_layers = int(split_state["train_resids"][0]["residual"].shape[0])
    batch_size = int(cfg["batch_size"])
    layer = int(cfg.get("resid_projection_layer", 22))
    layer_offset = cfg.get("resid_projection_layer_offset")
    auto_layer = bool(cfg.get("resid_projection_auto_layer", False))
    layer_selection_meta: dict[str, Any] | None = None
    if auto_layer:
        select_split = str(cfg.get("resid_projection_layer_select_split", "validation")).lower()
        select_rows = val_rows if select_split == "validation" else train_rows
        plateau_frac = float(cfg.get("resid_projection_layer_plateau_frac", 0.98))
        layer, layer_sweep = select_resid_projection_layer_by_spectrum(
            model,
            select_rows,
            detection_layer=int(probe_state["probe"].detection_layer),
            n_layers=n_layers,
            batch_size=batch_size,
            split_name=select_split,
            rank=int(cfg.get("resid_projection_rank", 16)),
            plateau_frac=plateau_frac,
        )
        layer_selection_meta = {
            "mode": "spectral_last_token_diff",
            "split": select_split,
            "detection_layer": int(probe_state["probe"].detection_layer),
            "selected_layer": int(layer),
            "plateau_frac": plateau_frac,
            "candidates": layer_sweep,
        }
        selected_metrics = next(item for item in layer_sweep if int(item["layer"]) == int(layer))
        logger.info(
            "[resid_projection auto-layer] split=%s detection=%d selected=%d fro_norm=%.4f top_r_energy=%.4f eff_rank=%.2f plateau_frac=%.3f",
            select_split,
            int(probe_state["probe"].detection_layer),
            int(layer),
            selected_metrics["fro_norm"],
            selected_metrics["top_r_explained_energy"],
            selected_metrics["effective_rank"],
            plateau_frac,
        )
    elif layer_offset is not None:
        layer = int(probe_state["probe"].detection_layer) + int(layer_offset)
    if layer < int(probe_state["probe"].detection_layer):
        raise ValueError(
            "resid_projection intervention layer cannot be earlier than the detection layer "
            f"(detection={int(probe_state['probe'].detection_layer)}, intervention={int(layer)})"
        )
    if layer >= n_layers:
        clamped = n_layers - 1
        logger.warning(
            "Requested resid_projection layer %s exceeds n_layers=%s; clamping to %s",
            layer,
            n_layers,
            clamped,
        )
        layer = clamped
    rank = int(cfg.get("resid_projection_rank", 16))
    scale = float(cfg.get("resid_projection_scale", 1.5))
    apply_mode = str(cfg.get("resid_projection_apply_mode", "last_token")).lower()
    tail_window_radius = int(cfg.get("resid_projection_tail_window_radius", 1))
    if apply_mode == "tail_bundle":
        x_train = collect_tail_bundle_features_for_rows(
            model,
            train_rows,
            layer=layer,
            text_key="steered_input",
            batch_size=batch_size,
            window_radius=tail_window_radius,
        )
        y_train = collect_tail_bundle_features_for_rows(
            model,
            train_rows,
            layer=layer,
            text_key="base_input",
            batch_size=batch_size,
            window_radius=tail_window_radius,
        )
    else:
        x_train = collect_last_token_features_for_rows(
            model,
            train_rows,
            layer=layer,
            text_key="steered_input",
            batch_size=batch_size,
        )
        y_train = collect_last_token_features_for_rows(
            model,
            train_rows,
            layer=layer,
            text_key="base_input",
            batch_size=batch_size,
        )
    params = fit_low_rank_resid_projection(
        x_train,
        y_train,
        rank=rank,
        steps=int(cfg.get("resid_projection_train_steps", 400)),
        lr=float(cfg.get("resid_projection_lr", 1e-2)),
        weight_decay=float(cfg.get("resid_projection_weight_decay", 1e-4)),
        cka_weight=float(cfg.get("resid_projection_cka_weight", 0.0)),
    )
    logger.info(
        "[resid_projection train loss] rank=%d cka_weight=%.4f total %.6f -> %.6f | mse %.6f -> %.6f | cka %.6f -> %.6f",
        rank,
        float(cfg.get("resid_projection_cka_weight", 0.0)),
        float(params.get("train_total_loss_first", float("nan"))),
        float(params.get("train_total_loss_last", float("nan"))),
        float(params.get("train_mse_first", float("nan"))),
        float(params.get("train_mse_last", float("nan"))),
        float(params.get("train_cka_loss_first", float("nan"))),
        float(params.get("train_cka_loss_last", float("nan"))),
    )

    train_eval = evaluate_resid_projection(
        model,
        CAND_MAP,
        train_rows,
        layer=layer,
        params=params,
        scale=scale,
        batch_size=batch_size,
        split_name="train",
        apply_mode=apply_mode,
        tail_window_radius=tail_window_radius,
    )
    val_eval = evaluate_resid_projection(
        model,
        CAND_MAP,
        val_rows,
        layer=layer,
        params=params,
        scale=scale,
        batch_size=batch_size,
        split_name="validation",
        apply_mode=apply_mode,
        tail_window_radius=tail_window_radius,
    )
    logger.info(
        "[resid_projection val] layer=%d rank=%d scale=%.3f flip_to_base=%.4f margin=%.4f",
        layer,
        rank,
        scale,
        val_eval["flip_to_base_rate"],
        val_eval["mean_base_minus_steer_margin"],
    )
    summary = {
        "builder": "resid_projection",
        "target_mode": apply_mode,
        "layer": layer,
        "layer_offset": None if layer_offset is None else int(layer_offset),
        "auto_layer": auto_layer,
        "rank": rank,
        "scale": scale,
        "tail_window_radius": tail_window_radius if apply_mode == "tail_bundle" else None,
        "train_subset": train_subset,
        "train_steps": int(cfg.get("resid_projection_train_steps", 400)),
        "lr": float(cfg.get("resid_projection_lr", 1e-2)),
        "weight_decay": float(cfg.get("resid_projection_weight_decay", 1e-4)),
        "cka_weight": float(cfg.get("resid_projection_cka_weight", 0.0)),
        "train_total_loss_first": params.get("train_total_loss_first"),
        "train_total_loss_last": params.get("train_total_loss_last"),
        "train_mse_first": params.get("train_mse_first"),
        "train_mse_last": params.get("train_mse_last"),
        "train_cka_loss_first": params.get("train_cka_loss_first"),
        "train_cka_loss_last": params.get("train_cka_loss_last"),
    }
    scheme = {
        "selected_components": [],
        "all_ranked_components": [],
        "projection": {
            "target_mode": apply_mode,
            "apply_mode": apply_mode,
            "layer": layer,
            "layer_offset": None if layer_offset is None else int(layer_offset),
            "auto_layer": auto_layer,
            "rank": rank,
            "scale": scale,
            "tail_window_radius": tail_window_radius if apply_mode == "tail_bundle" else None,
            "train_subset": train_subset,
            "u": params["u"].tolist(),
            "v": params["v"].tolist(),
            "b": params["b"].tolist(),
        },
    }
    if layer_selection_meta is not None:
        scheme["projection"]["layer_selection"] = layer_selection_meta
    return {
        "train_taxed_data": train_rows,
        "val_taxed_data": val_rows,
        "components": [],
        "gate_attr": None,
        "scheme": scheme,
        "train_intervention_results": summary,
        "val_intervention_results": summary,
        "train_eval": train_eval,
        "val_eval": val_eval,
        "val_joint_eval": val_eval,
        "comparison": summary if layer_selection_meta is None else {**summary, "layer_selection": layer_selection_meta},
        "builder": "resid_projection",
    }


def save_training_artifacts(
    context: dict[str, Any],
    data_state: dict[str, Any],
    split_state: dict[str, Any],
    probe_state: dict[str, Any],
    scheme_state: dict[str, Any],
) -> dict[str, Any]:
    cfg = context["cfg"]
    run_paths = context["run_paths"]
    run_dir = context["run_dir"]
    artifacts_dir = context["artifacts_dir"]
    logger = context["logger"]
    probe = probe_state["probe"]
    scheme = scheme_state["scheme"]
    scheme_builder = resolve_scheme_builder(cfg)
    train_intervention_results = scheme_state["train_intervention_results"]
    val_intervention_results = scheme_state["val_intervention_results"]
    train_eval = scheme_state["train_eval"]
    val_eval = scheme_state["val_eval"]
    val_joint_eval = scheme_state["val_joint_eval"]
    comparison = scheme_state["comparison"]
    cache_path = data_state["cache_path"]
    resid_cache_path = split_state["resid_cache_path"]
    train_data = split_state["train_data"]
    val_data = split_state["val_data"]
    train_taxed_data = scheme_state["train_taxed_data"]
    val_taxed_data = scheme_state["val_taxed_data"]

    probe_payload = probe.save_artifact(run_paths["probe_artifact_path"])
    save_json(artifacts_dir / "train_scheme_summary.json", train_eval)
    save_json(artifacts_dir / "val_scheme_summary.json", val_eval)
    save_json(artifacts_dir / "scheme_builder_meta.json", comparison)
    save_json(
        Path(run_paths["scheme_path"]),
        scheme,
    )
    save_json(
        artifacts_dir / "run_summary.json",
        {
            "run_name": cfg["run_name"],
            "run_id": cfg["run_id"],
            "method": cfg.get("method", METHOD_RESID_PROJECTION),
            "scheme_builder": scheme_builder,
            "model_name": cfg["model_name"],
            "model_id": cfg["model_id"],
            "intervention": cfg["intervention"],
            "seed": cfg["seed"],
            "run_dir": str(run_dir),
            "artifacts_dir": str(artifacts_dir),
            "settings_path": str(run_paths["settings_path"]),
            "probe_artifact_path": str(run_paths["probe_artifact_path"]),
            "scheme_path": str(run_paths["scheme_path"]),
            "profile_cache_path": str(cache_path),
            "resid_cache_path": str(resid_cache_path),
            "n_inputs": len(data_state["inputs"]),
            "n_train": len(train_data),
            "n_val": len(val_data),
            "n_train_taxed": len(train_taxed_data),
            "n_val_taxed": len(val_taxed_data),
            "probe_detection_layer": probe.detection_layer,
            "probe_threshold": probe.threshold,
            "probe_threshold_metric": probe.threshold_metric,
            "probe_only_eval": bool(cfg.get("probe_only_eval", False)),
            "scheme_num_components": len(scheme["selected_components"]),
            "scheme_train_eval": train_eval,
            "scheme_val_eval": val_eval,
            "scheme_val_joint_eval": val_joint_eval,
            "scheme_comparison": comparison,
            "resid_projection_layer": cfg.get("resid_projection_layer"),
            "resid_projection_layer_offset": cfg.get("resid_projection_layer_offset"),
            "resid_projection_auto_layer": bool(cfg.get("resid_projection_auto_layer", False)),
            "resid_projection_layer_select_split": cfg.get("resid_projection_layer_select_split"),
            "resid_projection_layer_plateau_frac": cfg.get("resid_projection_layer_plateau_frac"),
            "resid_projection_rank": cfg.get("resid_projection_rank"),
            "resid_projection_scale": cfg.get("resid_projection_scale", 1.0),
            "resid_projection_train_subset": cfg.get("resid_projection_train_subset"),
            "resid_projection_cka_weight": cfg.get("resid_projection_cka_weight", 0.0),
        },
    )
    save_json(
        Path(run_paths["settings_path"]),
        {
            "run_name": cfg["run_name"],
            "run_id": cfg["run_id"],
            "method": cfg.get("method", METHOD_RESID_PROJECTION),
            "model_name": cfg["model_name"],
            "model_id": cfg["model_id"],
            "intervention": cfg["intervention"],
            "run_dir": str(run_dir),
            "artifacts_dir": str(artifacts_dir),
            "logs_dir": str(run_paths["logs_dir"]),
            "probe_path": str(run_paths["probe_artifact_path"]),
            "scheme_path": str(run_paths["scheme_path"]),
            "detection_layer": int(probe.detection_layer),
            "probe_feature_mode": probe.feature_mode,
            "probe_classifier_type": probe_payload["classifier_type"],
            "probe_training_backend": probe_payload["training_backend"],
            "probe_input_dim": probe_payload["input_dim"],
            "probe_threshold": probe.threshold,
            "probe_threshold_metric": probe.threshold_metric,
            "probe_calibration": probe.calibration,
            "probe_only_eval": bool(cfg.get("probe_only_eval", False)),
            "scheme_builder": scheme_builder,
            "resid_projection_layer": cfg.get("resid_projection_layer"),
            "resid_projection_layer_offset": cfg.get("resid_projection_layer_offset"),
            "resid_projection_auto_layer": bool(cfg.get("resid_projection_auto_layer", False)),
            "resid_projection_layer_select_split": cfg.get("resid_projection_layer_select_split"),
            "resid_projection_layer_plateau_frac": cfg.get("resid_projection_layer_plateau_frac"),
            "resid_projection_rank": cfg.get("resid_projection_rank"),
            "resid_projection_scale": cfg.get("resid_projection_scale", 1.0),
            "resid_projection_train_subset": cfg.get("resid_projection_train_subset"),
            "resid_projection_cka_weight": cfg.get("resid_projection_cka_weight", 0.0),
        },
    )

    logger.info("saved results to: %s", run_dir)
    return {
        "probe_payload": probe_payload,
    }


def run_training_pipeline(cfg: dict[str, Any], model: Any | None = None) -> dict[str, Any]:
    assert model is not None, "Model must be provided to run_training_pipeline. Use main.py as the CLI entrypoint."
    context = prepare_run_context(cfg, model)
    method = str(cfg.get("method", METHOD_RESID_PROJECTION)).lower()
    if method != METHOD_RESID_PROJECTION:
        raise ValueError(f"Unsupported method for PRISM release: {method}")
    data_state = load_and_profile_data(context)
    split_state = split_and_collect_residuals(context, data_state)
    probe_state = fit_and_calibrate_probe(context, data_state, split_state)
    scheme_state = build_resid_projection_state(context, data_state, split_state, probe_state)
    save_state = save_training_artifacts(context, data_state, split_state, probe_state, scheme_state)

    split_state = {**split_state}
    split_state.pop("train_sample_ids", None)
    split_state.pop("val_sample_ids", None)

    return {
        "cfg": context["cfg"],
        "run_paths": context["run_paths"],
        "context": context,
        "data_state": data_state,
        "split_state": split_state,
        "probe_state": {**probe_state, **save_state},
        "scheme_state": scheme_state,
        "probe": probe_state["probe"],
        "probe_calibration": probe_state["probe_calibration"],
        "probe_payload": save_state["probe_payload"],
        "scheme": scheme_state["scheme"],
        "train_intervention_results": scheme_state["train_intervention_results"],
        "val_intervention_results": scheme_state["val_intervention_results"],
        "train_eval": scheme_state["train_eval"],
        "val_eval": scheme_state["val_eval"],
        "val_joint_eval": scheme_state["val_joint_eval"],
        "comparison": scheme_state["comparison"],
        "cache_path": data_state["cache_path"],
        "resid_cache_path": split_state["resid_cache_path"],
        "n_inputs": len(data_state["inputs"]),
        "n_train": len(split_state["train_data"]),
        "n_val": len(split_state["val_data"]),
        "n_train_taxed": len(scheme_state["train_taxed_data"]),
        "n_val_taxed": len(scheme_state["val_taxed_data"]),
    }


def train(cfg: dict[str, Any] | None = None, model: Any | None = None) -> dict[str, Any]:
    if cfg is None:
        raise ValueError("train(cfg) expects an explicit config. Use main.py as the CLI entrypoint.")
    return run_training_pipeline(cfg, model=model)
