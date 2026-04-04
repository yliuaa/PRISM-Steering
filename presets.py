from __future__ import annotations

DEFAULT_INTERVENTION = "moral_self_correction"
DEFAULT_METHOD = "resid_projection"
DEFAULT_BENCHMARKS = ("bbq", "diffaware")


def _model_preset(
    *,
    model_id: str,
    probe_num_subspaces: int,
    probe_subspace_rank: int,
) -> dict:
    return {
        "model_id": model_id,
        "method": DEFAULT_METHOD,
        "intervention": DEFAULT_INTERVENTION,
        "seed": 2026,
        "train_ratio": 0.7,
        "batch_size": 16,
        "profile_batch_size": 32,
        "probe_type": "sparse_subspace",
        "probe_metric": "val_auprc",
        "probe_threshold_metric": "f1",
        "probe_pos_weight": 0.25,
        "probe_sparse_epochs": 80,
        "probe_sparse_lr": 1e-3,
        "probe_sparse_batch_size": 256,
        "probe_sparse_weight_decay": 1e-4,
        "probe_orthogonality_weight": 1e-2,
        "probe_aggregator_l1_weight": 1e-3,
        "probe_num_subspaces": probe_num_subspaces,
        "probe_subspace_rank": probe_subspace_rank,
        "resid_projection_rank": 4,
        "resid_projection_scale": 1.5,
        # "resid_projection_layer": 22,
        "resid_projection_layer_offset": 0,
        "resid_projection_auto_layer": True,
        "resid_projection_layer_select_split": "validation",
        "resid_projection_layer_plateau_frac": 0.98,
        "resid_projection_apply_mode": "last_token",
        "resid_projection_tail_window_radius": 1,
        "resid_projection_train_subset": "taxed",
        "resid_projection_train_steps": 400,
        "resid_projection_lr": 1e-2,
        "resid_projection_weight_decay": 1e-4,
        "resid_projection_cka_weight": 0.0,
        "answer_format": "json_upper_v1",
        "model_load_mode": "no_processing",
        "specify_america": True,
        "prompt_id": 1,
    }


MODEL_PRESETS = {
    "qwen-2.5-7b": _model_preset(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        probe_num_subspaces=2,
        probe_subspace_rank=32,
    ),
    "qwen-2.5-14b": _model_preset(
        model_id="Qwen/Qwen2.5-14B-Instruct",
        probe_num_subspaces=4,
        probe_subspace_rank=32,
    ),
    "gemma-2-9b-it": _model_preset(
        model_id="google/gemma-2-9b-it",
        probe_num_subspaces=4,
        probe_subspace_rank=32,
    ),
}

SUPPORTED_MODELS = tuple(MODEL_PRESETS)
