from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dataset import AttributionDataset, COMMON_GUIDELINE, JSON_ANSWER_EXAMPLE
from probe import TaxProbe, load_probe_from_artifact
from train import (
    METHOD_RESID_PROJECTION,
    TRAINING_METHODS,
    build_run_paths,
    build_training_cfg,
    load_config,
    train,
)
from utils import char_to_token_idx, render_chat_template_with_fallback

ANSWER_PREFIX = '{"answer": "'


@dataclass
class RuntimeController:
    name: str
    model: Any
    probe: TaxProbe | None = None
    scheme: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    hooks_registered: bool = False
    hook_specs: list[tuple[str, Any]] = field(default_factory=list)
    trigger_mask: torch.Tensor | None = None
    last_trigger_mask: torch.Tensor | None = None
    current_pad_mask: torch.Tensor | None = None
    current_tail_positions: list[list[int]] | None = None

    def register_hooks(self) -> None:
        if self.name not in TRAINING_METHODS:
            self.hook_specs = []
            self.hooks_registered = True
            return
        detection_layer = int(self.probe.detection_layer)
        detection_hook_name = f"blocks.{detection_layer}.hook_resid_pre"
        if bool(self.settings.get("probe_only_eval", False)):
            def probe_only_hook(act, hook):
                if hook.name == detection_hook_name:
                    self.trigger_mask = self._compute_trigger_mask(act)
                return act

            self.hook_specs = [(detection_hook_name, probe_only_hook)]
            self.hooks_registered = True
            return
        if self.name == METHOD_RESID_PROJECTION:
            projection = dict(self.scheme.get("projection", {}))
            projection_layer = int(projection["layer"])
            projection_hook_name = f"blocks.{projection_layer}.hook_resid_pre"

            def projection_hook(act, hook):
                if hook.name == detection_hook_name:
                    self.trigger_mask = self._compute_trigger_mask(act)
                if hook.name == projection_hook_name:
                    return self._apply_resid_projection(act, self.trigger_mask)
                return act

            hook_names = [detection_hook_name]
            if projection_hook_name != detection_hook_name:
                hook_names.append(projection_hook_name)
            self.hook_specs = [(name, projection_hook) for name in hook_names]
            self.hooks_registered = True
            return
        raise ValueError(f"Unsupported runtime mode for PRISM release: {self.name}")

    def clear_hooks(self) -> None:
        self.hook_specs = []
        self.trigger_mask = None
        self.current_pad_mask = None
        self.current_tail_positions = None
        self.hooks_registered = False

    def _infer_pad_mask(self, *model_args) -> torch.Tensor | None:
        if not model_args:
            return None
        first = model_args[0]
        tokens = None
        if isinstance(first, torch.Tensor):
            tokens = first
        elif isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
            tokens = self.model.to_tokens(list(first))
        if tokens is None:
            return None
        pad_token_id = getattr(self.model.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            return torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        return tokens != int(pad_token_id)

    def _infer_tail_positions(
        self,
        *model_args,
        pad_mask: torch.Tensor | None = None,
    ) -> list[list[int]] | None:
        if not model_args:
            return None
        first = model_args[0]
        texts: list[str] | None = None
        if isinstance(first, torch.Tensor):
            token_rows = first
            texts = []
            row_masks = None
            if pad_mask is not None:
                row_masks = pad_mask.detach().cpu()
            for row_idx in range(token_rows.shape[0]):
                row_tokens = token_rows[row_idx]
                if row_masks is not None:
                    keep = row_masks[row_idx]
                    row_tokens = row_tokens[keep]
                texts.append(self.model.to_string(row_tokens))
        elif isinstance(first, str):
            texts = [first]
        elif isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
            texts = [str(x) for x in first]
        if texts is None:
            return None

        radius = max(0, int(self.settings.get("resid_projection_tail_window_radius", 1)))
        pad_mask_cpu = None if pad_mask is None else pad_mask.detach().cpu()
        positions: list[list[int]] = []
        for row_idx, text in enumerate(texts):
            n_tokens = int(self.model.to_tokens(text).shape[-1])
            if n_tokens <= 0:
                positions.append([])
                continue
            format_start_char = text.find(JSON_ANSWER_EXAMPLE)
            if format_start_char == -1:
                tail_start = max(0, n_tokens - 1)
            else:
                format_start_tok = int(char_to_token_idx(self.model, text, format_start_char))
                tail_start = max(0, format_start_tok - radius)
            token_positions = list(range(tail_start, n_tokens))
            if pad_mask_cpu is not None and row_idx < pad_mask_cpu.shape[0]:
                row_mask = pad_mask_cpu[row_idx]
                valid_positions = torch.nonzero(row_mask, as_tuple=False).view(-1)
                if valid_positions.numel() > 0:
                    offset = int(valid_positions[0].item())
                    token_positions = [offset + pos for pos in token_positions]
            positions.append(token_positions)
        return positions

    def _compute_trigger_mask(self, act) -> torch.Tensor:
        if self.probe.feature_mode == "norm":
            features = act[:, -1, :].float().norm(p=2, dim=-1, keepdim=True)
        else:
            features = act[:, -1, :].float()
        trigger = self.probe.should_trigger(features)
        return torch.as_tensor(trigger, device=act.device, dtype=torch.bool)

    def _apply_resid_projection(self, act, mask: torch.Tensor):
        if mask is None or not bool(mask.any()):
            return act
        projection = dict(self.scheme.get("projection", {}))
        if not projection:
            return act
        out = act.clone()
        u = torch.as_tensor(projection["u"], device=act.device, dtype=torch.float32)
        v = torch.as_tensor(projection["v"], device=act.device, dtype=torch.float32)
        b = torch.as_tensor(projection["b"], device=act.device, dtype=torch.float32)
        scale = float(projection.get("scale", self.settings.get("resid_projection_scale", 1.0)))
        apply_mode = str(
            projection.get(
                "apply_mode",
                self.settings.get("resid_projection_apply_mode", "last_token"),
            )
        ).lower()
        if apply_mode == "tail_bundle":
            tail_positions = self.current_tail_positions or []
            masked_indices = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
            for batch_idx in masked_indices:
                if batch_idx >= len(tail_positions):
                    continue
                token_positions = tail_positions[batch_idx]
                if not token_positions:
                    continue
                x = out[batch_idx, token_positions, :].to(torch.float32)
                correction = (x @ v) @ u.T + b
                out[batch_idx, token_positions, :] = (x - scale * correction).to(out.dtype)
            return out
        x = out[mask, -1, :].to(torch.float32)
        correction = (x @ v) @ u.T + b
        out[mask, -1, :] = (x - scale * correction).to(out.dtype)
        return out

    # Forward
    def run_with_hooks(self, *model_args, **model_kwargs):
        if not self.hooks_registered:
            self.register_hooks()
        self.trigger_mask = None
        self.current_pad_mask = self._infer_pad_mask(*model_args)
        self.current_tail_positions = self._infer_tail_positions(*model_args, pad_mask=self.current_pad_mask)
        with torch.inference_mode():
            out = self.model.run_with_hooks(*model_args, fwd_hooks=self.hook_specs, **model_kwargs)
        self.last_trigger_mask = None if self.trigger_mask is None else self.trigger_mask.detach().cpu().clone()
        return out

    # Generate
    def generate_with_hooks(self, *model_args, **model_kwargs):
        if not self.hooks_registered:
            self.register_hooks()
        self.trigger_mask = None
        self.current_pad_mask = self._infer_pad_mask(*model_args)
        self.current_tail_positions = self._infer_tail_positions(*model_args, pad_mask=self.current_pad_mask)
        model_kwargs.setdefault("use_past_kv_cache", False)
        with torch.inference_mode():
            with self.model.hooks(fwd_hooks=self.hook_specs):
                out = self.model.generate(*model_args, **model_kwargs)
        self.last_trigger_mask = None if self.trigger_mask is None else self.trigger_mask.detach().cpu().clone()
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "hooks_registered": self.hooks_registered,
            "n_hook_specs": len(self.hook_specs),
            "detection_layer": self.settings.get("detection_layer"),
            "scheme_components": len(self.scheme.get("selected_components", [])),
        }

    def get_last_batch_trace(self, batch_size: int | None = None) -> dict[str, Any]:
        if self.last_trigger_mask is None:
            return {"is_triggered": [None] * batch_size if batch_size is not None else None}
        return {"is_triggered": self.last_trigger_mask.tolist()}


def build_runtime_controller(
    *,
    mode: str,
    model: Any,
    probe: TaxProbe,
    scheme: dict[str, Any],
    settings: dict[str, Any],
) -> RuntimeController:
    resolved_mode = str(mode or "prism").lower()
    return RuntimeController(
        name=resolved_mode,
        model=model,
        probe=probe,
        scheme=scheme,
        settings=settings,
    )


class PRISM:
    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        cfg: dict[str, Any] | None = None,
        run_id: str,
        model: Any | None = None,
        intervention: str | None = None,
        model_name: str | None = None,
        model_id: str | None = None,
        method: str | None = None,
        auto_build: bool = False,
    ) -> None:
        if cfg is None and config_path is None:
            raise ValueError("Provide either cfg or config_path when constructing PRISM.")

        base_cfg = dict(cfg) if cfg is not None else load_config(Path(config_path))
        self.cfg = build_training_cfg(
            base_cfg,
            run_id=run_id,
            intervention=intervention,
            model_name=model_name,
            model_id=model_id,
            method=method,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.run_paths = build_run_paths(self.cfg)
        self.model = model
        self.tokenizer = None if model is None else model.tokenizer
        self.artifact_status: dict[str, bool] = {}
        self.settings: dict[str, Any] | None = None
        self.probe: TaxProbe | None = None
        self.scheme: dict[str, Any] | None = None
        self.runtime_controller: RuntimeController | None = None
        self.intervention_prompt = AttributionDataset.get_debias_prompt(str(self.cfg["intervention"]))
        self.init_runtime(auto_build=auto_build)

    def init_runtime(self, *, auto_build: bool = False) -> None:
        method = str(self.cfg.get("method", METHOD_RESID_PROJECTION)).lower()
        paths = self.run_paths
        artifact_run_id = self.cfg.get("artifact_run_id")
        if artifact_run_id:
            artifact_cfg = dict(self.cfg)
            artifact_cfg["run_id"] = str(artifact_run_id)
            paths = build_run_paths(artifact_cfg)
        run_dir = Path(paths["run_dir"])
        settings_path = Path(paths["settings_path"])
        probe_path = Path(paths["probe_artifact_path"])
        scheme_path = Path(paths["scheme_path"])
        if method not in TRAINING_METHODS:
            self.artifact_status = {"settings": False, "probe": False, "scheme": False}
            if self.model is not None:
                self.runtime_controller = build_runtime_controller(
                    mode=method,
                    model=self.model,
                    probe=None,
                    scheme={},
                    settings={},
                )
                self.runtime_controller.register_hooks()
            return
        self.artifact_status = {
            "settings": settings_path.exists(),
            "probe": probe_path.exists(),
            "scheme": scheme_path.exists(),
        }

        if not all(self.artifact_status.values()):
            if not auto_build:
                raise FileNotFoundError(
                    f"Missing PRISM artifacts for run '{self.run_paths['run_name']}' at {run_dir}: "
                    f"{self.artifact_status}"
                )
            if self.model is None:
                raise ValueError("auto_build=True requires a shared model object.")
            self.logger.info("Missing PRISM artifacts. Building them via training pipeline.")
            train(self.cfg, model=self.model)
            self.artifact_status = {
                "settings": settings_path.exists(),
                "probe": probe_path.exists(),
                "scheme": scheme_path.exists(),
            }
        else:
            self.logger.info("Found existing PRISM artifacts in %s", run_dir)

        with settings_path.open("r") as f:
            self.settings = json.load(f)
        with scheme_path.open("r") as f:
            self.scheme = json.load(f)
        runtime_overrides = {
            "probe_only_eval": bool(self.cfg.get("probe_only_eval")) if self.cfg.get("probe_only_eval") is not None else None,
            "resid_projection_scale": self.cfg.get("resid_projection_scale"),
            "resid_projection_apply_mode": self.cfg.get("resid_projection_apply_mode"),
            "resid_projection_tail_window_radius": self.cfg.get("resid_projection_tail_window_radius"),
        }
        for key, value in runtime_overrides.items():
            if value is not None:
                self.settings[key] = value
        if "projection" in self.scheme:
            if self.cfg.get("resid_projection_scale") is not None:
                self.scheme["projection"]["scale"] = float(self.cfg["resid_projection_scale"])
            if self.cfg.get("resid_projection_apply_mode") is not None:
                self.scheme["projection"]["apply_mode"] = str(self.cfg["resid_projection_apply_mode"])
        device = self.cfg.get("model_kwargs", {}).get("device", "cuda")
        self.probe = load_probe_from_artifact(
            probe_path,
            device=device,
        )
        if self.model is not None:
            self.runtime_controller = build_runtime_controller(
                mode=str(self.cfg.get("method", METHOD_RESID_PROJECTION)),
                model=self.model,
                probe=self.probe,
                scheme=self.scheme,
                settings=self.settings,
            )
            self.runtime_controller.register_hooks()

    def init_prism(self, *, auto_build: bool = False) -> None:
        self.init_runtime(auto_build=auto_build)

    def unload_runtime(self) -> None:
        if self.runtime_controller is not None:
            self.runtime_controller.clear_hooks()

    def _require_runtime(self) -> RuntimeController:
        if self.runtime_controller is None or self.model is None or self.tokenizer is None:
            raise ValueError("PRISM runtime is not initialized with model weights.")
        return self.runtime_controller

    def _build_system_prompt(self, system_prompt_prefix: str | None = None) -> str:
        prefix = (system_prompt_prefix or "").strip()
        intervention = str(self.intervention_prompt or "").strip()
        if prefix and intervention == COMMON_GUIDELINE.strip():
            return prefix
        if prefix and intervention:
            return f"{prefix}\n\n{intervention}"
        if prefix:
            return prefix
        return intervention

    def _format_prompt(
        self,
        prompt: str,
        answer_prefix: str = "",
        system_prompt_prefix: str | None = None,
    ) -> str:
        text = render_chat_template_with_fallback(
            self.tokenizer,
            system_prompt=self._build_system_prompt(system_prompt_prefix),
            user_prompt=str(prompt),
            add_generation_prompt=True,
        )
        return text + answer_prefix

    def _normalize_candidates(self, candidates, batch_size: int) -> list[list[str]]:
        if candidates is None:
            return [["A", "B", "C"] for _ in range(batch_size)]
        if isinstance(candidates[0], str):
            return [list(candidates) for _ in range(batch_size)]
        return [list(row) for row in candidates]

    def chat(
        self,
        prompts,
        eval_mode: str = "logits",
        candidates=None,
        max_new_tokens: int | None = None,
        system_prompt_prefix: str | None = None,
    ):
        controller = self._require_runtime()
        single = isinstance(prompts, str)
        prompts = [prompts] if single else list(prompts)

        if eval_mode == "logits":
            texts = [
                self._format_prompt(
                    prompt,
                    answer_prefix=ANSWER_PREFIX,
                    system_prompt_prefix=system_prompt_prefix,
                )
                for prompt in prompts
            ]
            logits = controller.run_with_hooks(texts, return_type="logits")
            cand_rows = self._normalize_candidates(candidates, len(prompts))
            prefix_ids = self.tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
            outputs = []
            for i, row in enumerate(cand_rows):
                token_scores = {}
                for cand in row:
                    token = str(cand).upper()
                    token_ids = self.tokenizer.encode(ANSWER_PREFIX + token, add_special_tokens=False)
                    if len(token_ids) == len(prefix_ids) + 1:
                        token_scores[token] = logits[i, -1, token_ids[-1]].item()
                if not token_scores:
                    raise ValueError(f"No single-token candidates found for row {row}")
                best = max(token_scores, key=token_scores.get)
                outputs.append(f'{{"answer": "{best}"}}')
            return outputs[0] if single else outputs

        texts = [
            self._format_prompt(
                prompt,
                answer_prefix=ANSWER_PREFIX,
                system_prompt_prefix=system_prompt_prefix,
            )
            for prompt in prompts
        ]
        prompt_toks = self.model.to_tokens(texts)
        generated = controller.generate_with_hooks(
            texts,
            max_new_tokens=max_new_tokens or int(self.cfg.get("max_new_tokens", 50)),
            do_sample=False,
            return_type="tokens",
            verbose=False,
        )
        start = prompt_toks.shape[1]
        decoded = [self.model.to_string(row[start:]) for row in generated]
        return decoded[0] if single else decoded

    def get_choice_scores(self, prompts, choices_list, batch_size: int = 16, system_prompt_prefix: str | None = None):
        controller = self._require_runtime()
        single = isinstance(prompts, str)
        prompts = [prompts] if single else list(prompts)
        pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            flat = []
            for i, (prompt, choices) in enumerate(zip(prompts, choices_list)):
                base = self._format_prompt(prompt, system_prompt_prefix=system_prompt_prefix)
                for choice in choices:
                    flat.append((i, base, base + str(choice)))

            scores = [[] for _ in prompts]
            pad_id = self.tokenizer.pad_token_id
            for start in range(0, len(flat), batch_size):
                batch = flat[start : start + batch_size]
                prompt_texts = [x[1] for x in batch]
                full_texts = [x[2] for x in batch]
                prompt_toks = self.model.to_tokens(prompt_texts)
                full_toks = self.model.to_tokens(full_texts)
                logits = controller.run_with_hooks(full_toks, return_type="logits")
                prompt_lens = (prompt_toks != pad_id).sum(dim=1)
                full_lens = (full_toks != pad_id).sum(dim=1)
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                labels = full_toks[:, 1:]
                token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                for j, (row_idx, _, _) in enumerate(batch):
                    start_idx = int(prompt_lens[j].item()) - 1
                    end_idx = int(full_lens[j].item()) - 1
                    score = float(token_log_probs[j, start_idx:end_idx].sum().item()) if start_idx < end_idx else 0.0
                    scores[row_idx].append(score)
            return scores[0] if single else scores
        finally:
            self.tokenizer.padding_side = pad_side

    def describe(self) -> dict[str, Any]:
        paths = self.run_paths
        return {
            "run_name": paths["run_name"],
            "run_dir": str(Path(paths["run_dir"])),
            "method": self.cfg.get("method", METHOD_RESID_PROJECTION),
            "artifacts_ready": all(self.artifact_status.values()),
            "artifact_status": self.artifact_status,
            "settings_loaded": self.settings is not None,
            "probe_loaded": self.probe is not None or Path(paths["probe_artifact_path"]).exists(),
            "scheme_loaded": self.scheme is not None,
            "model_loaded": self.model is not None,
            "runtime_controller": None if self.runtime_controller is None else self.runtime_controller.describe(),
        }

    def get_last_batch_trace(self, batch_size: int | None = None) -> dict[str, Any]:
        controller = self._require_runtime()
        return controller.get_last_batch_trace(batch_size=batch_size)
