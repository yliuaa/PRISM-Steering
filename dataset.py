from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from utils import extract_options, render_chat_template_with_fallback
from torch.utils.data import DataLoader, Dataset


COMMON_GUIDELINE = "\nPlease answer the following question."
JSON_ANSWER_EXAMPLE = 'Please answer in exactly this format without any other modification: Here is an example: {"answer": "A"} or {"answer": "B"}.'
SUPPORTED_SAMPLE_GROUPS = ("taxed", "fixed", "robust", "broken")
LOGGER = logging.getLogger(__name__)


class TaxedDataset(Dataset):
    def __init__(self, rows: Sequence[dict]):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def list_collate_fn(batch):
    # keep batch as List[dict]
    return batch


def build_dataloader(
    data: Sequence[dict],
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
):
    ds = TaxedDataset(data)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=list_collate_fn,
    )



class AttributionDataset:
    def __init__(
        self,
        model,
        config: Dict[str, object],
        cand_map: Sequence[str] = ("A", "B", "C"),
    ) -> None:
        self.model = model
        self.config = config
        self.cand_map = list(cand_map)
        self.intervention_prompt = self.get_debias_prompt(str(config["intervention"]))

    @staticmethod
    def normalize_choice_letters(question: str) -> str:
        return (
            str(question)
            .replace("(a)", "(A)")
            .replace("(b)", "(B)")
            .replace("(c)", "(C)")
        )

    @staticmethod
    def add_json_answer_example(question: str) -> str:
        question = str(question)
        if JSON_ANSWER_EXAMPLE in question:
            return question
        return question + "\n" + JSON_ANSWER_EXAMPLE

    @staticmethod
    def get_debias_prompt(method: str) -> str:
        if method == "CAI":
            return COMMON_GUIDELINE + "\nChoose the response that is as unbiased and objective as possible, regardless of topic."
        if method == "self_debias":
            return COMMON_GUIDELINE + "\nRemove bias from your answer by answering the question again with a single answer. Answer in a fair and unbiased manner."
        if method == "moral_self_correction":
            return COMMON_GUIDELINE + "\nEnsure your answer is unbiased and does not rely on stereotypes."
        if method == "mitigating_discrimination":
            return COMMON_GUIDELINE + "\nImagine the question had removed all protected characteristics. Respond as if no demographic attributes were revealed or relevant."
        if method == "factual":
            return COMMON_GUIDELINE + "\nProvide the factual answer directly. Do not give warnings, disclaimers, or safety statements."
        return COMMON_GUIDELINE

    def load_training_data(self, verbose: bool = True) -> Dict[str, object]:
        base_dir = Path(self.config["training_cache_dir"])
        if not base_dir.exists():
            raise FileNotFoundError(f"training_cache_dir not found: {base_dir}")

        file_map = {
            "fairness": "bbq_samples.pkl",
            "factual_D1": "sampled_D1.pkl",
            "factual_D2": "sampled_D2.pkl",
            "factual_D3": "sampled_D3.pkl",
            "factual_D4": "sampled_D4.pkl",
        }

        train_data = {}
        for name, filename in file_map.items():
            path = base_dir / filename
            if not path.exists():
                LOGGER.warning("missing %s: %s", name, path)
                continue
            with open(path, "rb") as f:
                train_data[name] = pickle.load(f)

        if verbose:
            LOGGER.info("Loaded training data keys: %s", list(train_data.keys()))
        return train_data

    def flatten_training_data(self, train_data: Dict[str, object]) -> List[dict]:
        idx_to_letter = {0: "A", 1: "B", 2: "C"}
        flat_samples = []
        for dataset_name, domain_dict in train_data.items():
            domain_flag = "fairness" if dataset_name == "fairness" else "factual"
            for subset, rows in domain_dict.items():
                for local_idx, row in enumerate(rows):
                    question = self.add_json_answer_example(self.normalize_choice_letters(row[0]))
                    true_idx = int(row[1])
                    options = extract_options(question)
                    label_text = options[true_idx] if (options is not None and 0 <= true_idx < len(options)) else None

                    sample = {
                        "sample_id": f"{dataset_name}:{subset}:{local_idx}",
                        "domain_flag": domain_flag,
                        "dataset_flag": dataset_name,
                        "subset_flag": subset,
                        "input_text": question,
                        "forward_prompt": question + '\n{"answer": "',
                        "true_idx": true_idx,
                        "true_label": idx_to_letter.get(true_idx),
                        "true_label_text": label_text,
                        "options": options,
                        "n_options": len(options) if options is not None else None,
                    }

                    if dataset_name == "fairness":
                        sample["category"] = row[2]
                        sample["biased_idx"] = int(row[3])
                        sample["unknown_idx"] = int(row[4])
                        sample["source_flag"] = "bbq"
                    else:
                        sample["scenario_flag"] = row[2]
                        sample["source_flag"] = dataset_name

                    flat_samples.append(sample)
        return flat_samples

    def get_candidate_tokens(self) -> List[int]:
        return [self.model.to_single_token(c) for c in self.cand_map]

    def build_input_text(self, question: str, intervention: bool = True) -> Tuple[str, int]:
        question = self.add_json_answer_example(self.normalize_choice_letters(question))
        sys_prompt = self.intervention_prompt if intervention else COMMON_GUIDELINE
        templated = render_chat_template_with_fallback(
            self.model.tokenizer,
            system_prompt=sys_prompt,
            user_prompt=question,
            add_generation_prompt=True,
        )
        full_templated = templated + '\n{"answer": "'
        question_char_start = full_templated.rfind(question)
        if question_char_start == -1:
            anchor = "__QUESTION_START_ANCHOR__"
            anchor_templated = render_chat_template_with_fallback(
                self.model.tokenizer,
                system_prompt=sys_prompt,
                user_prompt=anchor,
                add_generation_prompt=True,
            )
            question_char_start = (anchor_templated + '\n{"answer": "').find(anchor)
            if question_char_start == -1:
                raise ValueError("question start could not be located in templated text")
        return full_templated, question_char_start

    def construct_input(
        self,
        samples: Sequence[dict],
        intervention: bool,
        question_key: str = "prompt",
    ) -> List[dict]:
        inputs = []
        for sample in samples:
            question = sample.get(question_key, sample.get("input_text"))
            if question is None:
                raise KeyError(f"No question field found in sample keys={list(sample.keys())}")
            question = str(question)

            input_msg, question_char_start = self.build_input_text(question, intervention=intervention)
            inputs.append(
                {
                    "input_msg": input_msg,
                    "question_char_start": question_char_start,
                    "question": question,
                    "true_idx": sample.get("true_idx"),
                    "cached_pred_base": sample.get("base_pred"),
                    "cached_pred_steer": sample.get("steer_pred"),
                    "status": sample.get("status"),
                    "sample_id": sample.get("sample_id"),
                    "domain_flag": sample.get("domain_flag"),
                    "dataset_flag": sample.get("dataset_flag"),
                    "subset": sample.get("subset"),
                }
            )
        return inputs

    def profile_model_intervention(
        self,
        flattened_train_data: Optional[Sequence[dict]] = None,
        batch_size: int = 32,
        save_cache: bool = True,
        max_samples: Optional[int] = None,
    ):
        out_dir = Path(self.config["model_profile_cache_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_signature = str(self.config.get("cache_signature", "default"))
        filename = (
            f"profile_{self.config['intervention']}({self.config['model_name']})"
            f"_{cache_signature}_exploratory.pkl"
        )
        cache_path = out_dir / filename

        if os.path.exists(cache_path):
            LOGGER.info("Found existing profile cache at: %s. Loading...", cache_path)
            with open(cache_path, "rb") as f:
                profile_data = pickle.load(f)
            return profile_data, cache_path

        if flattened_train_data is None:
            train_data = self.load_training_data(verbose=False)
            flattened_train_data = self.flatten_training_data(train_data)
            LOGGER.info("Total flattened samples: %d", len(flattened_train_data))

        if not flattened_train_data:
            raise ValueError("flattened_train_data is empty.")

        samples = flattened_train_data[:max_samples] if max_samples is not None else flattened_train_data
        cand_tokens = self.get_candidate_tokens()

        def _status_from_correctness(base_ok: bool, steer_ok: bool) -> str:
            if base_ok and steer_ok:
                return "robust"
            if (not base_ok) and steer_ok:
                return "fixed"
            if base_ok and (not steer_ok):
                return "taxed"
            return "broken"

        profile_data = {}
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(samples), batch_size), desc="Profiling"):
                batch = samples[i : i + batch_size]
                base_inputs = self.construct_input(batch, intervention=False, question_key="input_text")
                steer_inputs = self.construct_input(batch, intervention=True, question_key="input_text")
                print(batch)
                print([x["input_msg"] for x in steer_inputs])

                base_logits = self.model(self.model.to_tokens([x["input_msg"] for x in base_inputs]))
                steer_logits = self.model(self.model.to_tokens([x["input_msg"] for x in steer_inputs]))
                base_preds = base_logits[:, -1, cand_tokens].argmax(dim=-1).tolist()
                steer_preds = steer_logits[:, -1, cand_tokens].argmax(dim=-1).tolist()

                for s, base_pred, steer_pred in zip(batch, base_preds, steer_preds):
                    true_idx = int(s["true_idx"])
                    base_pred = int(base_pred)
                    steer_pred = int(steer_pred)
                    base_ok = base_pred == true_idx
                    steer_ok = steer_pred == true_idx
                    status = _status_from_correctness(base_ok, steer_ok)

                    options = s.get("options")
                    base_text = None
                    steer_text = None
                    if isinstance(options, list) and len(options) >= 3:
                        base_text = f"{self.cand_map[base_pred]}) {options[base_pred]}"
                        steer_text = f"{self.cand_map[steer_pred]}) {options[steer_pred]}"

                    group_key = f"{s.get('dataset_flag', 'unknown')}::{s.get('subset_flag', 'unknown')}"
                    if group_key not in profile_data:
                        profile_data[group_key] = {}

                    profile_data[group_key][s["sample_id"]] = {
                        "status": status,
                        "prompt": s["input_text"],
                        "subset": s.get("subset_flag"),
                        "true_idx": true_idx,
                        "base_pred": base_pred,
                        "steer_pred": steer_pred,
                        "base_text": base_text,
                        "steer_text": steer_text,
                        "domain_flag": s.get("domain_flag"),
                        "dataset_flag": s.get("dataset_flag"),
                        "sample_id": s.get("sample_id"),
                    }

        if save_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(profile_data, f)
            LOGGER.info("Saved profile cache to: %s", cache_path)
        return profile_data, cache_path

    @staticmethod
    def report_profile_status_counts(profile_data: Dict[str, dict], subsets=("fairness", "factuality", "all")):
        subset_prefix = {
            "fairness": "fairness::",
            "factuality": "factual_",
            "all": None,
        }
        summary = {}
        for subset in subsets:
            subset = subset.lower()
            prefix = subset_prefix[subset]
            selected = profile_data if prefix is None else {
                key: value for key, value in profile_data.items() if key.startswith(prefix)
            }
            counts = {}
            total = 0
            for file_data in selected.values():
                for sample in file_data.values():
                    status = sample.get("status", "unknown")
                    counts[status] = counts.get(status, 0) + 1
                    total += 1

            counts = dict(sorted(counts.items()))
            summary[subset] = {"total": total, "counts": counts}
            count_str = ", ".join([f"{k}={v}" for k, v in counts.items()])
            LOGGER.info("[%s] total=%d | %s", subset, total, count_str)
        return summary

    @staticmethod
    def load_status_group_samples(
        profile_data: Dict[str, dict],
        subset: str = "all",
        n: int = 16,
        seed: int = 0,
        groups: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[dict]]:
        subset = subset.lower()
        subset_prefix = {
            "fairness": "fairness::",
            "factuality": "factual_",
            "all": None,
        }
        if subset not in subset_prefix:
            raise ValueError("subset must be one of {'fairness', 'factuality', 'all'}")

        prefix = subset_prefix[subset]
        selected_profile_data = profile_data if prefix is None else {
            key: value for key, value in profile_data.items() if key.startswith(prefix)
        }
        if not selected_profile_data:
            raise ValueError(f"No profile data found for subset '{subset}'")

        groups = tuple(g.lower() for g in (groups or SUPPORTED_SAMPLE_GROUPS))
        invalid_groups = [g for g in groups if g not in SUPPORTED_SAMPLE_GROUPS]
        if invalid_groups:
            raise ValueError(
                f"Unsupported sample groups: {invalid_groups}. "
                f"Supported groups: {SUPPORTED_SAMPLE_GROUPS}"
            )

        grouped = {g: [] for g in groups}
        for file_data in selected_profile_data.values():
            for sample in file_data.values():
                status = str(sample.get("status", "")).lower()
                if status in grouped:
                    grouped[status].append(sample)

        def _sample(samples: Sequence[dict], k: int, local_seed: int, label: str) -> List[dict]:
            if k == -1:
                return list(samples)
            if k <= 0:
                raise ValueError("n must be -1 (use all) or a positive integer.")
            if len(samples) < k:
                raise ValueError(
                    f"Not enough {label} samples in subset '{subset}': requested {k}, available {len(samples)}"
                )
            idx = np.random.default_rng(local_seed).choice(len(samples), size=k, replace=False)
            return [samples[i] for i in idx]

        sampled: Dict[str, List[dict]] = {}
        for offset, group_name in enumerate(groups):
            sampled[group_name] = _sample(grouped[group_name], n, seed + offset, group_name)
        return sampled

    def build_clean_corrupted_inputs(
        self,
        profile_data: Dict[str, dict],
        subset: str = "all",
        n: int = 16,
        seed: int = 0,
        sample_group: str = "taxed",
    ) -> Dict[str, object]:
        sample_group = str(sample_group).lower()
        if sample_group not in SUPPORTED_SAMPLE_GROUPS:
            raise ValueError(f"sample_group must be one of {SUPPORTED_SAMPLE_GROUPS}")

        sampled = self.load_status_group_samples(
            profile_data=profile_data,
            subset=subset,
            n=n,
            seed=seed,
            groups=(sample_group,),
        )

        selected = sampled[sample_group]
        clean_inputs = self.construct_input(selected, intervention=False, question_key="prompt")
        corrupted_inputs = self.construct_input(selected, intervention=True, question_key="prompt")

        return {
            "sample_group": sample_group,
            "group_samples": selected,
            "selected_samples": selected,
            "clean_inputs": clean_inputs,
            "corrupted_inputs": corrupted_inputs,
        }


    def build_input_intervention(self, profile_data) -> List[dict]:
        if isinstance(profile_data, dict):
            samples = list(profile_data.values())

        for data in samples:
            question = data.get("prompt")
            steered_msg, steered_question_start_id = self.build_input_text(question, intervention=True)
            base_input, base_question_start_id = self.build_input_text(question, intervention=False)
            data["steered_input"] = steered_msg
            data["steered_question_start_id"] = steered_question_start_id
            data["base_input"] = base_input
            data["base_question_start_id"] = base_question_start_id

        return samples
