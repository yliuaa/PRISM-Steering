import logging
import numpy as np
from pathlib import Path
import re
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


LOGGER = logging.getLogger(__name__)

class TaxProbe:

    def __init__(self, train_data, val_data, feature_mode='residual', pos_weight: float = 1.0):
        assert len(train_data) != 0 and len(val_data) != 0, "train_data and val_data must not be empty."
        self.data ={
            "train": train_data,
            "val": val_data
        }
        self.best_classifier = None
        self.detection_layer = None
        self.probe_results = []
        self.training_backend = None
        self.threshold = 0.5
        self.threshold_metric = None
        self.calibration = None
        self.steering_vec = None

        self.feature_mode = feature_mode
        self.pos_weight = float(pos_weight)

    def preprocess_data(self, layer=0, split="train", backend="torch"):
        X, y = [], []
        n_layers = int(self.data[split][0]["residual"].shape[0])
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"layer={layer} out of range for split='{split}', n_layers={n_layers}")
        for d in self.data[split]:
            resid = d["residual"].detach()
            if self.feature_mode == "residual":
                if backend == "torch":
                    x = resid[layer].float()
                else:
                    x = resid[layer].float().cpu().numpy().astype(np.float32, copy=False)
            elif self.feature_mode == "norm":
                if backend == "torch":
                    feat = resid.float().norm(p=2, dim=-1)
                    x = feat[layer].unsqueeze(0)
                else:
                    feat = resid.float().cpu().norm(p=2, dim=-1)
                    x = np.array([float(feat[layer].item())], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")
            X.append(x)
            y.append(int(d["label"]))
        if backend == "torch":
            return torch.stack(X, dim=0), torch.tensor(y, dtype=torch.int64)
        if backend == "sklearn":
            return np.stack(X, axis=0), np.asarray(y, dtype=np.int64)
        raise ValueError(f"Unsupported backend: {backend}")

    def train_LR_probe(self, train_data_tuple, val_data_tuple, layer=0):
        LOGGER.info("Training LR probe on torch")
        X_train, y_train = train_data_tuple
        X_val, y_val = val_data_tuple
        if np.unique(y_train).size < 2:
            raise ValueError("train_data must contain both classes for linear probe training.")
        probe = LogisticRegression(
            max_iter=2000,
            class_weight={0: 1.0, 1: self.pos_weight},
            random_state=0,
        )
        probe.fit(X_train, y_train)
        val_pred = probe.predict(X_val)
        val_prob = probe.predict_proba(X_val)[:, 1]
        metrics = {
            "layer": int(layer),
            "val_acc": float(accuracy_score(y_val, val_pred)),
            "val_f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "val_recall": float(recall_score(y_val, val_pred, zero_division=0)),
        }
        try:
            metrics["val_auc"] = float(roc_auc_score(y_val, val_prob))
        except ValueError:
            metrics["val_auc"] = float("nan")
        try:
            metrics["val_auprc"] = float(average_precision_score(y_val, val_prob))
        except ValueError:
            metrics["val_auprc"] = float("nan")

        return probe, metrics

    def train_linear_probe_torch(self, train_data_tuple, val_data_tuple, layer=0, epochs=50, lr=1e-3, batch_size=256, weight_decay=0.0):
        X_train, y_train = train_data_tuple
        X_val, y_val = val_data_tuple
        if isinstance(y_train, torch.Tensor):
            n_classes = int(torch.unique(y_train).numel())
        else:
            n_classes = int(np.unique(y_train).size)
        if n_classes < 2:
            raise ValueError("train_data must contain both classes for linear probe training.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)

        probe = torch.nn.Linear(X_train_t.shape[1], 1).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        pos_weight_t = torch.tensor([self.pos_weight], dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

        probe.train()
        n_train = X_train_t.shape[0]
        batch_size = max(1, min(int(batch_size), int(n_train)))
        for _ in range(int(epochs)):
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                logits = probe(X_train_t[idx])
                loss = criterion(logits, y_train_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val_t)
            val_prob = torch.sigmoid(val_logits).squeeze(1).detach().cpu().numpy()
            val_pred = (val_prob >= 0.5).astype(np.int64)
        y_val_np = y_val.detach().cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val

        metrics = {
            "layer": int(layer),
            "val_acc": float(accuracy_score(y_val_np, val_pred)),
            "val_f1": float(f1_score(y_val_np, val_pred, zero_division=0)),
            "val_precision": float(precision_score(y_val_np, val_pred, zero_division=0)),
            "val_recall": float(recall_score(y_val_np, val_pred, zero_division=0)),
        }
        try:
            metrics["val_auc"] = float(roc_auc_score(y_val_np, val_prob))
        except ValueError:
            metrics["val_auc"] = float("nan")
        try:
            metrics["val_auprc"] = float(average_precision_score(y_val_np, val_prob))
        except ValueError:
            metrics["val_auprc"] = float("nan")
        return probe, metrics

    def _predict_proba_with_classifier(self, classifier, X, backend: str):
        if backend == "torch":
            device = next(classifier.parameters()).device
            X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = classifier(X_t).squeeze(1)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            return probs
        if backend == "sklearn":
            X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
            return classifier.predict_proba(X_np)[:, 1]
        raise ValueError(f"Unsupported backend: {backend}")

    def predict_proba(self, X, classifier=None):
        if classifier is None:
            if self.best_classifier is None:
                raise ValueError("best_classifier is not initialized.")
            classifier = self.best_classifier
        backend = self.training_backend
        if backend is None:
            raise ValueError("training_backend is not set. Fit the probe before predicting.")
        return self._predict_proba_with_classifier(classifier, X, backend=backend)

    def predict_logits(self, X, classifier=None):
        if classifier is None:
            if self.best_classifier is None:
                raise ValueError("best_classifier is not initialized.")
            classifier = self.best_classifier
        backend = self.training_backend
        if backend != "torch":
            probs = self.predict_proba(X, classifier=classifier)
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            return np.log(probs / (1 - probs))

        device = next(classifier.parameters()).device
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = classifier(X_t).squeeze(1)
        return logits.detach().cpu().numpy()

    def should_trigger(self, X):
        probs = self.predict_proba(X)
        trigger = probs >= float(self.threshold)
        return trigger.astype(np.bool_)

    def _build_threshold_rows(self, y_true, probs, candidate_thresholds):
        rows = []
        for threshold in candidate_thresholds:
            pred = (probs >= float(threshold)).astype(np.int64)
            rows.append({
                "threshold": float(threshold),
                "trigger_rate": float(pred.mean()),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "acc": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
            })
        return rows

    def _select_threshold_by_f1(self, rows):
        best = None
        for row in rows:
            if best is None:
                best = row
                continue
            if row["f1"] > best["f1"]:
                best = row
            elif row["f1"] == best["f1"] and row["acc"] > best["acc"]:
                best = row
        return best

    def _select_threshold_by_trigger_budget(self, rows, budget: float):
        eligible = [row for row in rows if row["trigger_rate"] <= budget]
        if eligible:
            return max(
                eligible,
                key=lambda row: (row["recall"], row["precision"], row["f1"], -row["trigger_rate"]),
            )
        return min(
            rows,
            key=lambda row: (abs(row["trigger_rate"] - budget), -row["precision"], -row["recall"]),
        )

    def _select_threshold_by_min_precision(self, rows, min_precision: float):
        eligible = [row for row in rows if row["precision"] >= min_precision]
        if eligible:
            return max(
                eligible,
                key=lambda row: (row["recall"], row["f1"], -row["trigger_rate"], row["precision"]),
            )
        return max(rows, key=lambda row: (row["precision"], row["f1"], -row["trigger_rate"]))

    def calibrate_threshold(self, metric: str = "f1", split: str = "val"):
        if self.best_classifier is None or self.detection_layer is None:
            raise ValueError("Probe must be fitted before threshold calibration.")

        backend = self.training_backend or "torch"
        X_split, y_split = self.preprocess_data(self.detection_layer, split=split, backend=backend)
        probs = self.predict_proba(X_split)
        y_true = y_split.detach().cpu().numpy() if isinstance(y_split, torch.Tensor) else np.asarray(y_split)

        candidate_thresholds = np.unique(np.concatenate([
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            probs.astype(np.float32, copy=False),
        ]))
        rows = self._build_threshold_rows(y_true, probs, candidate_thresholds)
        metric_name = str(metric).lower().strip()
        if metric_name == "f1":
            best = self._select_threshold_by_f1(rows)
        else:
            budget_match = re.fullmatch(r"(?:trigger_)?budget[_@-]?([0-9]*\.?[0-9]+)", metric_name)
            precision_match = re.fullmatch(r"(?:min_)?precision[_@-]?([0-9]*\.?[0-9]+)", metric_name)
            if budget_match:
                best = self._select_threshold_by_trigger_budget(rows, float(budget_match.group(1)))
            elif precision_match:
                best = self._select_threshold_by_min_precision(rows, float(precision_match.group(1)))
            else:
                raise ValueError(f"Unsupported calibration metric: {metric}")

        assert best is not None
        self.threshold = best["threshold"]
        self.threshold_metric = metric
        self.calibration = {
            "split": split,
            "metric": metric,
            "threshold": best["threshold"],
            "f1": best["f1"],
            "acc": best["acc"],
            "precision": best["precision"],
            "recall": best["recall"],
            "trigger_rate": best["trigger_rate"],
            "n_samples": int(len(y_true)),
            "n_pos": int(np.sum(y_true == 1)),
            "n_neg": int(np.sum(y_true == 0)),
        }
        return self.calibration

    def export_artifact(self):
        if self.best_classifier is None or self.detection_layer is None:
            raise ValueError("Probe must be fitted before exporting an artifact.")

        if isinstance(self.best_classifier, torch.nn.Linear):
            weight = self.best_classifier.weight.detach().cpu()
            bias = self.best_classifier.bias.detach().cpu()
            classifier_type = "torch_linear"
        elif isinstance(self.best_classifier, LogisticRegression):
            weight = torch.as_tensor(self.best_classifier.coef_, dtype=torch.float32)
            bias = torch.as_tensor(self.best_classifier.intercept_, dtype=torch.float32)
            classifier_type = "sklearn_logreg"
        else:
            raise TypeError(f"Unsupported classifier type: {type(self.best_classifier)!r}")

        return {
            "classifier_type": classifier_type,
            "training_backend": self.training_backend,
            "feature_mode": self.feature_mode,
            "detection_layer": int(self.detection_layer),
            "input_dim": int(weight.shape[-1]),
            "weight": weight,
            "bias": bias.flatten(),
            "threshold": float(self.threshold),
            "threshold_metric": self.threshold_metric,
            "calibration": self.calibration,
            "steering_vec": None if self.steering_vec is None else self.steering_vec.detach().cpu(),
        }

    def save_artifact(self, path: str | Path):
        payload = self.export_artifact()
        torch.save(payload, Path(path))
        return payload

    @staticmethod
    def load_artifact(path: str | Path, map_location: str | torch.device = "cpu"):
        return torch.load(Path(path), map_location=map_location, weights_only=True)

    @staticmethod
    def build_classifier_from_artifact(payload, device: str | torch.device = "cpu"):
        classifier_type = payload["classifier_type"]
        input_dim = int(payload["input_dim"])
        linear = torch.nn.Linear(input_dim, 1)
        linear.weight.data.copy_(payload["weight"].reshape(1, input_dim))
        linear.bias.data.copy_(payload["bias"].reshape(1))
        linear.to(device=device, dtype=torch.float32)
        linear.eval()
        if classifier_type not in {"torch_linear", "sklearn_logreg"}:
            raise ValueError(f"Unsupported classifier_type in artifact: {classifier_type}")
        return linear

    @classmethod
    def from_artifact(cls, path: str | Path, device: str | torch.device = "cpu"):
        payload = cls.load_artifact(path, map_location="cpu")
        probe = cls.__new__(cls)
        probe.data = {}
        probe.best_classifier = cls.build_classifier_from_artifact(payload, device=device)
        probe.detection_layer = int(payload["detection_layer"])
        probe.probe_results = []
        probe.training_backend = str(payload["training_backend"])
        probe.threshold = float(payload["threshold"])
        probe.threshold_metric = payload.get("threshold_metric")
        probe.calibration = payload.get("calibration")
        probe.feature_mode = str(payload["feature_mode"])
        probe.steering_vec = payload.get("steering_vec")
        return probe
    
    def fit_scan_layers(self, layer_range, metric="val_acc", backend="torch"):
        # Scan through possible layers
        metric_aliases = {
            "auprc": "val_auprc",
            "aurpc": "val_auprc",
            "auc": "val_auc",
        }
        metric = metric_aliases.get(metric, metric)
        self.probe_results = []
        self.training_backend = backend
        best_metric = -float("inf")
        for layer in tqdm(layer_range, desc="Linear probing layers"):
            train_data_tuple = self.preprocess_data(layer, split="train", backend=backend)
            val_data_tuple = self.preprocess_data(layer, split="val", backend=backend)
            if backend == "sklearn":
                clf, clf_result = self.train_LR_probe(train_data_tuple, val_data_tuple, layer)
            elif backend == "torch":
                clf, clf_result = self.train_linear_probe_torch(train_data_tuple, val_data_tuple, layer)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            self.probe_results.append(clf_result)
            if self.best_classifier is None or clf_result[metric] > best_metric:
                self.best_classifier = clf
                self.detection_layer = layer
                best_metric = clf_result[metric]
        
        LOGGER.info("Finished layer scanning.")
        LOGGER.info("Best detection layer: %s", self.detection_layer)

    def compute_steering_vec(self, pos_data, neg_data, layer=None):
        if layer is None:
            if self.detection_layer is None:
                raise ValueError("detection_layer must be set before computing steering_vec.")
            layer = int(self.detection_layer)
        if self.feature_mode != "residual":
            raise ValueError(f"steering_vec currently only supports residual feature_mode, got {self.feature_mode}")
        if not pos_data or not neg_data:
            raise ValueError("Both positive and negative data are required to compute steering_vec.")

        pos_mean = torch.stack([row["residual"][layer].detach().float().cpu() for row in pos_data], dim=0).mean(dim=0)
        neg_mean = torch.stack([row["residual"][layer].detach().float().cpu() for row in neg_data], dim=0).mean(dim=0)
        self.steering_vec = pos_mean - neg_mean
        return self.steering_vec


class SparseSubspaceProbe(TaxProbe):

    def __init__(
        self,
        train_data,
        val_data,
        val_cal_data=None,
        feature_mode='residual',
        pos_weight: float = 1.0,
        *,
        num_subspaces: int = 2,
        subspace_rank: int = 32,
        orthogonality_weight: float = 1e-2,
        aggregator_l1_weight: float = 1e-3,
        epochs: int = 80,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
    ):
        super().__init__(train_data, val_data, feature_mode=feature_mode, pos_weight=pos_weight)
        if val_cal_data is not None:
            self.data["val_cal"] = val_cal_data
        self.num_subspaces = int(num_subspaces)
        self.subspace_rank = int(subspace_rank)
        self.orthogonality_weight = float(orthogonality_weight)
        self.aggregator_l1_weight = float(aggregator_l1_weight)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)

    @staticmethod
    def _quantile_summary(values, prefix: str) -> dict:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return {
                f"{prefix}_min": float("nan"),
                f"{prefix}_p10": float("nan"),
                f"{prefix}_p50": float("nan"),
                f"{prefix}_p90": float("nan"),
                f"{prefix}_max": float("nan"),
                f"{prefix}_mean": float("nan"),
                f"{prefix}_std": float("nan"),
            }
        qs = np.quantile(arr, [0.1, 0.5, 0.9])
        return {
            f"{prefix}_min": float(arr.min()),
            f"{prefix}_p10": float(qs[0]),
            f"{prefix}_p50": float(qs[1]),
            f"{prefix}_p90": float(qs[2]),
            f"{prefix}_max": float(arr.max()),
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_std": float(arr.std()),
        }

    @staticmethod
    def _compute_sparse_expert_logits(
        x_t: torch.Tensor,
        projectors: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_bias: torch.Tensor,
    ) -> torch.Tensor:
        expert_feats = torch.einsum("bd,kdr->bkr", x_t, projectors)
        expert_logits = (expert_feats * expert_weights.unsqueeze(0)).sum(dim=-1) + expert_bias.unsqueeze(0)
        return expert_logits

    @staticmethod
    def _subspace_orthogonality_penalty(projectors: torch.Tensor) -> torch.Tensor:
        q_list = []
        for proj in projectors:
            q, _ = torch.linalg.qr(proj, mode="reduced")
            q_list.append(q)
        penalty = torch.zeros((), device=projectors.device, dtype=projectors.dtype)
        for idx_i in range(len(q_list)):
            for idx_j in range(idx_i + 1, len(q_list)):
                cross = q_list[idx_i].transpose(0, 1) @ q_list[idx_j]
                penalty = penalty + (cross ** 2).sum()
        return penalty

    def _build_sparse_classifier(
        self,
        projectors: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_bias: torch.Tensor,
        aggregator_weight: torch.Tensor,
        aggregator_bias: torch.Tensor,
    ) -> dict:
        return {
            "classifier_type": "sparse_subspace",
            "projectors": projectors.detach().cpu(),
            "expert_weights": expert_weights.detach().cpu(),
            "expert_bias": expert_bias.detach().cpu(),
            "aggregator_weight": aggregator_weight.detach().cpu(),
            "aggregator_bias": aggregator_bias.detach().cpu(),
        }

    def _predict_sparse_logits(self, X, classifier=None):
        if classifier is None:
            if self.best_classifier is None:
                raise ValueError("best_classifier is not initialized.")
            classifier = self.best_classifier
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        projectors = torch.as_tensor(classifier["projectors"], dtype=torch.float32, device=device)
        expert_weights = torch.as_tensor(classifier["expert_weights"], dtype=torch.float32, device=device)
        expert_bias = torch.as_tensor(classifier["expert_bias"], dtype=torch.float32, device=device)
        aggregator_weight = torch.as_tensor(classifier["aggregator_weight"], dtype=torch.float32, device=device)
        aggregator_bias = torch.as_tensor(classifier["aggregator_bias"], dtype=torch.float32, device=device)

        expert_logits = self._compute_sparse_expert_logits(
            x_t,
            projectors,
            expert_weights,
            expert_bias,
        )
        final_logits = expert_logits @ aggregator_weight + aggregator_bias
        return final_logits.detach().cpu().numpy()

    def predict_proba(self, X, classifier=None):
        logits = self._predict_sparse_logits(X, classifier=classifier)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_logits(self, X, classifier=None):
        return self._predict_sparse_logits(X, classifier=classifier)

    def train_sparse_probe_torch(self, train_data_tuple, val_data_tuple, layer=0):
        X_train, y_train = train_data_tuple
        X_val, y_val = val_data_tuple
        if isinstance(y_train, torch.Tensor):
            n_classes = int(torch.unique(y_train).numel())
        else:
            n_classes = int(np.unique(y_train).size)
        if n_classes < 2:
            raise ValueError("train_data must contain both classes for sparse subspace probe training.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)

        input_dim = int(X_train_t.shape[1])
        rank = max(1, min(int(self.subspace_rank), input_dim))
        k = max(1, int(self.num_subspaces))

        projectors = torch.nn.Parameter(torch.empty(k, input_dim, rank, device=device))
        expert_weights = torch.nn.Parameter(torch.empty(k, rank, device=device))
        expert_bias = torch.nn.Parameter(torch.zeros(k, device=device))
        torch.nn.init.normal_(projectors, std=0.02)
        torch.nn.init.normal_(expert_weights, std=0.02)

        optimizer = torch.optim.AdamW(
            [projectors, expert_weights, expert_bias],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        pos_weight_t = torch.tensor([self.pos_weight], dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

        n_train = int(X_train_t.shape[0])
        batch_size = max(1, min(int(self.batch_size), n_train))
        expert_epoch_history: list[dict[str, float]] = []

        for epoch_idx in range(int(self.epochs)):
            perm = torch.randperm(n_train, device=device)
            epoch_total = 0.0
            epoch_bce = 0.0
            epoch_ortho = 0.0
            epoch_batches = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                x_batch = X_train_t[idx]
                y_batch = y_train_t[idx]

                expert_logits = self._compute_sparse_expert_logits(
                    x_batch,
                    projectors,
                    expert_weights,
                    expert_bias,
                )
                y_expand = y_batch.unsqueeze(1).expand_as(expert_logits)
                bce_loss = criterion(expert_logits.reshape(-1), y_expand.reshape(-1))
                ortho_penalty = self._subspace_orthogonality_penalty(projectors)
                loss = (
                    bce_loss
                    + self.orthogonality_weight * ortho_penalty
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_total += float(loss.detach().item())
                epoch_bce += float(bce_loss.detach().item())
                epoch_ortho += float(ortho_penalty.detach().item())
                epoch_batches += 1

            denom = max(1, epoch_batches)
            expert_epoch_history.append(
                {
                    "epoch": float(epoch_idx + 1),
                    "total_loss": epoch_total / denom,
                    "bce_loss": epoch_bce / denom,
                    "ortho_penalty": epoch_ortho / denom,
                }
            )

        with torch.no_grad():
            train_expert_logits = self._compute_sparse_expert_logits(
                X_train_t,
                projectors,
                expert_weights,
                expert_bias,
            )
            val_expert_logits = self._compute_sparse_expert_logits(
                X_val_t,
                projectors,
                expert_weights,
                expert_bias,
            )

        aggregator_weight = torch.nn.Parameter(torch.zeros(k, device=device))
        aggregator_bias = torch.nn.Parameter(torch.zeros((), device=device))
        aggregator_optimizer = torch.optim.AdamW(
            [aggregator_weight, aggregator_bias],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        aggregator_epoch_history: list[dict[str, float]] = []
        for epoch_idx in range(int(self.epochs)):
            perm = torch.randperm(n_train, device=device)
            epoch_total = 0.0
            epoch_bce = 0.0
            epoch_l1 = 0.0
            epoch_batches = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                expert_logit_batch = train_expert_logits[idx]
                y_batch = y_train_t[idx]

                final_logits = expert_logit_batch @ aggregator_weight + aggregator_bias
                bce_loss = criterion(final_logits, y_batch)
                agg_sparse_penalty = aggregator_weight.abs().sum()
                loss = (
                    bce_loss
                    + self.aggregator_l1_weight * agg_sparse_penalty
                )

                aggregator_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                aggregator_optimizer.step()

                epoch_total += float(loss.detach().item())
                epoch_bce += float(bce_loss.detach().item())
                epoch_l1 += float(agg_sparse_penalty.detach().item())
                epoch_batches += 1

            denom = max(1, epoch_batches)
            aggregator_epoch_history.append(
                {
                    "epoch": float(epoch_idx + 1),
                    "total_loss": epoch_total / denom,
                    "bce_loss": epoch_bce / denom,
                    "l1_penalty": epoch_l1 / denom,
                }
            )

        classifier = self._build_sparse_classifier(
            projectors=projectors,
            expert_weights=expert_weights,
            expert_bias=expert_bias,
            aggregator_weight=aggregator_weight,
            aggregator_bias=aggregator_bias,
        )

        with torch.no_grad():
            val_logits = val_expert_logits @ aggregator_weight + aggregator_bias
            val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_pred = (val_prob >= 0.5).astype(np.int64)
            train_prob = torch.sigmoid(train_expert_logits @ aggregator_weight + aggregator_bias).detach().cpu().numpy()
            val_logit_np = val_logits.detach().cpu().numpy()
            expert_logit_np = val_expert_logits.detach().cpu().numpy()
            expert_prob_np = torch.sigmoid(val_expert_logits).detach().cpu().numpy()
            aggregator_weight_np = aggregator_weight.detach().cpu().numpy()
        y_val_np = y_val.detach().cpu().numpy() if isinstance(y_val, torch.Tensor) else np.asarray(y_val)

        metrics = {
            "layer": int(layer),
            "val_acc": float(accuracy_score(y_val_np, val_pred)),
            "val_f1": float(f1_score(y_val_np, val_pred, zero_division=0)),
            "val_precision": float(precision_score(y_val_np, val_pred, zero_division=0)),
            "val_recall": float(recall_score(y_val_np, val_pred, zero_division=0)),
            "num_subspaces": int(k),
            "subspace_rank": int(rank),
            "aggregator_l1_weight": float(self.aggregator_l1_weight),
            "orthogonality_weight": float(self.orthogonality_weight),
            "aggregator_nnz": int((aggregator_weight.detach().abs().cpu().numpy() > 1e-4).sum()),
        }
        if expert_epoch_history:
            first = expert_epoch_history[0]
            last = expert_epoch_history[-1]
            metrics.update(
                {
                    "expert_train_total_loss_first": float(first["total_loss"]),
                    "expert_train_total_loss_last": float(last["total_loss"]),
                    "expert_train_bce_loss_first": float(first["bce_loss"]),
                    "expert_train_bce_loss_last": float(last["bce_loss"]),
                    "expert_train_ortho_penalty_first": float(first["ortho_penalty"]),
                    "expert_train_ortho_penalty_last": float(last["ortho_penalty"]),
                }
            )
            LOGGER.info(
                (
                    "Sparse probe layer=%s expert train summary: "
                    "total %.4f -> %.4f | bce %.4f -> %.4f | "
                    "ortho %.4f -> %.4f"
                ),
                layer,
                first["total_loss"],
                last["total_loss"],
                first["bce_loss"],
                last["bce_loss"],
                first["ortho_penalty"],
                last["ortho_penalty"],
            )
        if aggregator_epoch_history:
            first = aggregator_epoch_history[0]
            last = aggregator_epoch_history[-1]
            metrics.update(
                {
                    "agg_train_total_loss_first": float(first["total_loss"]),
                    "agg_train_total_loss_last": float(last["total_loss"]),
                    "agg_train_bce_loss_first": float(first["bce_loss"]),
                    "agg_train_bce_loss_last": float(last["bce_loss"]),
                    "agg_train_l1_penalty_first": float(first["l1_penalty"]),
                    "agg_train_l1_penalty_last": float(last["l1_penalty"]),
                }
            )
            LOGGER.info(
                (
                    "Sparse probe layer=%s aggregator train summary: "
                    "total %.4f -> %.4f | bce %.4f -> %.4f | "
                    "l1 %.4f -> %.4f"
                ),
                layer,
                first["total_loss"],
                last["total_loss"],
                first["bce_loss"],
                last["bce_loss"],
                first["l1_penalty"],
                last["l1_penalty"],
            )
        metrics.update(self._quantile_summary(train_prob, "train_final_prob"))
        metrics.update(self._quantile_summary(val_prob, "val_final_prob"))
        metrics.update(self._quantile_summary(val_logit_np, "val_final_logit"))
        metrics.update(self._quantile_summary(aggregator_weight_np, "aggregator_weight"))
        for expert_idx in range(k):
            metrics.update(self._quantile_summary(expert_logit_np[:, expert_idx], f"val_expert{expert_idx}_logit"))
            metrics.update(self._quantile_summary(expert_prob_np[:, expert_idx], f"val_expert{expert_idx}_prob"))
        LOGGER.info(
            "Sparse probe layer=%s aggregator weights=%s val_final_prob[p10=%.4f p50=%.4f p90=%.4f] val_final_logit[p10=%.4f p50=%.4f p90=%.4f]",
            layer,
            np.array2string(aggregator_weight_np, precision=4, separator=", "),
            metrics["val_final_prob_p10"],
            metrics["val_final_prob_p50"],
            metrics["val_final_prob_p90"],
            metrics["val_final_logit_p10"],
            metrics["val_final_logit_p50"],
            metrics["val_final_logit_p90"],
        )
        try:
            metrics["val_auc"] = float(roc_auc_score(y_val_np, val_prob))
        except ValueError:
            metrics["val_auc"] = float("nan")
        try:
            metrics["val_auprc"] = float(average_precision_score(y_val_np, val_prob))
        except ValueError:
            metrics["val_auprc"] = float("nan")
        return classifier, metrics

    def fit_scan_layers(self, layer_range, metric="val_acc", backend="torch"):
        metric_aliases = {
            "auprc": "val_auprc",
            "aurpc": "val_auprc",
            "auc": "val_auc",
        }
        metric = metric_aliases.get(metric, metric)
        self.probe_results = []
        self.training_backend = "torch"
        best_metric = -float("inf")
        for layer in tqdm(layer_range, desc="Sparse subspace probing layers"):
            train_data_tuple = self.preprocess_data(layer, split="train", backend="torch")
            val_data_tuple = self.preprocess_data(layer, split="val", backend="torch")
            clf, clf_result = self.train_sparse_probe_torch(train_data_tuple, val_data_tuple, layer)
            self.probe_results.append(clf_result)
            if self.best_classifier is None or clf_result[metric] > best_metric:
                self.best_classifier = clf
                self.detection_layer = layer
                best_metric = clf_result[metric]

        LOGGER.info("Finished sparse subspace layer scanning.")
        LOGGER.info("Best detection layer: %s", self.detection_layer)

    def export_artifact(self):
        if self.best_classifier is None or self.detection_layer is None:
            raise ValueError("Probe must be fitted before exporting an artifact.")
        payload = {
            "probe_type": "sparse_subspace",
            "training_backend": self.training_backend,
            "feature_mode": self.feature_mode,
            "detection_layer": int(self.detection_layer),
            "input_dim": int(self.best_classifier["projectors"].shape[1]),
            "threshold": float(self.threshold),
            "threshold_metric": self.threshold_metric,
            "calibration": self.calibration,
            "steering_vec": None,
            "num_subspaces": int(self.num_subspaces),
            "subspace_rank": int(self.subspace_rank),
            "orthogonality_weight": float(self.orthogonality_weight),
            "aggregator_l1_weight": float(self.aggregator_l1_weight),
        }
        payload.update(self.best_classifier)
        return payload

    @classmethod
    def from_artifact(cls, path: str | Path, device: str | torch.device = "cpu"):
        payload = cls.load_artifact(path, map_location="cpu")
        probe = cls.__new__(cls)
        probe.data = {}
        probe.best_classifier = {
            "classifier_type": "sparse_subspace",
            "projectors": payload["projectors"],
            "expert_weights": payload["expert_weights"],
            "expert_bias": payload["expert_bias"],
            "aggregator_weight": payload["aggregator_weight"],
            "aggregator_bias": payload["aggregator_bias"],
        }
        probe.detection_layer = int(payload["detection_layer"])
        probe.probe_results = []
        probe.training_backend = str(payload["training_backend"])
        probe.threshold = float(payload["threshold"])
        probe.threshold_metric = payload.get("threshold_metric")
        probe.calibration = payload.get("calibration")
        probe.feature_mode = str(payload["feature_mode"])
        probe.steering_vec = payload.get("steering_vec")
        probe.pos_weight = 1.0
        probe.num_subspaces = int(payload.get("num_subspaces", 2))
        probe.subspace_rank = int(payload.get("subspace_rank", 32))
        probe.orthogonality_weight = float(payload.get("orthogonality_weight", 1e-2))
        probe.aggregator_l1_weight = float(payload.get("aggregator_l1_weight", 1e-3))
        probe.epochs = 0
        probe.lr = 0.0
        probe.batch_size = 0
        probe.weight_decay = 0.0
        return probe


def load_probe_from_artifact(path: str | Path, device: str | torch.device = "cpu"):
    payload = TaxProbe.load_artifact(path, map_location="cpu")
    probe_type = str(payload.get("probe_type", "linear")).lower().strip()
    classifier_type = str(payload.get("classifier_type", "")).lower().strip()
    if probe_type == "sparse_subspace" or classifier_type == "sparse_subspace":
        return SparseSubspaceProbe.from_artifact(path, device=device)
    return TaxProbe.from_artifact(path, device=device)
