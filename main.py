from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmarks import BBQBenchmark, DiffAwareBenchmark
from prism import PRISM
from presets import DEFAULT_BENCHMARKS, DEFAULT_INTERVENTION, MODEL_PRESETS, SUPPORTED_MODELS
from train import METHOD_RESID_PROJECTION, build_run_name, build_run_paths, build_training_cfg, init_model, load_config, train
from utils import init_logger


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PRISM entrypoint")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yml"))
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--run-id", default="prism_release")
    parser.add_argument("--model-name", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--benchmarks", default="all", help="Comma-separated subset of: bbq,diffaware")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Overrides model_kwargs.device")
    parser.add_argument("--auto-layer", action="store_true", help="Enable residual projection auto-layer selection")
    parser.add_argument("--projection-layer", type=int, default=None, help="Overrides fixed residual projection layer")
    parser.add_argument("--layer-offset", type=int, default=None, help="Overrides residual projection layer offset")
    return parser.parse_args()


def parse_benchmarks(raw: str) -> list[str]:
    if raw == "all":
        return list(DEFAULT_BENCHMARKS)
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in DEFAULT_BENCHMARKS]
    if invalid:
        raise ValueError(f"Unsupported benchmarks: {invalid}. Allowed: {DEFAULT_BENCHMARKS}")
    return requested


def build_prism_cfg(args: argparse.Namespace) -> dict:
    base_cfg = load_config(args.config)
    model_cfg = MODEL_PRESETS[args.model_name]
    cfg = build_training_cfg(
        base_cfg,
        run_id=args.run_id,
        intervention=DEFAULT_INTERVENTION,
        model_name=args.model_name,
        model_id=model_cfg["model_id"],
        method=METHOD_RESID_PROJECTION,
    )
    cfg.update(model_cfg)

    if args.device:
        model_kwargs = dict(cfg.get("model_kwargs", {}))
        model_kwargs["device"] = args.device
        cfg["model_kwargs"] = model_kwargs

    if args.auto_layer:
        cfg["resid_projection_auto_layer"] = True
    if args.projection_layer is not None:
        cfg["resid_projection_layer"] = int(args.projection_layer)
    if args.layer_offset is not None:
        cfg["resid_projection_layer_offset"] = int(args.layer_offset)

    benchmark_root = Path(cfg["results_root"]) / "benchmark_results"
    cfg.setdefault("bbq_results_path", str(benchmark_root / "bbq"))
    cfg.setdefault("diffaware_results_path", str(benchmark_root / "diffaware"))
    cfg["benchmarks"] = parse_benchmarks(args.benchmarks)
    return cfg


def run_selected_benchmarks(cfg: dict, model, batch_size: int) -> None:
    prism_model = PRISM(cfg=cfg, run_id=cfg["run_id"], model=model, auto_build=False)
    model_logname = build_run_name(cfg)

    if "bbq" in cfg["benchmarks"]:
        LOGGER.info("starting benchmark: bbq")
        BBQBenchmark(cfg).run_bbq_test(prism_model, model_logname, batch_size=batch_size)
        LOGGER.info("finished benchmark: bbq")

    if "diffaware" in cfg["benchmarks"]:
        LOGGER.info("starting benchmark: diffaware")
        DiffAwareBenchmark(cfg, examine=False, load_default_subset=False).run_diffaware_test(
            prism_model,
            model_logname,
            batch_size=batch_size,
        )
        LOGGER.info("finished benchmark: diffaware")


def main() -> None:
    args = parse_args()
    cfg = build_prism_cfg(args)
    run_paths = build_run_paths(cfg)
    init_logger(Path(run_paths["logs_dir"]), cfg["run_id"])
    LOGGER.info("starting PRISM with mode=%s model=%s run_id=%s", args.mode, cfg["model_name"], cfg["run_id"])

    model = init_model(cfg)
    try:
        if args.mode in {"train", "all"}:
            LOGGER.info("training PRISM artifacts")
            train(cfg, model=model)

        if args.mode in {"eval", "all"}:
            LOGGER.info("running benchmarks: %s", ",".join(cfg["benchmarks"]))
            run_selected_benchmarks(cfg, model=model, batch_size=args.batch_size)
    finally:
        try:
            model.reset_hooks()
        except Exception:
            pass


if __name__ == "__main__":
    main()
