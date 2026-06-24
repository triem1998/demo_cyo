#!/usr/bin/env python3
"""Submitit/local launcher for cryo-ET demos.

Usage
-----
# Local run (uses GPU if available):
    python main.py --config configs/conf_equivariant_full_local.yml

# SLURM via submitit:
    python main.py --config configs/conf_equivariant_full.yml

"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import asdict
from pathlib import Path
import yaml
import submitit

# Allow `python main.py` from repo root without editable install.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run import RunEIFullConfig, RunEIPatchConfig  # noqa: E402
from run import run_full as run_training_ei_full, run_patch as run_training_ei_patch  # noqa: E402
from inference.infer_full import RunEIFullInferenceConfig  # noqa: E402
from inference.infer_full import run_inference as run_inference_ei  # noqa: E402
from inference.infer_patch import RunEIPatchInferenceConfig  # noqa: E402
from inference.infer_patch import run_inference as run_inference_patch  # noqa: E402


def _print_config(cfg, header: str = "RunConfig") -> None:
    lines = [f"[config] {header}"]
    for key, val in asdict(cfg).items():
        lines.append(f"  {key}: {val}")
    print("\n".join(lines), flush=True)


def _make_out_dir(general: dict, slurm: dict, default_name: str) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = str(general.get("run_name", slurm.get("job_name", default_name)))
    output_root = Path(general.get("output_root", "./runs"))
    return str(output_root / f"{run_name}_{timestamp}")


def _get_pixel_size(section: dict, field: str = "pixel_size_angstrom") -> float | None:
    v = section.get(field)
    return float(v) if v is not None else None


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

_METHODS: dict[str, tuple] = {
    "equivariant_full":   (RunEIFullConfig,          run_training_ei_full),
    "equivariant_patch":  (RunEIPatchConfig,          run_training_ei_patch),
    "ei_inference":       (RunEIFullInferenceConfig,  run_inference_ei),
    "ei_patch_inference": (RunEIPatchInferenceConfig, run_inference_patch),
}

# Required YAML sections per method
_REQUIRED_SECTIONS: dict[str, list[str]] = {
    "equivariant_full":   ["general", "training", "distributed", "slurm"],
    "equivariant_patch":  ["general", "training", "patch", "equivariant", "slurm"],
    "ei_inference":       ["general", "distributed", "slurm"],
    "ei_patch_inference": ["general", "slurm"],
}


# ---------------------------------------------------------------------------
# SLURM job callable (picklable)
# ---------------------------------------------------------------------------

class CryoTrainingJob:
    def __init__(self, method: str, cfg_dict: dict):
        self.method = method
        self.cfg_dict = cfg_dict
        self._src = str(Path(__file__).resolve().parent / "src")

    def __call__(self):
        import sys
        if self._src not in sys.path:
            sys.path.insert(0, self._src)

        submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )
        env = submitit.JobEnvironment()

        # Local imports after sys.path is set up — avoids capturing module
        # references in the pickle (which would fail on the worker node).
        if self.method == "equivariant_full":
            from run import RunEIFullConfig, run_full as run_fn
            cfg = RunEIFullConfig(**self.cfg_dict)
        elif self.method == "equivariant_patch":
            from run import RunEIPatchConfig, run_patch as run_fn
            cfg = RunEIPatchConfig(**self.cfg_dict)
        elif self.method == "ei_inference":
            from inference.infer_full import RunEIFullInferenceConfig, run_inference as run_fn
            cfg = RunEIFullInferenceConfig(**self.cfg_dict)
        elif self.method == "ei_patch_inference":
            from inference.infer_patch import RunEIPatchInferenceConfig, run_inference as run_fn
            cfg = RunEIPatchInferenceConfig(**self.cfg_dict)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        cfg.output_dir = str(Path(cfg.output_dir) / f"slurm-{env.job_id}")

        is_inference = self.method in ("ei_inference", "ei_patch_inference")
        if not is_inference:
            print(
                f"[submitit] job_id={env.job_id} rank={env.global_rank} "
                f"local_rank={env.local_rank} world_size={env.num_tasks}",
                flush=True,
            )
            if env.global_rank == 0:
                _print_config(cfg)

        return run_fn(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submitit/local launcher for equivariant cryo-ET demo."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs/conf_equivariant_full_local.yml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local mode regardless of general.execution_mode in config.",
    )
    return parser.parse_args()


def _require_section(conf: dict, name: str) -> dict:
    section = conf.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{name}' section in config.")
    return section


def load_config(path: str | Path) -> dict:
    conf_path = Path(path)
    if not conf_path.exists():
        raise FileNotFoundError(f"Config file not found: {conf_path}")

    with conf_path.open("r", encoding="utf-8") as f:
        conf = yaml.safe_load(f) or {}

    if not isinstance(conf, dict):
        raise ValueError("Top-level YAML config must be a dictionary.")

    method = str(conf.get("method", "equivariant_full")).lower()
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Supported: {', '.join(_METHODS)}.")

    for section_name in _REQUIRED_SECTIONS[method]:
        _require_section(conf, section_name)

    return conf


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _normalize_checkpoint_batches(value):
    if value is None:
        return None
    if str(value).lower() in {"none", "null"}:
        return None
    return value


def _parse_target_shape(value):
    if value is None:
        return None
    return tuple(int(v) for v in value)


def _build_equivariant_full_config(conf: dict) -> RunEIFullConfig:
    general     = _require_section(conf, "general")
    training    = _require_section(conf, "training")
    data        = conf.get("data", {})
    distributed = _require_section(conf, "distributed")
    equivariant = _require_section(conf, "equivariant")
    slurm       = _require_section(conf, "slurm")
    default     = RunEIFullConfig()

    return RunEIFullConfig(
        output_dir=_make_out_dir(general, slurm, "demo-cryo-ei-full"),
        input_dir=str(general.get("input_dir", default.input_dir)),
        max_train_vols=general.get("max_train_vols", default.max_train_vols),
        max_val_vols=int(general.get("max_val_vols", default.max_val_vols)),
        seed=int(general.get("seed", default.seed)),
        target_shape=_parse_target_shape(general.get("target_shape", default.target_shape)),
        batch_size=int(data.get("batch_size", default.batch_size)),
        num_workers=int(data.get("num_workers", default.num_workers)),
        pin_memory=bool(data.get("pin_memory", default.pin_memory)),
        prefetch_factor=int(data.get("prefetch_factor", default.prefetch_factor)),
        persistent_workers=bool(data.get("persistent_workers", default.persistent_workers)),
        tilt_max=float(equivariant.get("tilt_max", default.tilt_max)),
        tilt_min=float(equivariant.get("tilt_min", default.tilt_min)),
        use_spherical_support=bool(equivariant.get("use_spherical_support", default.use_spherical_support)),
        wedge_double_size=bool(equivariant.get("wedge_double_size", default.wedge_double_size)),
        eq_weight=float(equivariant.get("eq_weight", default.eq_weight)),
        patch_size=tuple(int(v) for v in distributed.get("patch_size", default.patch_size)),
        overlap=tuple(int(v) for v in distributed.get("overlap", default.overlap)),
        max_batch_size=distributed.get("max_batch_size", default.max_batch_size),
        checkpoint_batches=_normalize_checkpoint_batches(
            distributed.get("checkpoint_batches", default.checkpoint_batches)
        ),
        num_epochs=int(training.get("num_epochs", default.num_epochs)),
        learning_rate=float(training.get("learning_rate", default.learning_rate)),
        grad_clip=training.get("grad_clip", default.grad_clip),
        ckp_interval=int(training.get("ckp_interval", default.ckp_interval)),
        eval_interval=int(training.get("eval_interval", default.eval_interval)),
        grad_accumulation_steps=int(training.get("grad_accumulation_steps", default.grad_accumulation_steps)),
        use_mixed_precision=bool(training.get("use_mixed_precision", default.use_mixed_precision)),
        model_type=str(training.get("model_type", default.model_type)),
        unet_dropout=float(training.get("unet_dropout", default.unet_dropout)),
        drunet_sigma=float(training.get("drunet_sigma", default.drunet_sigma)),
        loss_type=str(equivariant.get("loss_type", default.loss_type)),
        wedge_low_support=float(equivariant.get("wedge_low_support", default.wedge_low_support)),
        ref_wedge_support=float(equivariant.get("ref_wedge_support", default.ref_wedge_support)),
        eval_fsc=bool(equivariant.get("eval_fsc", default.eval_fsc)),
        fsc_threshold=float(equivariant.get("fsc_threshold", default.fsc_threshold)),
        pixel_size_angstrom=_get_pixel_size(equivariant),
        pretrained_ckpt=training.get("pretrained_ckpt", default.pretrained_ckpt) or None,
    )


def _build_equivariant_patch_config(conf: dict) -> RunEIPatchConfig:
    general     = _require_section(conf, "general")
    training    = _require_section(conf, "training")
    patch       = _require_section(conf, "patch")
    equivariant = _require_section(conf, "equivariant")
    slurm       = _require_section(conf, "slurm")
    default     = RunEIPatchConfig()

    return RunEIPatchConfig(
        output_dir=_make_out_dir(general, slurm, "demo-cryo-ei-patch"),
        input_dir=str(general.get("input_dir", default.input_dir)),
        max_train_vols=general.get("max_train_vols", default.max_train_vols),
        max_val_vols=int(general.get("max_val_vols", default.max_val_vols)),
        seed=int(general.get("seed", default.seed)),
        crop_size=int(patch.get("crop_size", default.crop_size)),
        n_crops_per_vol=int(patch.get("n_crops_per_vol", default.n_crops_per_vol)),
        batch_size=int(patch.get("batch_size", default.batch_size)),
        num_workers=int(patch.get("num_workers", default.num_workers)),
        pin_memory=bool(patch.get("pin_memory", default.pin_memory)),
        prefetch_factor=int(patch.get("prefetch_factor", default.prefetch_factor)),
        persistent_workers=bool(patch.get("persistent_workers", default.persistent_workers)),
        normalize=bool(patch.get("normalize", default.normalize)),
        tilt_max=float(equivariant.get("tilt_max", default.tilt_max)),
        tilt_min=float(equivariant.get("tilt_min", default.tilt_min)),
        use_spherical_support=bool(equivariant.get("use_spherical_support", default.use_spherical_support)),
        wedge_double_size=bool(equivariant.get("wedge_double_size", default.wedge_double_size)),
        wedge_low_support=float(equivariant.get("wedge_low_support", default.wedge_low_support)),
        ref_wedge_support=float(equivariant.get("ref_wedge_support", default.ref_wedge_support)),
        eq_weight=float(equivariant.get("eq_weight", default.eq_weight)),
        loss_type=str(equivariant.get("loss_type", default.loss_type)),
        num_epochs=int(training.get("num_epochs", default.num_epochs)),
        learning_rate=float(training.get("learning_rate", default.learning_rate)),
        grad_clip=training.get("grad_clip", default.grad_clip),
        ckp_interval=int(training.get("ckp_interval", default.ckp_interval)),
        eval_interval=int(training.get("eval_interval", default.eval_interval)),
        grad_accumulation_steps=int(training.get("grad_accumulation_steps", default.grad_accumulation_steps)),
        log_every_n_epochs=int(training.get("log_every_n_epochs", default.log_every_n_epochs)),
        infer_stride=int(training.get("infer_stride", default.infer_stride)),
        infer_batch_size=int(training.get("infer_batch_size", default.infer_batch_size)),
        infer_downsample=int(training.get("infer_downsample", default.infer_downsample)),
        infer_train=bool(training.get("infer_train", default.infer_train)),
        infer_val=bool(training.get("infer_val", default.infer_val)),
        save_mrc=bool(training.get("save_mrc", default.save_mrc)),
        use_mixed_precision=bool(training.get("use_mixed_precision", default.use_mixed_precision)),
        model_type=str(training.get("model_type", default.model_type)),
        unet_dropout=float(training.get("unet_dropout", default.unet_dropout)),
        drunet_sigma=float(training.get("drunet_sigma", default.drunet_sigma)),
        eval_fsc=bool(equivariant.get("eval_fsc", default.eval_fsc)),
        fsc_threshold=float(equivariant.get("fsc_threshold", default.fsc_threshold)),
        pixel_size_angstrom=_get_pixel_size(equivariant),
    )


def _build_ei_inference_config(conf: dict) -> RunEIFullInferenceConfig:
    general     = _require_section(conf, "general")
    distributed = _require_section(conf, "distributed")
    inference   = conf.get("inference", {})
    slurm       = _require_section(conf, "slurm")
    default     = RunEIFullInferenceConfig()

    return RunEIFullInferenceConfig(
        checkpoint_path=str(inference.get("checkpoint_path", default.checkpoint_path)),
        output_dir=_make_out_dir(general, slurm, "demo-cryo-ei-inference"),
        input_dir=str(general.get("input_dir", default.input_dir)),
        max_infer_vols=int(general.get("max_infer_vols", default.max_infer_vols)),
        seed=int(general.get("seed", default.seed)),
        target_shape=_parse_target_shape(general.get("target_shape", default.target_shape)),
        num_workers=int(inference.get("num_workers", default.num_workers)),
        pin_memory=bool(inference.get("pin_memory", default.pin_memory)),
        prefetch_factor=int(inference.get("prefetch_factor", default.prefetch_factor)),
        persistent_workers=bool(inference.get("persistent_workers", default.persistent_workers)),
        tilt_max=float(inference.get("tilt_max", default.tilt_max)),
        tilt_min=float(inference.get("tilt_min", default.tilt_min)),
        use_spherical_support=bool(inference.get("use_spherical_support", default.use_spherical_support)),
        wedge_double_size=bool(inference.get("wedge_double_size", default.wedge_double_size)),
        wedge_low_support=float(inference.get("wedge_low_support", default.wedge_low_support)),
        ref_wedge_support=float(inference.get("ref_wedge_support", default.ref_wedge_support)),
        patch_size=tuple(int(v) for v in distributed.get("patch_size", default.patch_size)),
        overlap=tuple(int(v) for v in distributed.get("overlap", default.overlap)),
        max_batch_size=distributed.get("max_batch_size", default.max_batch_size),
        checkpoint_batches=_normalize_checkpoint_batches(
            distributed.get("checkpoint_batches", default.checkpoint_batches)
        ),
        model_type=str(inference.get("model_type", default.model_type)),
        unet_dropout=float(inference.get("unet_dropout", default.unet_dropout)),
        drunet_sigma=float(inference.get("drunet_sigma", default.drunet_sigma)),
        fsc_threshold=float(inference.get("fsc_threshold", default.fsc_threshold)),
        pixel_size_angstrom=_get_pixel_size(inference),
        icecream_glob=str(inference.get("icecream_glob", default.icecream_glob)),
        isonet_glob=str(inference.get("isonet_glob", default.isonet_glob)),
        isonet_fallback_glob=str(inference.get("isonet_fallback_glob", default.isonet_fallback_glob)),
        save_recon_mrc=bool(inference.get("save_recon_mrc", default.save_recon_mrc)),
    )


def _build_ei_patch_inference_config(conf: dict) -> RunEIPatchInferenceConfig:
    general   = _require_section(conf, "general")
    inference = conf.get("inference", {})
    slurm     = _require_section(conf, "slurm")
    default   = RunEIPatchInferenceConfig()

    return RunEIPatchInferenceConfig(
        checkpoint_path=str(inference.get("checkpoint_path", default.checkpoint_path)),
        output_dir=_make_out_dir(general, slurm, "demo-cryo-ei-patch-inference"),
        input_dir=str(general.get("input_dir", default.input_dir)),
        max_infer_vols=int(general.get("max_infer_vols", default.max_infer_vols)),
        seed=int(general.get("seed", default.seed)),
        num_workers=int(inference.get("num_workers", default.num_workers)),
        pin_memory=bool(inference.get("pin_memory", default.pin_memory)),
        prefetch_factor=int(inference.get("prefetch_factor", default.prefetch_factor)),
        persistent_workers=bool(inference.get("persistent_workers", default.persistent_workers)),
        normalize=bool(inference.get("normalize", default.normalize)),
        tilt_max=float(inference.get("tilt_max", default.tilt_max)),
        tilt_min=float(inference.get("tilt_min", default.tilt_min)),
        use_spherical_support=bool(inference.get("use_spherical_support", default.use_spherical_support)),
        wedge_double_size=bool(inference.get("wedge_double_size", default.wedge_double_size)),
        wedge_low_support=float(inference.get("wedge_low_support", default.wedge_low_support)),
        ref_wedge_support=float(inference.get("ref_wedge_support", default.ref_wedge_support)),
        model_type=str(inference.get("model_type", default.model_type)),
        unet_dropout=float(inference.get("unet_dropout", default.unet_dropout)),
        drunet_sigma=float(inference.get("drunet_sigma", default.drunet_sigma)),
        crop_size=int(inference.get("crop_size", default.crop_size)),
        stride=int(inference.get("stride", default.stride)),
        infer_batch_size=int(inference.get("infer_batch_size", default.infer_batch_size)),
        infer_downsample=int(inference.get("infer_downsample", default.infer_downsample)),
        pre_pad=bool(inference.get("pre_pad", default.pre_pad)),
        fsc_threshold=float(inference.get("fsc_threshold", default.fsc_threshold)),
        pixel_size_angstrom=_get_pixel_size(inference),
        icecream_glob=str(inference.get("icecream_glob", default.icecream_glob)),
        isonet_glob=str(inference.get("isonet_glob", default.isonet_glob)),
        isonet_fallback_glob=str(inference.get("isonet_fallback_glob", default.isonet_fallback_glob)),
        save_recon_mrc=bool(inference.get("save_recon_mrc", default.save_recon_mrc)),
    )


_BUILDERS = {
    "equivariant_full":   _build_equivariant_full_config,
    "equivariant_patch":  _build_equivariant_patch_config,
    "ei_inference":       _build_ei_inference_config,
    "ei_patch_inference": _build_ei_patch_inference_config,
}


def build_run_config(conf: dict):
    method = str(conf.get("method", "equivariant_full")).lower()
    if method not in _BUILDERS:
        raise ValueError(f"Unknown method '{method}'. Supported: {', '.join(_BUILDERS)}.")
    return method, _BUILDERS[method](conf)


# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------

def submit_job(method: str, cfg, slurm: dict) -> None:
    submitit_folder = Path(cfg.output_dir) / "submitit_logs"
    submitit_folder.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(submitit_folder), slurm_python="python")
    gpus_per_node = int(slurm.get("gpus_per_node", 1))
    additional_params = dict(slurm.get("additional_parameters", {}))
    additional_params.update(
        {
            "ntasks-per-node": int(slurm.get("ntasks_per_node", 1)),
            "cpus-per-task":   int(slurm.get("cpus_per_task", 4)),
            "account":         str(slurm.get("account", "fio@h100")),
            "constraint":      str(slurm.get("constraint", "h100")),
            "qos":             str(slurm.get("qos", "qos_gpu_h100-dev")),
        }
    )

    executor.update_parameters(
        name=str(slurm.get("job_name", "demo-cryo-ei")),
        nodes=int(slurm.get("nodes", 1)),
        slurm_gres=str(slurm.get("gres", f"gpu:{gpus_per_node}")),
        slurm_time=str(slurm.get("time", "02:00:00")),
        slurm_stderr_to_stdout=bool(slurm.get("stderr_to_stdout", True)),
        slurm_additional_parameters=additional_params,
        slurm_setup=list(
            slurm.get(
                "setup",
                [
                    "module purge",
                    "module load arch/h100",
                    "module load pytorch-gpu/py3/2.7.0",
                    "export NCCL_DEBUG=INFO",
                ],
            )
        ),
    )

    job = executor.submit(CryoTrainingJob(method, asdict(cfg)))
    print(f"Submitted job: {job.job_id}")
    print(f"Submitit logs: {submitit_folder.resolve()}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    conf = load_config(args.config)
    general = _require_section(conf, "general")
    slurm   = _require_section(conf, "slurm")

    execution_mode = str(general.get("execution_mode", "local")).lower()
    if args.local:
        execution_mode = "local"

    method, cfg = build_run_config(conf)

    if execution_mode == "local":
        _print_config(cfg)
        _, run_fn = _METHODS[method]
        run_fn(cfg)
        return

    submit_job(method, cfg, slurm)


if __name__ == "__main__":
    main()
