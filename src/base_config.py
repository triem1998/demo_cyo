"""Shared base config and physics builder — imported by run.py and inference modules.

Kept in a separate file to avoid the circular import that would arise if inference
modules (imported by run.py) also imported from run.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from physics import MissingWedge


@dataclass
class RunEIBaseConfig:
    """Fields shared and identical between full-volume and patch training."""
    # ── Data ────────────────────────────────────────────────────────────────
    input_dir: str = "./dataset/empiar-11058"
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0

    # ── DataLoader (shared defaults) ─────────────────────────────────────────
    pin_memory: bool = True
    persistent_workers: bool = True

    # ── Physics ─────────────────────────────────────────────────────────────
    tilt_max: float = 60.0
    tilt_min: float = -60.0
    use_spherical_support: bool = True
    wedge_low_support: float = 0.0
    ref_wedge_support: float = 1.0

    # ── EI loss ─────────────────────────────────────────────────────────────
    eq_weight: float = 2.0
    loss_type: str = "icecream"   # "icecream" | "custom"

    # ── Training ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    grad_clip: float | None = 1.0
    ckp_interval: int = 10
    eval_interval: int = 1

    # ── Physics ─────────────────────────────────────────────────────────────
    wedge_double_size: bool = True

    # ── Mixed precision ──────────────────────────────────────────────────────
    use_mixed_precision: bool = True

    # ── Model ───────────────────────────────────────────────────────────────
    model_type: str = "unet"
    unet_dropout: float = 0.1
    drunet_sigma: float = 0.0

    # ── Evaluation ──────────────────────────────────────────────────────────
    fsc_threshold: float = 0.143
    pixel_size_angstrom: float | None = None

    # ── Pretrained init ──────────────────────────────────────────────────────
    pretrained_ckpt: str | None = None


def _build_physics(cfg: RunEIBaseConfig, crop_size: int, device) -> MissingWedge:
    return MissingWedge(
        tilt_max=float(cfg.tilt_max), tilt_min=float(cfg.tilt_min),
        crop_size=crop_size,
        use_spherical_support=bool(cfg.use_spherical_support),
        wedge_double_size=bool(cfg.wedge_double_size),
        wedge_low_support=float(cfg.wedge_low_support),
        ref_wedge_support=float(cfg.ref_wedge_support),
        device=str(device),
    ).to(device)
