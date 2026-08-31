from dataclasses import dataclass, field
from typing import Any

from src.pso import APSO, APSOVI, PSOLVIW, PSOTVAC, UAPSO, PSOVectorized

DEFAULT_N_PARTICLES: int = 50


@dataclass
class HyperparametersConfig:
    n_particles: int = DEFAULT_N_PARTICLES

    PSO: dict[str, Any] = field(
        default_factory=lambda: {
            "w": 0.8,
            "c1": 2.0,
            "c2": 2.0,
            "n_particles": DEFAULT_N_PARTICLES,
        }
    )

    PSOLVIW: dict[str, Any] = field(
        default_factory=lambda: {
            "w_max": 0.9,
            "w_min": 0.4,
            "c1": 2.0,
            "c2": 2.0,
            "n_particles": DEFAULT_N_PARTICLES,
        }
    )

    PSOTVAC: dict[str, Any] = field(
        default_factory=lambda: {
            "w_max": 0.9,
            "w_min": 0.4,
            "c1_max": 2.5,
            "c1_min": 0.5,
            "c2_max": 2.5,
            "c2_min": 0.5,
            "n_particles": DEFAULT_N_PARTICLES,
        }
    )

    APSOVI: dict[str, Any] = field(
        default_factory=lambda: {
            "w_max": 0.9,
            "w_min": 0.3,
            "step_size": 0.1,
            "c1": 1.496180,
            "c2": 1.496180,
            "n_particles": DEFAULT_N_PARTICLES,
        }
    )

    APSO: dict[str, Any] = field(
        default_factory=lambda: {
            "w": 0.9,
            "c1": 2.0,
            "c2": 2.0,
            "n_particles": DEFAULT_N_PARTICLES,
        }
    )

    UAPSO: dict[str, Any] = field(
        default_factory=lambda: {
            "w_min": 0.0,
            "w_max": 1.0,
            "c1_min": 0.0,
            "c1_max": 2.0,
            "c2_min": 0.0,
            "c2_max": 2.0,
            "learning_rate": 0.01,
            "threshold": 0.5,
            "n_particles": DEFAULT_N_PARTICLES,
        }
    )


@dataclass
class Algorithms:
    PSO: PSOVectorized
    PSOLVIW: PSOLVIW
    PSOTVAC: PSOTVAC
    APSOVI: APSOVI
    APSO: APSO
    UAPSO: UAPSO
