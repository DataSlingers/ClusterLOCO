from pathlib import Path
import json
from copy import deepcopy
N = 500 # base 200
BASE = {
    "K": 7,
    "n_per_cluster": N,
    "sim_seed": 123,
    "embed_seed": 123,
    "d0": 10,
    "informative_d": 10,
    "B": 5000, 
    "B_ramp":5000,
    "standardize":False,
    "oversample": 10,
    "gaps": [0.02] * 7,
    "shape_probs": {"moon": 0.5, "donut": 0.5},
}

BASE["topk"] = BASE["informative_d"]

DIFFICULTY_REGIMES = {
    "gaussian": {
        "easy": {"alpha": 2.2},
        "hard": {"alpha": 1.7},
    },
    "moon-donut": {
        "easy": {"alpha": 2.4},
        "hard": {"alpha": 1.2},
    },
    "gamma": {
        "easy": {"alpha": 10},
        "hard": {"alpha": 1},
    },
}

def make_exp(name, *, sim_method, noise_plan, difficulties=None, **kwargs):
    shared_cfg = deepcopy(BASE)
    shared_cfg.update(
        {
            "sim_method": sim_method,
            "noise_plan": noise_plan,
        }
    )
    shared_cfg.update(kwargs)

    if difficulties is None:
        difficulties = DIFFICULTY_REGIMES[sim_method]

    return {
        "name": name,
        "shared": shared_cfg,
        "difficulties": {
            difficulty_name: {
                **deepcopy(shared_cfg),
                **difficulty_overrides,
            }
            for difficulty_name, difficulty_overrides in difficulties.items()
        },
    }


EXPERIMENTS = {
    "gaussian_20": make_exp(
        "gaussian_20",
        d0 = 10,
        sim_method="gaussian",
        noise_plan=[{"type": "gaussian", "d": 10}],
    ),
    "gaussian_50": make_exp(
        "gaussian_50",
        d0 = 10,
        sim_method="gaussian",
        noise_plan=[{"type": "gaussian", "d": 40}],
    ),
    "gaussian_200": make_exp(
        "gaussian_200",
        d0 = 10,
        sim_method="gaussian",
        noise_plan=[{"type": "gaussian", "d": 190}],
    ),
    "gaussian_500": make_exp(
        "gaussian_500",
        d0 = 10,
        sim_method="gaussian",
        noise_plan=[{"type": "gaussian", "d": 490}],
    ),
    "gaussian_1000": make_exp(
        "gaussian_1000",
        d0 = 10,
        sim_method="gaussian",
        noise_plan=[{"type": "gaussian", "d": 990}],
    ),
    "moon_20": make_exp(
        "moon_20",
        d0 = 2,
        sim_method="moon-donut",
        shape_probs = {'donut':0.5, 'moon':0.5},
        noise_plan=[{"type": "uniform", "d": 10}],
    ),
    "moon_50": make_exp(
        "moon_50",
        d0 = 2,
        sim_method="moon-donut",
        shape_probs = {'donut':0.5, 'moon':0.5},
        noise_plan=[{"type": "uniform", "d": 40}],
    ),
    "moon_200": make_exp(
        "moon_200",
        d0 = 2,
        sim_method="moon-donut",
        shape_probs = {'donut':0.5, 'moon':0.5},
        noise_plan=[
            {"type": "gaussian", "d": 90},
            {"type": "uniform", "d": 100},
        ],
    ),
    "moon_500": make_exp(
        "moon_500",
        d0 = 2,
        sim_method="moon-donut",
        shape_probs = {'donut':0.5, 'moon':0.5},
        noise_plan=[
            {"type": "gaussian", "d": 200},
            {"type": "uniform", "d": 190},
            {"type": "permuted", "d": 100},
        ],
    ),
    "moon_1000": make_exp(
        "moon_1000",
        d0 = 2,
        sim_method="moon-donut",
        shape_probs = {'donut':0.5, 'moon':0.5},
        noise_plan=[
            {"type": "gaussian", "d": 290},
            {"type": "uniform", "d": 400},
            {"type": "permuted", "d": 300},
        ],
    ),
    "gamma_20": make_exp(
        "gamma_20",
        d0=10,
        sim_method="gamma",
        noise_plan=[{"type": "gamma", "d": 10, "shape": 2.0, "scale": 1.0}],
    ),
    "gamma_50": make_exp(
        "gamma_50",
        d0=10,
        sim_method="gamma",
        noise_plan=[{"type": "gamma", "d": 40, "shape": 2.0, "scale": 1.0}],
    ),
    "gamma_200": make_exp(
        "gamma_200",
        d0=10,
        sim_method="gamma",
        noise_plan=[{"type": "gamma", "d": 190, "shape": 2.0, "scale": 1.0}],
    ),
    "gamma_500": make_exp(
        "gamma_500",
        d0=10,
        sim_method="gamma",
        noise_plan=[{"type": "gamma", "d": 490, "shape": 2.0, "scale": 1.0}],
    ),
    "gamma_1000": make_exp(
        "gamma_1000",
        d0=10,
        sim_method="gamma",
        noise_plan=[{"type": "gamma", "d": 990, "shape": 2.0, "scale": 1.0}],
    ),
}


def write_config(path=f"./cfgs/experiments_N{N}.json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "difficulty_regimes": DIFFICULTY_REGIMES,
        "experiments": EXPERIMENTS,
    }

    with open(path, "w") as f:
        json.dump(config, f, indent=2)

    return str(path)


if __name__ == "__main__":
    print(write_config())