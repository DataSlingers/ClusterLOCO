from pathlib import Path
import json
from copy import deepcopy

BASE = {
    "K": 7,
    "n_per_cluster": 200,
    "alpha": 7, # for gamma # 2, # for moon-donut
    "sim_seed":123,
    "embed_seed":123,
    "d0": 10, # 2 for moon-donut
    "informative_d": 10,
    "oversample": 10,
    "gaps": [0.02] * 7,
    "shape_probs": {"moon": 0.5, "donut": 0.5},
}

BASE['topk'] = BASE['informative_d'] 

def make_exp(name, **kwargs):
    cfg = deepcopy(BASE)
    cfg.update(kwargs)
    return {"name": name, "experiment": cfg}

EXPERIMENTS = {
    # "moon_20": make_exp(
    #     "moon_20",
    #     sim_method="moon-donut",
    #     noise_plan=[{'type':'gaussian','d':5},{'type':'permuted','d':5}]
    # ),
    # "moon_50": make_exp(
    #     "moon_50",
    #     sim_method="moon-donut",
    #     noise_plan=[{'type':'gaussian','d':20}, {'type':'permuted', 'd':20}],
    # ),
    # "moon_200": make_exp(
    #     "moon_200",
    #     sim_method="moon-donut",
    #     noise_plan = [{'type':'gaussian','d':95}, {'type':'permuted', 'd':95}]
    # ),
    "gamma_20": make_exp(
        "gamma_20", 
        sim_method='gamma',
        noise_plan = [{'type':'permuted','d':5},{'type':'gamma','d':5}]
    ),
    "gamma_50": make_exp(
        "gamma_50",
        sim_method='gamma',
        noise_plan = [{'type':'permuted', 'd':20}, {'type':'gamma', 'd':20}] 
    ),
    "gamma_200": make_exp(
        "gamma_200",
        sim_method='gamma',
        noise_plan = [{'type':'permuted', 'd':95}, {'type':'gamma', 'd':95}] 
    ),
}

def write_configs(out_dir="./cfgs/"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, cfg in EXPERIMENTS.items():
        path = out / f"{name}.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        paths[name] = str(path)
    return paths

if __name__ == "__main__":
    write_configs()