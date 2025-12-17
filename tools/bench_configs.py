"""
bench_configs.py: Default benchmark configuration grid sourced from settings.
"""

from niels_gpt.settings import default_settings


def get_default_grid() -> list[dict]:
    """
    Return the default benchmark grid based on settings.benchmark.
    """
    settings = default_settings()
    bench = settings.benchmark
    grid: list[dict] = []

    for T in bench.candidate_T:
        for model_cfg in bench.candidate_model_dims:
            for ckpt in bench.checkpointing_modes:
                d_ff = model_cfg.get("d_ff", 3 * model_cfg["C"])
                grid.append(
                    {
                        "T": T,
                        "C": model_cfg["C"],
                        "L": model_cfg["L"],
                        "H": model_cfg["H"],
                        "d_ff": d_ff,
                        "amp": True,
                        "amp_dtype": "fp16",
                        "activation_checkpointing": ckpt,
                    }
                )

    return grid
