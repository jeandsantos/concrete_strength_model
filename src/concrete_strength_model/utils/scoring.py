import numpy as np
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

scorers = {
    "rmse": {"scorer": mean_squared_error, "transform": np.sqrt},
    "mae": {"scorer": mean_absolute_error},
    "mape": {"scorer": mean_absolute_percentage_error, "transform": lambda x: x * 100},
    "max_error": {"scorer": max_error},
    "r_squared": {"scorer": r2_score},
}


def calculate_scores(
    y, y_pred, scorers: dict, verbose: bool = False, prefix: str = "", suffix: str = ""
) -> dict:

    metrics = {}

    for name, scorer in scorers.items():

        name = prefix + name + suffix

        val = scorer["scorer"](y, y_pred)

        if "transform" in scorer.keys():
            val = scorer["transform"](val)

        metrics[name] = val

        if verbose:
            print(f"{name:<36} {val:.5f}")

    return metrics
