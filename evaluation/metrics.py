import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict:
    return {
        "model": model_name,
        "MAE": round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE (%)": round(mape(y_true, y_pred), 4),
    }


def results_table(results: list) -> pd.DataFrame:
    return (
        pd.DataFrame(results)
        .set_index("model")
        .sort_values("RMSE")
    )
