import pandas as pd
import numpy as np
from prophet import Prophet
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def run_prophet(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    Treina Prophet e retorna previsões para o período de teste.
    Espera colunas [ds, y] no DataFrame.
    """
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(train[["ds", "y"]])

    future = model.make_future_dataframe(
        periods=len(test), freq="B")  # dias úteis
    forecast = model.predict(future)

    # Pega só as previsões do período de teste
    preds = forecast.tail(len(test))["yhat"].values
    return preds
