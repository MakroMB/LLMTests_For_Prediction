import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


def run_sarima(
    train: pd.DataFrame,
    test: pd.DataFrame,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 5),  # sazonalidade semanal
) -> np.ndarray:
    """
    Treina SARIMA e retorna previsões para o período de teste.
    """
    y_train = train["y"].values

    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    horizon = len(test)
    forecast = fit.forecast(steps=horizon)
    return np.array(forecast)
