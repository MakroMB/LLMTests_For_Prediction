import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MAE


def run_lstm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    ticker: str = "stock",
    horizon: int = 30,
    input_size: int = 60,
    max_steps: int = 200,
) -> np.ndarray:
    """
    Treina LSTM via NeuralForecast e retorna previsões para o período de teste.
    """
    # NeuralForecast espera colunas: unique_id, ds, y
    train_nf = train.copy()
    train_nf["unique_id"] = ticker

    model = LSTM(
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        max_steps=max_steps,
        scaler_type="standard",
    )

    nf = NeuralForecast(models=[model], freq="B")
    nf.fit(df=train_nf)

    forecast = nf.predict()
    preds = forecast["LSTM"].values[:horizon]
    return preds
