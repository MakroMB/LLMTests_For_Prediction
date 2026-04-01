import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline


def run_chronos(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_size: str = "small",
) -> np.ndarray:
    """
    Roda inferência com Amazon Chronos (zero-shot, sem fine-tuning).
    Retorna a mediana das amostras como previsão pontual.

    model_size:
        - "tiny"  → amazon/chronos-t5-tiny   (mais rápido, CPU ok)
        - "small" → amazon/chronos-t5-small
        - "base"  → amazon/chronos-t5-base
        - "large" → amazon/chronos-t5-large  (requer GPU)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = f"amazon/chronos-t5-{model_size}"

    print(f"Carregando Chronos ({model_id}) em {device}...")
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    context = torch.tensor(train["y"].values, dtype=torch.float32)
    horizon = len(test)

    # context é posicional na versão atual da API
    forecast = pipeline.predict(
        context,
        horizon,
        num_samples=20,
    )

    # forecast shape: (1, num_samples, horizon) → mediana
    median_forecast = np.median(forecast[0].numpy(), axis=0)
    return median_forecast
