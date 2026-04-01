import yfinance as yf
import pandas as pd
from typing import List, Tuple


TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"


def fetch_stock_data(
    tickers: List[str] = TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict[str, pd.DataFrame]:
    """
    Baixa dados históricos do yfinance para uma lista de tickers.
    Retorna um dict {ticker: DataFrame} com colunas [ds, y]
    prontas para os modelos.
    """
    data = {}
    for ticker in tickers:
        print(f"Baixando {ticker}...")
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if raw.empty:
            print(f"  Aviso: sem dados para {ticker}")
            continue
        df = raw[["Close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.columns = ["y"]
        df.index.name = "ds"
        df = df.reset_index()
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
        df = df.dropna()
        data[ticker] = df
        print(f"  {len(df)} registros carregados.")
    return data


def train_test_split(df: pd.DataFrame, test_size:
                     int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa os últimos `test_size` dias como teste."""
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test
