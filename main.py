"""
stock-forecast-benchmark
Compara SARIMA, Prophet, LSTM e Amazon Chronos em ações da B3.

Uso:
    python main.py
    python main.py --tickers PETR4.SA VALE3.SA --horizon 30
"""

import argparse
import os
import matplotlib.pyplot as plt

from data.fetch import fetch_stock_data, train_test_split
from models.sarima import run_sarima
from models.prophet_model import run_prophet  # noqa: E402
from models.lstm import run_lstm
from models.chronos_model import run_chronos  # noqa: E402
from evaluation.metrics import evaluate, results_table


def parse_args():
    parser = argparse.ArgumentParser(description="Stock Forecast Benchmark")
    parser.add_argument(
        "--tickers", nargs="+", default=["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--horizon", type=int,
                        default=30, help="Dias de teste")
    parser.add_argument(
        "--skip-lstm", action="store_true", help="Pular LSTM (mais lento)"
    )
    parser.add_argument(
        "--skip-chronos", action="store_true", help="Pular Chronos")
    parser.add_argument(
        "--chronos-size",
        default="tiny",
        choices=["tiny", "small", "base", "large"],
    )
    return parser.parse_args()


def plot_forecast(ticker, train, test, forecasts: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        train["ds"].tail(90), train["y"].tail(90),
        label="Histórico", color="gray", linewidth=1,
    )
    ax.plot(test["ds"], test["y"], label="Real", color="black", linewidth=2)

    colors = ["#e63946", "#2a9d8f", "#e9c46a", "#457b9d"]
    for (name, preds), color in zip(forecasts.items(), colors):
        ax.plot(
            test["ds"], preds,
            label=name, linestyle="--", color=color, linewidth=1.5,
        )

    ax.set_title(f"{ticker} — Previsão {len(test)} dias")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (R$)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Gráfico salvo: {save_path}")


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    print("=== Baixando dados ===")
    all_data = fetch_stock_data(args.tickers, args.start, args.end)

    all_results = []

    for ticker, df in all_data.items():
        print(f"\n=== {ticker} ===")
        train, test = train_test_split(df, test_size=args.horizon)
        y_true = test["y"].values
        forecasts = {}

        # SARIMA
        print("  Rodando SARIMA...")
        sarima_preds = run_sarima(train, test)
        forecasts["SARIMA"] = sarima_preds
        all_results.append(
            evaluate(y_true, sarima_preds, f"{ticker} — SARIMA"))

        # Prophet
        print("  Rodando Prophet...")
        prophet_preds = run_prophet(train, test)
        forecasts["Prophet"] = prophet_preds
        all_results.append(
            evaluate(y_true, prophet_preds, f"{ticker} — Prophet")
        )

        # LSTM
        if not args.skip_lstm:
            print("  Rodando LSTM...")
            lstm_preds = run_lstm(
                train, test, ticker=ticker, horizon=args.horizon
            )
            forecasts["LSTM"] = lstm_preds
            all_results.append(
                evaluate(y_true, lstm_preds, f"{ticker} — LSTM")
            )

        # Chronos
        if not args.skip_chronos:
            print("  Rodando Chronos...")
            chronos_preds = run_chronos(
                train, test, model_size=args.chronos_size
            )
            forecasts["Chronos"] = chronos_preds
            all_results.append(
                evaluate(y_true, chronos_preds, f"{ticker} — Chronos")
            )

        plot_forecast(
            ticker, train, test, forecasts,
            f"results/{ticker.replace('.', '_')}_forecast.png",
        )

    print("\n=== Resultados Finais ===")
    df_results = results_table(all_results)
    print(df_results.to_string())

    df_results.to_csv("results/metrics.csv")
    print("\nMétricas salvas em results/metrics.csv")


if __name__ == "__main__":
    main()
