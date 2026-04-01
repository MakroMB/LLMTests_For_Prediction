# 📈 stock-forecast-benchmark

Benchmark comparativo de modelos de previsão de séries temporais para ações da B3.

## Modelos

| Modelo  | Tipo                     | Observações                                |
| ------- | ------------------------ | ------------------------------------------ |
| SARIMA  | Estatístico clássico     | `statsmodels`, sazonalidade semanal        |
| Prophet | Decomposição aditiva     | Meta/Facebook, robusto a feriados          |
| LSTM    | Deep Learning            | Via `neuralforecast` (PyTorch)             |
| Chronos | LLM temporal (zero-shot) | Amazon, `chronos-t5-tiny/small/base/large` |

## Métricas

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error

## Ações padrão

`PETR4.SA`, `VALE3.SA`, `ITUB4.SA` — escolhidas por perfis de volatilidade distintos.

## Instalação

```bash
pip install -r requirements.txt
```

> **Nota:** o Chronos requer instalação separada:
>
> ```bash
> pip install git+https://github.com/amazon-science/chronos-forecasting.git
> ```

## Uso

```bash
# Rodar tudo com defaults
python main.py

# Customizar tickers e horizon
python main.py --tickers PETR4.SA BBDC4.SA --horizon 60

# Pular modelos mais lentos
python main.py --skip-lstm --skip-chronos

# Escolher tamanho do Chronos
python main.py --chronos-size small
```

## Outputs

```
results/
├── PETR4_SA_forecast.png
├── VALE3_SA_forecast.png
├── ITUB4_SA_forecast.png
└── metrics.csv
```

## Estrutura

```
stock-forecast-benchmark/
├── data/
│   └── fetch.py           # Download via yfinance + split treino/teste
├── models/
│   ├── sarima.py
│   ├── prophet_model.py
│   ├── lstm.py
│   └── chronos_model.py
├── evaluation/
│   └── metrics.py         # MAE, RMSE, MAPE
├── results/               # Gráficos e CSV gerados
├── main.py
└── requirements.txt
```

## Referências

- [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting)
- [NeuralForecast (Nixtla)](https://github.com/Nixtla/neuralforecast)
- [Prophet (Meta)](https://facebook.github.io/prophet/)
