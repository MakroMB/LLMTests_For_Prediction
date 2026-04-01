# 📈 B3 Forecast Benchmark — LLM vs Classical Models

> **English** | [Português](#-benchmark-de-previsão-b3--llm-vs-modelos-clássicos)

Benchmarking **SARIMA**, **Prophet**, **LSTM** and **Amazon Chronos** (zero-shot LLM) on Brazilian stock market data (B3), using 30-day price forecasts for PETR4, VALE3 and ITUB4.

The test period covers **November–December 2024** — a particularly challenging window marked by sharp BRL/USD depreciation and rising Selic expectations, causing a broad sell-off. An ideal stress test for forecast models.

---

## 📊 Results

| Model       | Ticker   | MAE      | RMSE     | MAPE (%) |
| ----------- | -------- | -------- | -------- | -------- |
| **Chronos** | PETR4.SA | **0.92** | **1.11** | **2.81** |
| **SARIMA**  | VALE3.SA | 1.10     | **1.45** | **2.23** |
| SARIMA      | PETR4.SA | 1.39     | 1.53     | 4.27     |
| Chronos     | ITUB4.SA | 1.47     | 1.78     | 6.02     |
| SARIMA      | ITUB4.SA | 1.50     | 1.77     | 6.14     |
| LSTM        | ITUB4.SA | 1.71     | 2.01     | 7.00     |
| Chronos     | VALE3.SA | 1.61     | 2.04     | 3.27     |
| Prophet     | PETR4.SA | 1.83     | 1.92     | 5.67     |
| LSTM        | PETR4.SA | 1.97     | 2.13     | 6.04     |
| Prophet     | VALE3.SA | 3.85     | 4.58     | 7.81     |
| Prophet     | ITUB4.SA | 3.90     | 4.09     | 15.78    |
| LSTM        | VALE3.SA | 5.12     | 5.26     | 10.29    |

### Key Takeaways

- **Chronos (zero-shot LLM)** achieved the best overall performance on PETR4 with no training required
- **SARIMA** was surprisingly competitive, with the best MAPE of the entire experiment on VALE3 (2.23%)
- **Prophet** consistently overestimated prices due to long-term trend bias — failed to react to the market regime change
- **LSTM** was unstable: decent on ITUB4, completely off on VALE3

---

## 📉 Forecast Plots

**PETR4.SA**

![PETR4](https://raw.githubusercontent.com/MakroMB/LLMTests_For_Prediction/main/results/PETR4_SA_forecast.png)

**VALE3.SA**

![VALE3](https://raw.githubusercontent.com/MakroMB/LLMTests_For_Prediction/main/results/VALE3_SA_forecast.png)

**ITUB4.SA**

![ITUB4](https://raw.githubusercontent.com/MakroMB/LLMTests_For_Prediction/main/results/ITUB4_SA_forecast.png)

---

## 🧠 Models

| Model   | Type                   | Notes                             |
| ------- | ---------------------- | --------------------------------- |
| SARIMA  | Classical statistical  | `statsmodels`, weekly seasonality |
| Prophet | Additive decomposition | Meta/Facebook                     |
| LSTM    | Deep Learning          | Via `neuralforecast` (PyTorch)    |
| Chronos | Zero-shot temporal LLM | Amazon `chronos-t5-base`          |

---

## 🚀 Getting Started

```bash
git clone https://github.com/MakroMB/LLMTests_For_Prediction.git
cd LLMTests_For_Prediction
pip install -r requirements.txt
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

```bash
# All models
python3 main.py

# Skip slower models
python3 main.py --skip-lstm --skip-chronos

# Custom tickers and horizon
python3 main.py --tickers PETR4.SA BBDC4.SA --horizon 60

# Choose Chronos size (tiny | small | base | large)
python3 main.py --chronos-size base
```

---

## 🗂️ Structure

```
LLMTests_For_Prediction/
├── data/fetch.py
├── models/
│   ├── sarima.py
│   ├── prophet_model.py
│   ├── lstm.py
│   └── chronos_model.py
├── evaluation/metrics.py
├── notebooks/analysis.ipynb
├── results/
├── main.py
└── requirements.txt
```

---

## 🛠️ Stack

`yfinance` · `prophet` · `statsmodels` · `neuralforecast` · `chronos-forecasting` · `PyTorch` · `pandas` · `matplotlib` · `plotly`

---

---

# 📈 Benchmark de Previsão B3 — LLM vs Modelos Clássicos

> [English](#-b3-forecast-benchmark--llm-vs-classical-models) | **Português**

Benchmark comparativo entre **SARIMA**, **Prophet**, **LSTM** e **Amazon Chronos** (LLM zero-shot) para previsão de preços de ações da B3, com horizonte de 30 dias para PETR4, VALE3 e ITUB4.

O período de teste cobre **novembro–dezembro de 2024** — marcado por forte depreciação do real e expectativas de alta da Selic, causando queda generalizada na bolsa. Um estresse real para os modelos.

---

## 📊 Resultados

| Modelo      | Ticker   | MAE      | RMSE     | MAPE (%) |
| ----------- | -------- | -------- | -------- | -------- |
| **Chronos** | PETR4.SA | **0.92** | **1.11** | **2.81** |
| **SARIMA**  | VALE3.SA | 1.10     | **1.45** | **2.23** |
| SARIMA      | PETR4.SA | 1.39     | 1.53     | 4.27     |
| Chronos     | ITUB4.SA | 1.47     | 1.78     | 6.02     |
| SARIMA      | ITUB4.SA | 1.50     | 1.77     | 6.14     |
| LSTM        | ITUB4.SA | 1.71     | 2.01     | 7.00     |
| Chronos     | VALE3.SA | 1.61     | 2.04     | 3.27     |
| Prophet     | PETR4.SA | 1.83     | 1.92     | 5.67     |
| LSTM        | PETR4.SA | 1.97     | 2.13     | 6.04     |
| Prophet     | VALE3.SA | 3.85     | 4.58     | 7.81     |
| Prophet     | ITUB4.SA | 3.90     | 4.09     | 15.78    |
| LSTM        | VALE3.SA | 5.12     | 5.26     | 10.29    |

### Conclusões principais

- **Chronos (LLM zero-shot)** melhor desempenho geral na PETR4, sem nenhum treinamento nos dados alvo
- **SARIMA** surpreendentemente competitivo, com o melhor MAPE do experimento na VALE3 (2.23%)
- **Prophet** superestimou sistematicamente os preços por viés de tendência — não reagiu à mudança de regime
- **LSTM** instável: razoável na ITUB4, completamente errado na VALE3

---

## 📉 Gráficos de Previsão

**PETR4.SA**

![PETR4](https://raw.githubusercontent.com/MakroMB/LLMTests_For_Prediction/main/results/PETR4_SA_forecast.png)

**VALE3.SA**

![VALE3](https://raw.githubusercontent.com/MakroMB/LLMTests_For_Prediction/main/results/VALE3_SA_forecast.png)

**ITUB4.SA**

![ITUB4](https://raw.githubusercontent.com/MakroMB/LLMTests_For_Prediction/main/results/ITUB4_SA_forecast.png)

---

## 🧠 Modelos

| Modelo  | Tipo                   | Observações                         |
| ------- | ---------------------- | ----------------------------------- |
| SARIMA  | Estatístico clássico   | `statsmodels`, sazonalidade semanal |
| Prophet | Decomposição aditiva   | Meta/Facebook                       |
| LSTM    | Deep Learning          | Via `neuralforecast` (PyTorch)      |
| Chronos | LLM temporal zero-shot | Amazon `chronos-t5-base`            |

---

## 🚀 Como rodar

```bash
git clone https://github.com/MakroMB/LLMTests_For_Prediction.git
cd LLMTests_For_Prediction
pip install -r requirements.txt
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

```bash
# Todos os modelos
python3 main.py

# Pular modelos mais lentos
python3 main.py --skip-lstm --skip-chronos

# Tickers e horizonte customizados
python3 main.py --tickers PETR4.SA BBDC4.SA --horizon 60

# Tamanho do Chronos (tiny | small | base | large)
python3 main.py --chronos-size base
```

---

## 🗂️ Estrutura

```
LLMTests_For_Prediction/
├── data/fetch.py
├── models/
│   ├── sarima.py
│   ├── prophet_model.py
│   ├── lstm.py
│   └── chronos_model.py
├── evaluation/metrics.py
├── notebooks/analysis.ipynb
├── results/
├── main.py
└── requirements.txt
```

---

## 🛠️ Stack

`yfinance` · `prophet` · `statsmodels` · `neuralforecast` · `chronos-forecasting` · `PyTorch` · `pandas` · `matplotlib` · `plotly`
