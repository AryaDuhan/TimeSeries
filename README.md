## 📈 Stock Price Prediction - LSTM vs. Prophet

A comprehensive **machine learning project** that predicts stock prices using two powerful time-series forecasting models: **Long Short-Term Memory (LSTM)** and **Prophet**.

This repository includes two main components:

-   A **fully interactive Streamlit dashboard** for real-time predictions and model comparison.
-   A **Jupyter Notebook (`forecasting.ipynb`)** and Python script (`forecasting.py`) for a deeper, standalone analysis.

The project demonstrates an end-to-end data science workflow, from data fetching and preprocessing to model training, evaluation, and visualization.

---

## 🎨 Interactive Streamlit App

### ✅ Model Comparison Dashboard

The app provides a **side-by-side comparison** of the LSTM and Prophet models, allowing you to configure parameters and see the results instantly.

<table align="center">
  <tr>
    <td align="center"><b>Dashboard Configuration</b></td>
    <td align="center"><b>Prediction Visualization</b></td>
  </tr>
  <tr>
    <td>
      <i></i>
    </td>
    <td>
      <i></i>
    </td>
  </tr>
</table>

**Key Interface Features:**

-   **Dynamic Ticker Input** – Analyze any stock ticker symbol (e.g., AAPL, GOOGL, TSLA).
-   **Configurable Parameters** – Adjust the date range, training split, and LSTM hyperparameters.
-   **Real-time Predictions** – On-the-fly model training and forecasting.
-   **Performance Metrics** – Instantly compare model performance using Root Mean Squared Error (RMSE).
-   **Interactive Charts** – Visualize historical prices against LSTM and Prophet predictions.

---

## ✨ Key Features

-   **Dual-Model Forecasting** – Leverages both a deep learning (LSTM) and a statistical (Prophet) model.
-   **End-to-End Workflow** – Covers data fetching (`yfinance`), preprocessing, training, and visualization.
-   **Interactive Web App** – Built with Streamlit for a seamless user experience.
-   **Performance Evaluation** – Automatically determines and highlights the better-performing model.
-   **Clean Codebase** – Includes a standalone script, a Jupyter Notebook for analysis, and a modular Streamlit app.

---

## 🏗 Architectural Comparison

This project implements and compares two popular and effective time-series forecasting models.

### 🔹 Model 1: Long Short-Term Memory (LSTM)

**Architecture**:

-   A type of Recurrent Neural Network (RNN) designed to remember patterns over long sequences.
-   **PyTorch** is used for the implementation.

✅ **Pros**: Excellent at capturing complex, non-linear patterns in sequential data.
❌ **Cons**: Can be computationally expensive and requires careful tuning of hyperparameters (time steps, epochs, hidden size).

---

### 🔹 Model 2: Prophet

**Architecture**:

-   A decomposable time series model developed by Facebook.
-   It models trends, seasonality, and holidays.

✅ **Pros**: Fast, robust to missing data, and works well with time series that have strong seasonal patterns. Generally requires less tuning than LSTMs.
❌ **Cons**: May not capture complex non-linear relationships as effectively as an LSTM.

---

### 📊 Performance at a Glance

| Feature              | LSTM Model                      | Prophet Model                      |
| -------------------- | ------------------------------- | ---------------------------------- |
| **Model Type** | Deep Learning (RNN)             | Decomposable Statistical Model     |
| **Best For** | Complex, non-linear patterns    | Trends and seasonality             |
| **Training Speed** | Slower                          | Faster                             |
| **Implementation** | PyTorch                         | Prophet (formerly Facebook Prophet)|
| **Key Metric** | Root Mean Squared Error (RMSE)  | Root Mean Squared Error (RMSE)     |

---

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/TimeSeries.git](https://github.com/your-username/TimeSeries.git)
cd TimeSeries
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run stock_predictor_app.py
```
### 📊 Performance at a Glance
The following is a sample output from running the analysis in forecasting.ipynb for the AAPL ticker:

```bash
============================================================
MODEL PERFORMANCE METRICS
============================================================
LSTM Train RMSE:    $19.54
LSTM Test RMSE:     $61.82
Prophet Test RMSE:  $16.27

Best Model: Prophet (RMSE difference: $45.56)
```

### 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.
