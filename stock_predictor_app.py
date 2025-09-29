import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide", page_icon="📈")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📈 Stock Price Prediction Dashboard</p>', unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, GOOGL, MSFT)")
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
train_split = st.sidebar.slider("Training Data Split (%)", min_value=50, max_value=90, value=80, step=5)
time_step = st.sidebar.slider("LSTM Time Steps", min_value=30, max_value=150, value=100, step=10, help="Number of previous days to use for prediction")
lstm_epochs = st.sidebar.slider("LSTM Training Epochs", min_value=10, max_value=100, value=50, step=10)
hidden_size = st.sidebar.slider("LSTM Hidden Size", min_value=50, max_value=200, value=100, step=25)

# LSTM Model Definition
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Function to create sequences for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Cache data fetching to avoid repeated downloads
@st.cache_data
def fetch_stock_data(ticker_symbol, start, end):
    try:
        data = yf.download(ticker_symbol, start=start, end=end, progress=False)
        if data.empty:
            return None
        data = data[['Close']].reset_index()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to load a pre-trained model
def load_pretrained_model(model_path, hidden_size):
    model = StockLSTM(hidden_layer_size=hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Cache model training to avoid retraining on every interaction
@st.cache_resource
def train_lstm_model(_X_train_tensor, _y_train_tensor, hidden_size, epochs):
    model = StockLSTM(hidden_layer_size=hidden_size)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(_X_train_tensor)
        loss = loss_function(y_pred, _y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            progress_bar.progress((i + 1) / epochs)
            status_text.text(f'Training LSTM: Epoch {i + 1}/{epochs}, Loss: {loss.item():.6f}')
    
    progress_bar.empty()
    status_text.empty()
    return model

# Main execution
if st.sidebar.button("Run Prediction", type="primary"):
    with st.spinner("Fetching stock data..."):
        data = fetch_stock_data(ticker, start_date, end_date)
    
    if data is None or len(data) < time_step + 50:
        st.error("❌ Unable to fetch sufficient data. Please check the ticker symbol or date range.")
        st.stop()
    
    st.success(f"✅ Successfully fetched {len(data)} days of data for {ticker}")
    
    # Data preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    training_size = int(len(data_scaled) * (train_split / 100))
    test_size = len(data_scaled) - training_size
    train_data = data_scaled[0:training_size, :]
    test_data = data_scaled[training_size:len(data_scaled), :1]
    
    # Create datasets
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    if len(X_test) == 0:
        st.error("❌ Insufficient test data. Please adjust the date range or training split.")
        st.stop()
    
    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)
    
    # Train or load LSTM model
    st.subheader("🤖 Loading/Training LSTM Model")
    model_path = "aapl_lstm_model.pth"
    if ticker.upper() == 'AAPL' and os.path.exists(model_path):
        with st.spinner("Loading pre-trained AAPL model..."):
            model = load_pretrained_model(model_path, hidden_size)
        st.success("✅ Pre-trained AAPL model loaded successfully!")
    else:
        with st.spinner("Training LSTM model..."):
            model = train_lstm_model(X_train_tensor, y_train_tensor, hidden_size, lstm_epochs)
        st.success("✅ LSTM model trained successfully!")
    
    # LSTM Predictions
    model.eval()
    with torch.no_grad():
        train_predict = model(X_train_tensor).numpy()
        test_predict = model(X_test_tensor).numpy()
    
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_orig = scaler.inverse_transform(y_train_tensor.numpy())
    y_test_orig = scaler.inverse_transform(y_test_tensor.numpy())
    
    # Train Prophet Model
    st.subheader("📊 Training Prophet Model")
    with st.spinner("Training Prophet..."):
        prophet_data = data[['Date', 'Close']].copy()
        prophet_data.columns = ['ds', 'y']
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        prophet_data['y'] = prophet_data['y'].astype(float)
        
        prophet_train = prophet_data.iloc[:training_size].reset_index(drop=True)
        prophet_test = prophet_data.iloc[training_size:].reset_index(drop=True)
        
        prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=False)
        prophet_model.fit(prophet_train)
        
        future = prophet_model.make_future_dataframe(periods=len(prophet_test))
        forecast = prophet_model.predict(future)
        prophet_predictions = forecast['yhat'].iloc[-len(prophet_test):].values
    
    st.success("✅ Prophet model trained successfully")
    
    # Performance Metrics
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    lstm_train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_predict))
    lstm_test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_predict))
    prophet_test_rmse = np.sqrt(mean_squared_error(prophet_test['y'].values, prophet_predictions))
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("LSTM Train RMSE", f"${lstm_train_rmse:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("LSTM Test RMSE", f"${lstm_test_rmse:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Prophet Test RMSE", f"${prophet_test_rmse:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Determine best model
    best_model = "LSTM" if lstm_test_rmse < prophet_test_rmse else "Prophet"
    improvement = abs(lstm_test_rmse - prophet_test_rmse)
    st.info(f"🏆 **Best Model**: {best_model} (RMSE difference: ${improvement:.2f})")
    
    # Visualization
    st.subheader("📈 Stock Price Prediction Visualization")
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual prices
    ax.plot(data['Date'], data['Close'], label='Actual Price', color='#1f77b4', linewidth=2.5, alpha=0.8)
    
    # LSTM predictions - align with test data dates
    test_start_idx = training_size + time_step + 1
    lstm_test_dates = data['Date'].iloc[test_start_idx:test_start_idx + len(test_predict)]
    
    if len(lstm_test_dates) == len(test_predict):
        ax.plot(lstm_test_dates, test_predict.flatten(), label='LSTM Prediction', 
                color='#ff7f0e', linewidth=2.5, linestyle='--', alpha=0.8)
    
    # Plot Prophet predictions
    if len(prophet_test['ds']) == len(prophet_predictions):
        ax.plot(prophet_test['ds'], prophet_predictions, label='Prophet Prediction', 
                color='#2ca02c', linewidth=2.5, linestyle=':', alpha=0.8)
    
    ax.set_title(f'Stock Price Prediction for {ticker}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Close Price (USD)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis
    plt.xticks(rotation=45, ha='right')
    
    # Improve layout
    plt.tight_layout()
    
    # Display in streamlit
    st.pyplot(fig)
    plt.close()
    
    # Additional insights
    st.subheader("💡 Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Summary**")
        st.write(f"- Total Days: {len(data)}")
        st.write(f"- Training Days: {training_size}")
        st.write(f"- Testing Days: {test_size}")
        st.write(f"- Date Range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.write("**Price Statistics**")
        current_price = float(data['Close'].iloc[-1])
        highest_price = float(data['Close'].max())
        lowest_price = float(data['Close'].min())
        average_price = float(data['Close'].mean())
        st.write(f"- Current Price: ${current_price:.2f}")
        st.write(f"- Highest: ${highest_price:.2f}")
        st.write(f"- Lowest: ${lowest_price:.2f}")
        st.write(f"- Average: ${average_price:.2f}")

else:
    st.info("👈 Configure the parameters in the sidebar and click 'Run Prediction' to start the analysis.")
    
    # Display sample tickers
    st.subheader("Popular Stock Tickers")
    st.write("**Tech**: AAPL (Apple), GOOGL (Google), MSFT (Microsoft), TSLA (Tesla), NVDA (NVIDIA)")
    st.write("**Finance**: JPM (JP Morgan), BAC (Bank of America), GS (Goldman Sachs)")
    st.write("**Other**: AMZN (Amazon), META (Meta), NFLX (Netflix)")