from binance.client import Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
from io import BytesIO
import base64

# Binance client
client = Client(api_key='', api_secret='')

def get_historical_klines(symbol='BTCUSDT', interval='1d', lookback='180 days ago UTC'):
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def create_features(df, window_size=3):
    X, y = [], []
    for i in range(window_size, len(df)):
        features = df['close'].iloc[i - window_size:i].values
        target = df['close'].iloc[i]
        X.append(features)
        y.append(target)
    return pd.DataFrame(X), pd.Series(y)

def train_and_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test.values, y_pred

# --- Dark Theme Matplotlib Settings ---
plt.style.use('dark_background')
DARK_FIG_BG = '#121212'
DARK_AX_BG = '#1e1e1e'
TEXT_COLOR = 'white'
GRID_COLOR = '#444'

def generate_prediction_plot():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor(DARK_AX_BG)
    fig.patch.set_facecolor(DARK_FIG_BG)

    symbols = {
        'ETHUSDT': ('cyan', 'deepskyblue'),
        'BNBUSDT': ('lime', 'lightgreen'),
        'USDCUSDT': ('violet', 'orchid'),
        'XRPUSDT': ('orange', 'gold')
    }

    for symbol, (actual_color, pred_color) in symbols.items():
        try:
            df = get_historical_klines(symbol)
            X, y = create_features(df)
            y_test, y_pred = train_and_predict(X, y)
            ax.plot(y_test, label=f"{symbol} Actual", color=actual_color)
            ax.plot(y_pred, label=f"{symbol} Predicted", color=pred_color, linestyle='--')
        except Exception as e:
            print(f"Error with {symbol}: {e}")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"Actual vs Predicted Prices (Updated: {now})", color=TEXT_COLOR)
    ax.set_xlabel("Test Data Points", color=TEXT_COLOR)
    ax.set_ylabel("Price (USDT)", color=TEXT_COLOR)
    ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR)
    ax.legend(facecolor=DARK_AX_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5)

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)

    return image_base64
