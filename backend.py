import matplotlib.pyplot as plt
import bcrypt
import pymongo
from urllib.parse import quote_plus
from datetime import datetime, timedelta
from binance.client import Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# --- MongoDB Connection Setup ---
username = quote_plus("db_project")
password = quote_plus("12345678910")
cluster = "cluster0.rblnxeq.mongodb.net"

uri = f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority"
client = pymongo.MongoClient(uri)
db = client["cryptoProject"]

# --- Collections ---
users_col = db["users"]
stocks_col = db["stocks"]
stock_prices_col = db["stock_prices"]
predictions_col = db["predictions"]
watchlists_col = db["watchlists"]
investments_col = db["investments"]



def get_watchlist_items(email):
    items = watchlists_col.find({"email": email})
    watchlist_data = []

    for item in items:
        ticker = item["ticker"]
        # Get the latest price from stock_prices collection
        price_doc = stock_prices_col.find_one(
            {"ticker": ticker}, sort=[("timestamp", -1)]
        )
        latest_price = price_doc["price"] if price_doc else 0.0
        currency = item.get("currency", "USDT")

        watchlist_data.append({
            "ticker": ticker,
            "currency": currency,
            "latest_price": latest_price
        })

    return watchlist_data

def save_investment(user_id, ticker, amount, currency):
    investment = {
        "user_id": user_id,
        "ticker": ticker,
        "amount": amount,
        "currency": currency,
        "invested_at": datetime.utcnow()
    }
    investments_col.insert_one(investment)


# --- Ensure unique index on user_id + ticker in watchlists ---
watchlists_col.create_index(
    [("user_id", pymongo.ASCENDING), ("ticker", pymongo.ASCENDING)],
    unique=True
)

# --- Binance API Client ---
# binance_client = Client()
binance_client = Client(api_key="YOUR_API_KEY", api_secret="YOUR_SECRET_KEY")


# --- CRUD Operations ---

def create_user(username, email):
    user = {
        "username": username,
        "email": email,
        "created_at": datetime.utcnow()
    }
    return users_col.insert_one(user)

def get_user_by_email(email):
    return users_col.find_one({"email": email})

def get_all_users():
    return list(users_col.find({}))

# 
def fetch_and_store_crypto_price(symbol):
    try:
        ticker_data = binance_client.get_symbol_ticker(symbol=symbol.upper())
        price = float(ticker_data['price'])

        data = {
            "ticker": symbol.upper(),
            "price": price,
            "currency": symbol[-4:],  # Assumes ending like USDT, BUSD, etc.
            "date": datetime.utcnow()
        }
        return stock_prices_col.insert_one(data)
    except Exception as e:
        print("Error fetching Binance data:", e)
        return None

def update_all_watchlist_prices():
    """Fetch current prices for all unique tickers in watchlists"""
    unique_tickers = watchlists_col.distinct("ticker")
    for ticker in unique_tickers:
        fetch_and_store_crypto_price(ticker)
    print(f"Updated prices for {len(unique_tickers)} tickers")
    


def add_to_watchlist(user_email, ticker):
    
    user = get_user_by_email(user_email)
    if not user:
        print(f"User not found: {user_email}")
        return None

    entry = {
        "user_id": user["_id"],
        "email": user_email,  # Explicitly store email
        "ticker": ticker.upper(),
        "currency": ticker.upper()[-4:],
        "added_at": datetime.utcnow()
    }
    
    try:
        existing = watchlists_col.find_one({
            "email": user_email,  # Changed from user_id to email
            "ticker": ticker.upper()
        })
        if existing:
            print(f"Ticker {ticker.upper()} already in watchlist")
            return None
     

        return watchlists_col.insert_one(entry)
    except pymongo.errors.DuplicateKeyError:
        print(f"Duplicate key error")
        return None
   
    # user = get_user_by_email(user_email)
    # if not user:
    #     print(f"User not found: {user_email}")
    #     return None

    # entry = {
    #     "user_id": user["_id"],
    #     "ticker": ticker.upper(),
    #     "currency": ticker.upper()[-4:],
    #     "added_at": datetime.utcnow()
    # }

    # try:
    #     existing = watchlists_col.find_one({
    #         "user_id": user["_id"],
    #         "ticker": ticker.upper()
    #     })
    #     if existing:
    #         print(f"Ticker {ticker.upper()} already in watchlist for user {user_email}")
    #         return None

    #     inserted = watchlists_col.insert_one(entry)
    #     print(f"Added ticker {ticker.upper()} to watchlist for user {user_email}")
    #     return inserted
    # except pymongo.errors.DuplicateKeyError:
    #     print(f"Duplicate key error: {ticker.upper()} already in watchlist for {user_email}")
    #     return None


def update_user_email(old_email, new_email):
    return users_col.update_one({"email": old_email}, {"$set": {"email": new_email}})

def remove_from_watchlist(user_email, ticker):
    user = get_user_by_email(user_email)
    if user:
        return watchlists_col.delete_one({"user_id": user["_id"], "ticker": ticker.upper()})
    return None

# def get_watchlist_with_prices(email):
#     user = get_user_by_email(email)
#     if not user:
#         return []

#     watchlist = watchlists_col.find({"user_id": user["_id"]})
#     result = []

#     for entry in watchlist:
#         ticker = entry["ticker"]
#         latest_price = stock_prices_col.find_one(
#             {"ticker": ticker},
#             sort=[("date", pymongo.DESCENDING)]
#         )

#         result.append({
#             "ticker": ticker,
#             "added_at": entry["added_at"],
#             "latest_price": latest_price["price"] if latest_price else "No price data"
#         })

#     return result
def get_watchlist_with_prices(email):
    user = get_user_by_email(email)
    if not user:
        return []

    watchlist = watchlists_col.find({"user_id": user["_id"]})
    result = []

    for entry in watchlist:
        ticker = entry["ticker"]
        latest_price = stock_prices_col.find_one(
            {"ticker": ticker},
            sort=[("date", pymongo.DESCENDING)]
        )

        result.append({
            "ticker": ticker,
            "currency": ticker[-4:],  # USDT, BUSD, etc.
            "added_at": entry["added_at"],
            "latest_price": latest_price["price"] if latest_price else "No price data"
        })

    return result


# def clean_watchlist_duplicates(user_email):
#     user = get_user_by_email(user_email)
#     if not user:
#         print(f"User not found: {user_email}")
#         return

#     cursor = watchlists_col.find({"user_id": user["_id"]}).sort("added_at", pymongo.ASCENDING)
#     seen = set()
#     duplicates = []

#     for doc in cursor:
#         ticker = doc["ticker"]
#         if ticker in seen:
#             duplicates.append(doc["_id"])
#         else:
#             seen.add(ticker)

#     for dup_id in duplicates:
#         watchlists_col.delete_one({"_id": dup_id})

#     print(f"Removed {len(duplicates)} duplicate entries from watchlist for user {user_email}")

# --- ML Functions ---
def prepare_ml_data(ticker, days=30):
    """Fetch historical data for ML training"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get historical data from Binance
    klines = binance_client.get_historical_klines(
        symbol=ticker,
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str=start_date.strftime("%d %b %Y %H:%M:%S"),
        end_str=end_date.strftime("%d %b %Y %H:%M:%S")
    )
def fetch_latest_price(ticker):
    latest_price = stock_prices_col.find_one(
        {"ticker": ticker},
        sort=[("date", pymongo.DESCENDING)]
    )
    return float(latest_price['price']) if latest_price else None

    
    # Convert to DataFrame
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Convert to numeric and extract relevant columns
    data['close'] = pd.to_numeric(data['close'])
    data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
    
    return data[['date', 'close']]

def train_price_prediction_model(ticker):
    """Train a simple linear regression model for price prediction"""
    try:
        # Get data
        data = prepare_ml_data(ticker)
        
        if len(data) < 10:
            print(f"Not enough data for {ticker} to train model")
            return None
        
        # Prepare features (using day number as feature)
        data['day_num'] = np.arange(len(data))
        X = data[['day_num']].values
        y = data['close'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model trained for {ticker} with MSE: {mse:.2f}")
        
        return model
    except Exception as e:
        print(f"Error training model for {ticker}: {str(e)}")
        return None

def predict_next_price(ticker, model):
    """Predict next day's price using trained model"""
    try:
        # Get latest day number
        data = prepare_ml_data(ticker)
        last_day = len(data) - 1
        
        # Predict next day
        next_day = np.array([[last_day + 1]])
        predicted_price = model.predict(next_day)[0]
        
        print(f"Predicted next price for {ticker}: ${predicted_price:.2f}")
        return predicted_price
    except Exception as e:
        print(f"Error predicting price for {ticker}: {str(e)}")
        return None
# import bcrypt

def create_user(username, email, password):
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user = {
        "username": username,
        "email": email,
        "password": hashed_pw,
        "created_at": datetime.utcnow()
    }
    return users_col.insert_one(user)

def verify_user(email, password):
    user = users_col.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        return user
    return None


# --- Main Execution ---
if __name__ == "__main__":
    # Create 10 dummy users
    for i in range(1, 11):
        email = f"user{i}@example.com"
        if not get_user_by_email(email):
            username = f"user{i}"
            create_user(username, email, "testpassword")

    # Fetch and store BTC/ETH prices
    # fetch_and_store_crypto_price("BTCUSDT")
    # fetch_and_store_crypto_price("ETHUSDT")
    initial_tickers = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
    for ticker in initial_tickers:
        fetch_and_store_crypto_price(ticker)
    
    # Update all watchlist prices periodically
    update_all_watchlist_prices()

    # Add to watchlists
    for i in range(1, 11):
        email = f"user{i}@example.com"
        if i % 2 == 1:
            add_to_watchlist(email, "BTCUSDT")
        else:
            add_to_watchlist(email, "ETHUSDT")

    # Save a sample investment for each user
    for i in range(1, 11):
        email = f"user{i}@example.com"
        user = get_user_by_email(email)
        save_investment(user["_id"], "BTCUSDT" if i % 2 == 1 else "ETHUSDT", amount=100 + i, currency="USDT")

    print("✅ Database reset complete.")
