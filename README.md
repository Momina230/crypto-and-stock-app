# crypto-and-stock-app
A full-stack web application that allows users to track cryptocurrency and stock prices, manage watchlists, record investments, and view predictions using machine learning.

---

##  Features

-  **User Authentication**: Sign up, log in, and manage secure sessions.
-  **Live Price Tracking**: Real-time updates from APIs (e.g., Binance) for both stocks and cryptocurrencies.
-  **Watchlist Management**: Add/remove assets to a personal watchlist with current prices shown.
-  **Investment Recording**: Record the amount and type of investment made, and track its performance.
-  **Price Prediction (ML)**: Linear regression-based predictions on asset trends.
-  **Interactive UI**: Built with HTML, CSS, and JavaScript for a responsive frontend.
-  **Database**: MongoDB for storing users, watchlists, investments, and predictions.

---

##  Technologies Used

| Stack | Tech |
|-------|------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python, Flask |
| **Database** | MongoDB |
| **Machine Learning** | Scikit-learn (Linear Regression) |
| **API** | Binance API / Stock Market APIs |
| **Styling** | Bootstrap (or your own CSS) |

---

## 🗂️ Project Structure

project/
├── static/
│ ├── css/
│ └── js/
├── templates/
│ ├── login.html
│ ├── dashboard.html
│ └── ...
├── app.py
├── database.py
├── ml_model.py
├── requirements.txt
└── README.md



---

##  Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/crypto-stock-tracker.git
cd crypto-stock-tracker
2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
3. Install dependencies

pip install -r requirements.txt
4. Set up MongoDB
Make sure MongoDB is running locally or use MongoDB Atlas.

Update the Mongo URI in database.py or app.py.

5. Run the app

python app.py
Visit http://127.0.0.1:5000 in your browser.

 Machine Learning
Historical price data is fetched from API.

Linear Regression is used to predict the next-day price.

ML logic is inside ml_model.py.

 Todo / Future Enhancements
Add charting with Chart.js or Recharts

Support for multiple currencies (e.g., USDT, BUSD)

Advanced ML models (e.g., LSTM for time series)




 Author
Momina Nasir
GitHub • LinkedIn

---

