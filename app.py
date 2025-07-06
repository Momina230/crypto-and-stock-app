from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify  # ✅ ADDED jsonify
from backend import (
    create_user, verify_user, get_user_by_email,add_to_watchlist,save_investment
    
)
from backend import get_watchlist_with_prices,get_watchlist_items
from binancedata import generate_prediction_plot
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Login Page (GET only) ---
@app.route('/', methods=['GET'])
def login():
    return render_template('login.html')



# --- Login Form Handler (POST) ---
@app.route('/login', methods=['POST'])

def handle_login():
    email = request.form['email']
    password = request.form['password']
    
    user = verify_user(email, password)
    if user:
        session['user_email'] = user['email']
        return jsonify({"success": True, "email": user['email']})
    else:
        return jsonify({"success": False, "message": "Invalid credentials!"})
# --- Home Page (after login) ---
@app.route('/home', methods=['GET'])
def home():
    if 'user_email' not in session:
        flash("You must be logged in to view this page.", "warning")
        return redirect(url_for('login'))

    return render_template("home.html", email=session['user_email'])

# --- Signup Page + Handler ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for('signup'))

        if get_user_by_email(email):
            flash("Email already registered.", "warning")
            return redirect(url_for('signup'))

        create_user(username, email, password)
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

# --- View Prediction Chart ---
@app.route('/view')
def view():
    if 'user_email' not in session:
        flash("Please login to access predictions.", "warning")
        return redirect(url_for('login'))

    chart = generate_prediction_plot()
    return render_template('view_stocks.html', chart=chart)

# --- Market Investment Page ---


@app.route('/invest', methods=['GET', 'POST'])
def invest():
    if 'user_email' not in session:
        flash("Please login to access market info.", "warning")
        return redirect(url_for('login'))

    if request.method == 'GET':
        return render_template('invest_market.html')

    data = request.get_json()
    ticker = data.get('ticker', '').upper()
    amount = float(data.get('amount', 0))
    currency = data.get('currency', 'USD').upper()

    user = get_user_by_email(session['user_email'])
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    add_to_watchlist(session['user_email'], ticker)
    save_investment(user['_id'], ticker, amount, currency)

    return jsonify({"success": True, "message": f"Investment in {ticker} recorded."})
@app.route('/watchlist/add', methods=['POST'])
def add_watchlist():
    # Get data from JSON payload instead of session
    data = request.get_json()
    ticker = data.get("ticker", "").upper()
    email = data.get("email")  # Get email from request
    
    if not ticker:
        return jsonify({"success": False, "message": "No ticker provided"}), 400
    
    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400

    # Verify user exists
    user = get_user_by_email(email)
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    add_to_watchlist(email, ticker)
    return jsonify({"success": True, "message": f"{ticker} added to watchlist."})

# @app.route('/watchlist/add', methods=['POST'])
# def add_watchlist():
#     if 'user_email' not in session:
#         return jsonify({"success": False, "message": "Unauthorized"}), 403

#     data = request.get_json()
#     ticker = data.get("ticker", "").upper()

#     if not ticker:
#         return jsonify({"success": False, "message": "No ticker provided"}), 400

#     add_to_watchlist(session['user_email'], ticker)

#     return jsonify({"success": True, "message": f"{ticker} added to watchlist."})
@app.route('/watchlist')
def get_watchlist():
    email = request.args.get('email')
    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400

    watchlist_items = get_watchlist_items(email)
    return jsonify({"success": True, "watchlist": watchlist_items})





if __name__ == '__main__':
    app.run(debug=True)
