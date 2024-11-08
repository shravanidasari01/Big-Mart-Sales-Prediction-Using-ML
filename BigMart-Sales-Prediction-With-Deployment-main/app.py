from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import joblib
import os
import numpy as np
import hashlib

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Dummy user storage (for demonstration purposes, you can replace with a database)
users = {}

# Load model and scaler
scaler_path = r'C:\Users\leneo\Downloads\BigMart-Sales-Prediction-With-Deployment-main\BigMart-Sales-Prediction-With-Deployment-main\models\sc.sav'
model_path = r'C:\Users\leneo\Downloads\BigMart-Sales-Prediction-With-Deployment-main\BigMart-Sales-Prediction-With-Deployment-main\models\lr.sav'

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)


@app.route("/")
def index():
    if 'username' in session:
        return render_template("home.html")
    return redirect(url_for('login'))


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Hash the password for comparison (simple demonstration, not secure)
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        if users.get(username) == password_hash:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash("Invalid credentials! Please try again.")

    return render_template("login.html")


@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Hash the password for storage (simple demonstration, not secure)
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        if username not in users:
            users[username] = password_hash
            flash("Registration successful! You can log in now.")
            return redirect(url_for('login'))
        else:
            flash("Username already exists. Please choose another.")

    return render_template("register.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        # Retrieve and process input data
        item_weight = float(request.form['item_weight'])
        item_fat_content = float(request.form['item_fat_content'])
        item_visibility = float(request.form['item_visibility'])
        item_type = float(request.form['item_type'])
        item_mrp = float(request.form['item_mrp'])
        outlet_establishment_year = float(request.form['outlet_establishment_year'])
        outlet_size = float(request.form['outlet_size'])
        outlet_location_type = float(request.form['outlet_location_type'])
        outlet_type = float(request.form['outlet_type'])

        # Prepare input for model
        X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                       outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

        # Scale the input data
        X_std = scaler.transform(X)

        # Predict the output
        Y_pred = model.predict(X_std)

        # Return prediction with styled HTML for color output
        return render_template("result.html", prediction=float(Y_pred))

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)
