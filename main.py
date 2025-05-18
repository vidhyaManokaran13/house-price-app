import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import io
import base64
import os
import zipfile

from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key

# 🔓 Unzip the model if not already unzipped
ZIP_FILE = 'house_price_model.zip'
PKL_FILE = 'house_price_model.pkl'

if not os.path.exists(PKL_FILE):
    print("Unzipping model...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall()
    print("Model extracted.")

# ✅ Load model after unzipping
model = joblib.load(PKL_FILE)

# Load dataset
df = pd.read_csv('Chennai houseing sale.csv')

# Dropdown options from dataset
area_options = sorted(df['AREA'].dropna().unique())
street_options = sorted(df['STREET'].dropna().unique())
park_options = sorted(df['PARK_FACIL'].dropna().unique())
dist_mainroad_options = sorted(df['DIST_MAINROAD'].dropna().unique())
buildtype_options = sorted(df['BUILDTYPE'].dropna().unique())

# Dummy user database
users = {}

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users and users[uname] == pwd:
            session['username'] = uname
            return redirect(url_for('form'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        confirm_pwd = request.form['confirm_password']
        if uname in users:
            return render_template('register.html', error='Username already exists')
        if pwd != confirm_pwd:
            return render_template('register.html', error='Passwords do not match')
        users[uname] = pwd
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if 'username' not in session:
        return redirect(url_for('login'))

    predicted_price = None
    pred_year = None
    graph_image = None

    if request.method == 'POST':
        try:
            area = request.form['area']
            street = request.form['street']
            park = request.form['park_facil']
            dist_mainroad = request.form['dist_mainroad']
            buildtype = request.form['buildtype']
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            rooms = int(request.form['rooms'])
            current_year = int(request.form['current_year'])
            pred_year = int(request.form['pred_year'])
        except Exception as e:
            return render_template('form.html', areas=area_options, streets=street_options,
                                   parks=park_options, dist_mainroads=dist_mainroad_options,
                                   buildtypes=buildtype_options,
                                   error="Invalid input: " + str(e))

        build_age = pred_year - current_year

        input_data = {
            'AREA': area,
            'STREET': street,
            'PARK_FACIL': park,
            'DIST_MAINROAD': dist_mainroad,
            'BUILDTYPE': buildtype,
            'QS_BEDROOM': df['QS_BEDROOM'].median(),
            'QS_BATHROOM': df['QS_BATHROOM'].median(),
            'QS_ROOMS': df['QS_ROOMS'].median(),
            'QS_OVERALL': df['QS_OVERALL'].median(),
            'REG_FEE': df['REG_FEE'].median(),
            'UTILITY_AVAIL': 'Yes',
            'COMMIS': 0,
            'SALE_COND': 'Normal',
            'MZZONE': 'Residential',
            'INT_SQFT': df['INT_SQFT'].median(),
            'N_ROOM': rooms,
            'N_BEDROOM': bedrooms,
            'N_BATHROOM': bathrooms,
            'BUILD_AGE': build_age
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        predicted_price = round(prediction)

        # Trend for 10 years
        years = np.arange(current_year, current_year + 11)
        prices = [round(model.predict(pd.DataFrame([{
            **input_data,
            'BUILD_AGE': year - current_year
        }]))[0]) for year in years]

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(years, prices, 'b-o', linewidth=2, markersize=6)
        plt.title(f"Predicted Price Trend: {area}")
        plt.xlabel("Year")
        plt.ylabel("Price (₹)")
        plt.grid(True)

        if pred_year in years:
            idx = list(years).index(pred_year)
            plt.scatter(pred_year, prices[idx], color='red', s=80)
            plt.annotate(f'₹{prices[idx]:,}', (pred_year, prices[idx]),
                         textcoords="offset points", xytext=(0,10), ha='center')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_image = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    return render_template('form.html',
                           areas=area_options, streets=street_options,
                           parks=park_options, dist_mainroads=dist_mainroad_options,
                           buildtypes=buildtype_options,
                           predicted_price=predicted_price,
                           pred_year=pred_year,
                           graph_image=graph_image)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
