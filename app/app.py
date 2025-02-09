from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

app = Flask(__name__)

# Configure CORS
cors = CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize global variables
trained_model = None
feature_scaler = None
category_encoder = None
product_encoder = None

# ============ Data Loading and Preprocessing ============
def load_and_preprocess_data():
    """Load and preprocess the sales data"""
    try:
        # Load data
        df = pd.read_csv('dropandshop.csv')
        
        # Convert date and extract features
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Month'] = df['Order Date'].dt.month
        df['Year'] = df['Order Date'].dt.year
        df['Day'] = df['Order Date'].dt.day
        
        # Encode categorical variables
        le_category = LabelEncoder()
        le_product = LabelEncoder()
        
        df['Category_encoded'] = le_category.fit_transform(df['Category'])
        df['Product_encoded'] = le_product.fit_transform(df['Product'])
        
        # Store encoders in global scope for prediction
        global category_encoder, product_encoder
        category_encoder = le_category
        product_encoder = le_product
        
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

# ============ Model Training ============
def train_price_prediction_model(df):
    """Train the price prediction model"""
    try:
        # Prepare features for price prediction
        features = ['Category_encoded', 'Product_encoded', 'Month', 'Year']
        X = df[features]
        y = df['Price (PKR)']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': {
                'mse': mse,
                'r2': r2,
                'rmse': np.sqrt(mse)
            },
            'test_data': {
                'actual': y_test.tolist(),
                'predicted': y_pred.tolist()
            }
        }
    except Exception as e:
        raise Exception(f"Error training model: {str(e)}")

# ============ API Endpoints ============
@app.route('/api/train', methods=['GET'])
def train_model():
    """Train the model and return metrics"""
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Train model
        model_info = train_price_prediction_model(df)
        
        # Store model globally
        global trained_model, feature_scaler
        trained_model = model_info['model']
        feature_scaler = model_info['scaler']
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': model_info['metrics'],
            'sample_predictions': {
                'actual_prices': model_info['test_data']['actual'][:5],
                'predicted_prices': model_info['test_data']['predicted'][:5]
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict_price():
    """Predict price based on input features"""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Check if model is trained
        if trained_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not trained. Please train the model first using /api/train'
            }), 400
            
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
            
        # Validate required fields
        required_fields = ['category', 'product', 'month', 'year']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Encode categorical inputs
        try:
            category_encoded = category_encoder.transform([data['category']])[0]
            product_encoded = product_encoder.transform([data['product']])[0]
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid category or product: {str(e)}'
            }), 400
        
        # Prepare features
        features = np.array([[
            category_encoded,
            product_encoded,
            int(data['month']),
            int(data['year'])
        ]])
        
        # Scale features
        features_scaled = feature_scaler.transform(features)
        
        # Make prediction
        predicted_price = trained_model.predict(features_scaled)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_price': round(predicted_price, 2),
            'currency': 'PKR',
            'input_data': data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard analytics data"""
    try:
        df = load_and_preprocess_data()
        
        # Sales by category
        category_sales = df.groupby('Category')['Total Sales (PKR)'].sum().to_dict()
        
        # Monthly sales trend
        monthly_sales = df.groupby(['Year', 'Month'])['Total Sales (PKR)'].sum().to_dict()
        
        # Top selling products
        top_products = df.groupby('Product')[['Quantity Sold', 'Total Sales (PKR)']].sum().to_dict()
        
        # Calculate growth rates
        monthly_growth = calculate_growth_rates(df)
        
        response = {
            'status': 'success',
            'data': {
                'category_sales': category_sales,
                'monthly_sales': {f"{k[0]}-{k[1]}": v for k, v in monthly_sales.items()},
                'top_products': {
                    'quantity': top_products['Quantity Sold'],
                    'sales': top_products['Total Sales (PKR)']
                },
                'growth_metrics': monthly_growth
            }
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def calculate_growth_rates(df):
    """Calculate monthly growth rates"""
    monthly_sales = df.groupby(['Year', 'Month'])['Total Sales (PKR)'].sum().reset_index()
    monthly_sales['Growth'] = monthly_sales['Total Sales (PKR)'].pct_change() * 100
    
    return {
        'average_monthly_growth': round(monthly_sales['Growth'].mean(), 2),
        'latest_growth': round(monthly_sales['Growth'].iloc[-1], 2) if len(monthly_sales) > 1 else 0
    }

@app.route('/api/stats', methods=['GET'])
def get_stats():
    df = load_and_preprocess_data()
    
    stats = {
        'total_sales': float(df['Total Sales (PKR)'].sum()),
        'total_products': len(df['Product'].unique()),
        'total_categories': len(df['Category'].unique()),
        'avg_price': float(df['Price (PKR)'].mean()),
        'total_quantity_sold': int(df['Quantity Sold'].sum())
    }
    
    return jsonify(stats) 