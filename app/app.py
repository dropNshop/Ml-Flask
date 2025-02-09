from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import os

app = Flask(__name__)

# Configure CORS
cors = CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize global variables for model persistence
trained_model = None
feature_scaler = None
category_encoder = None
product_encoder = None
_cached_df = None

# ============ Data Loading and Preprocessing ============
def load_and_preprocess_data():
    """Load and preprocess the sales data with caching"""
    global _cached_df, category_encoder, product_encoder
    
    if _cached_df is not None:
        return _cached_df
        
    try:
        # Load data
        df = pd.read_csv('dropandshop.csv')
        
        # Convert date and extract features
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Month'] = df['Order Date'].dt.month
        df['Year'] = df['Order Date'].dt.year
        
        # Encode categorical variables
        category_encoder = LabelEncoder()
        product_encoder = LabelEncoder()
        
        df['Category_encoded'] = category_encoder.fit_transform(df['Category'])
        df['Product_encoded'] = product_encoder.fit_transform(df['Product'])
        
        # Cache the preprocessed dataframe
        _cached_df = df
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

# ============ Model Training ============
def train_price_prediction_model(df):
    """Train the price prediction model"""
    try:
        features = ['Category_encoded', 'Product_encoded', 'Month', 'Year']
        X = df[features]
        y = df['Price (PKR)']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model with smaller test size
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.1, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate basic metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': {
                'rmse': round(np.sqrt(mse), 2),
                'r2': round(r2_score(y_test, y_pred), 4)
            }
        }
    except Exception as e:
        raise Exception(f"Error training model: {str(e)}")

# ============ API Endpoints ============
@app.route('/api/train', methods=['GET'])
def train_model():
    """Train the model and return metrics"""
    try:
        df = load_and_preprocess_data()
        model_info = train_price_prediction_model(df)
        
        global trained_model, feature_scaler
        trained_model = model_info['model']
        feature_scaler = model_info['scaler']
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': model_info['metrics']
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
        if trained_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not trained. Please train the model first using /api/train'
            }), 400
            
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
            
        required_fields = ['category', 'product', 'month', 'year']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        try:
            features = np.array([[
                category_encoder.transform([data['category']])[0],
                product_encoder.transform([data['product']])[0],
                int(data['month']),
                int(data['year'])
            ]])
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid input data: {str(e)}'
            }), 400
        
        features_scaled = feature_scaler.transform(features)
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
        
        # Efficient aggregations
        category_sales = df.groupby('Category')['Total Sales (PKR)'].sum()
        monthly_sales = df.groupby(['Year', 'Month'])['Total Sales (PKR)'].sum()
        product_metrics = df.groupby('Product').agg({
            'Quantity Sold': 'sum',
            'Total Sales (PKR)': 'sum'
        })
        
        # Calculate growth
        monthly_totals = monthly_sales.reset_index()
        monthly_totals['Growth'] = monthly_totals['Total Sales (PKR)'].pct_change() * 100
        
        response = {
            'status': 'success',
            'data': {
                'category_sales': category_sales.to_dict(),
                'monthly_sales': {f"{y}-{m}": v for (y, m), v in monthly_sales.items()},
                'top_products': {
                    'quantity': product_metrics['Quantity Sold'].to_dict(),
                    'sales': product_metrics['Total Sales (PKR)'].to_dict()
                },
                'growth_metrics': {
                    'average_monthly_growth': round(monthly_totals['Growth'].mean(), 2),
                    'latest_growth': round(monthly_totals['Growth'].iloc[-1], 2) if len(monthly_totals) > 1 else 0
                }
            }
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get basic statistics"""
    try:
        df = load_and_preprocess_data()
        
        stats = {
            'total_sales': float(df['Total Sales (PKR)'].sum()),
            'total_products': len(df['Product'].unique()),
            'total_categories': len(df['Category'].unique()),
            'avg_price': round(float(df['Price (PKR)'].mean()), 2),
            'total_quantity_sold': int(df['Quantity Sold'].sum())
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 