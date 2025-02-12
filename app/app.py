from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import os
import calendar
from scipy import stats

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

@app.route('/api/forecast', methods=['GET'])
def get_forecast_data():
    """Get 6-month demand forecasting data using historical patterns"""
    try:
        df = load_and_preprocess_data()
        
        # Get current date and next 6 months
        current_date = datetime.now()
        months = []
        for i in range(6):
            future_date = current_date + timedelta(days=31*i)
            months.append(future_date.strftime('%B'))

        # ============ Dynamic Product Selection ============
        # Select top products based on sales volume and consistency
        product_stats = df.groupby('Product').agg({
            'Quantity Sold': ['sum', 'mean', 'std'],
            'Total Sales (PKR)': 'sum'
        }).reset_index()
        
        # Calculate coefficient of variation for stability
        product_stats['cv'] = product_stats[('Quantity Sold', 'std')] / product_stats[('Quantity Sold', 'mean')]
        
        # Select products with high sales and stable demand
        top_products = product_stats[
            (product_stats[('Quantity Sold', 'sum')] > product_stats[('Quantity Sold', 'sum')].median()) &
            (product_stats['cv'] < product_stats['cv'].median())
        ]['Product'].head(6).tolist()

        # ============ Brand Analysis ============
        # Dynamically get brand associations
        brands_mapping = {}
        for product in top_products:
            brands = df[df['Product'] == product]['Brand'].unique().tolist()
            brands_mapping[product] = brands

        # ============ Forecast Generation ============
        monthly_predictions = {}
        forecast_models = {}

        for product in top_products:
            # Create time series for each product
            product_data = df[df['Product'] == product].copy()
            product_data['YearMonth'] = product_data['Year'].astype(str) + '-' + product_data['Month'].astype(str).str.zfill(2)
            monthly_qty = product_data.groupby('YearMonth')['Quantity Sold'].sum().reset_index()
            
            # Prepare features for forecasting
            X = np.arange(len(monthly_qty)).reshape(-1, 1)
            y = monthly_qty['Quantity Sold'].values
            
            # Fit linear trend with seasonal adjustment
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate seasonal factors
            seasonal_factors = []
            for month in range(1, 13):
                month_data = product_data[product_data['Month'] == month]['Quantity Sold']
                if not month_data.empty:
                    factor = month_data.mean() / product_data['Quantity Sold'].mean()
                    seasonal_factors.append(factor)
                else:
                    seasonal_factors.append(1.0)
            
            forecast_models[product] = {
                'trend_model': model,
                'seasonal_factors': seasonal_factors,
                'base_qty': y[-1]  # Last known quantity
            }

        # Generate predictions
        for month in months:
            month_num = list(calendar.month_name).index(month)
            monthly_predictions[month] = {}
            
            for product in top_products:
                model_info = forecast_models[product]
                
                # Combine trend and seasonality
                trend = model_info['trend_model'].predict([[len(monthly_qty) + month_num]])[0]
                seasonal_factor = model_info['seasonal_factors'][month_num - 1]
                base_prediction = trend * seasonal_factor
                
                # Add confidence-based variation
                std_dev = df[df['Product'] == product]['Quantity Sold'].std()
                variation = np.random.normal(0, std_dev * 0.1)  # 10% of standard deviation
                
                predicted_qty = max(int(base_prediction + variation), 0)  # Ensure non-negative
                monthly_predictions[month][product] = predicted_qty

        # Format response
        response = {
            'status': 'success',
            'data': {
                'forecast_chart': {
                    'months': months,
                    'datasets': [
                        {
                            'product': product,
                            'values': [monthly_predictions[month][product] for month in months]
                        } for product in top_products
                    ]
                },
                'forecast_table': {
                    'headers': ['PRODUCT', 'BRANDS'] + [month.upper() for month in months],
                    'rows': [
                        {
                            'product': product,
                            'brands': ', '.join(brands_mapping[product]),
                            'predictions': [
                                f"{monthly_predictions[month][product]} {'liters' if 'Oil' in product else 'kg'}"
                                for month in months
                            ]
                        } for product in top_products
                    ]
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 