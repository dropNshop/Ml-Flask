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
    """Get comprehensive dashboard data including forecasts and statistics"""
    try:
        df = load_and_preprocess_data()
        current_date = datetime.now()
        
        # ============ Basic Stats ============
        total_sales = float(df['Total Sales (PKR)'].sum())
        total_products = len(df['Product'].unique())
        avg_order = round(float(df['Price (PKR)'].mean()), 2)
        
        # Calculate monthly growth (Fixed Period handling)
        df['YearMonth'] = df['Order Date'].dt.to_period('M')
        monthly_sales = df.groupby('YearMonth')['Total Sales (PKR)'].sum()
        monthly_growth = monthly_sales.pct_change() * 100
        avg_monthly_growth = round(float(monthly_growth.mean()), 1)

        # ============ Category Distribution ============
        category_sales = df.groupby('Category')['Total Sales (PKR)'].sum().round(2)
        category_distribution = [
            {"name": cat, "value": float(sales)} 
            for cat, sales in category_sales.items()
        ]

        # ============ Monthly Sales Trend (Fixed Period handling) ============
        monthly_trend = (df.groupby([df['Order Date'].dt.year, df['Order Date'].dt.month])
                        ['Total Sales (PKR)']
                        .sum()
                        .reset_index())
        monthly_trend.columns = ['Year', 'Month', 'Total Sales (PKR)']
        
        monthly_trend_data = [
            {
                "date": str(row['Month']),  # Convert month to string
                "sales": float(row['Total Sales (PKR)'])
            }
            for _, row in monthly_trend.iterrows()
        ]

        # ============ Top Products Performance ============
        top_products = df.groupby('Product').agg({
            'Quantity Sold': 'sum',
            'Price (PKR)': 'mean',
            'Total Sales (PKR)': 'sum'
        }).reset_index()
        
        top_products = top_products.nlargest(13, 'Total Sales (PKR)')
        top_products_data = [
            {
                "product": row['Product'],
                "quantity": int(row['Quantity Sold']),
                "price": float(row['Price (PKR)']),
                "sales": float(row['Total Sales (PKR)'])
            }
            for _, row in top_products.iterrows()
        ]

        # ============ Forecast Data ============
        # Define product categories and their brands
        product_categories = {
            'Dairy': ['Milk', 'Yogurt', 'Cheese', 'Butter', 'Cream'],
            'Fruits': ['Apples', 'Bananas', 'Oranges', 'Mangoes', 'Watermelon'],
            'Groceries': ['Rice (Basmati)', 'Cooking Oil', 'Tea', 'Sugar', 'Flour (Atta)', 'Pulses (Daal)'],
            'Pharmacy': ['Pain Relievers', 'Cold Medicine', 'Vitamins', 'First Aid', 'Sanitizers'],
            'Vegetables': ['Tomatoes', 'Potatoes', 'Onions', 'Green Chilies', 'Carrots']
        }

        brands_mapping = {
            'Rice (Basmati)': ['Falak', 'Guard', 'Kernel'],
            'Cooking Oil': ['Dalda', 'Sufi', 'Eva', 'Habib'],
            'Tea': ['Lipton', 'Tapal', 'Vital', 'Supreme'],
            'Sugar': ['Al-Arabia', 'Nishat'],
            'Flour (Atta)': ['Sunridge', 'Bake Parlor', 'Fauji'],
            'Pulses (Daal)': ['Mitchell\'s', 'National']
        }

        # Generate next 6 months
        months = []
        for i in range(6):
            future_date = current_date + timedelta(days=31*i)
            months.append(future_date.strftime('%B'))

        # Generate forecast data for each product
        forecast_data = []
        for month in months:
            month_data = {"month": month}
            
            for category, products in product_categories.items():
                for product in products:
                    # Generate realistic forecast based on historical data
                    if product in df['Product'].unique():
                        historical_qty = df[df['Product'] == product]['Quantity Sold'].mean()
                        std_dev = df[df['Product'] == product]['Quantity Sold'].std()
                        
                        # Add seasonal and random variation
                        seasonal_factor = 1 + np.random.uniform(-0.2, 0.2)
                        forecast_qty = int(historical_qty * seasonal_factor + np.random.normal(0, std_dev * 0.1))
                        month_data[product] = max(0, forecast_qty)
                    else:
                        # Default forecast for products not in historical data
                        month_data[product] = int(np.random.uniform(100, 500))
            
            forecast_data.append(month_data)

        response = {
            "status": "success",
            "data": {
                "stats": {
                    "total_sales": total_sales,
                    "total_products": total_products,
                    "avg_order": avg_order,
                    "monthly_growth": avg_monthly_growth
                },
                "category_distribution": category_distribution,
                "monthly_trend": monthly_trend_data,
                "top_products": top_products_data,
                "forecast": {
                    "months": months,
                    "categories": product_categories,
                    "brands_mapping": brands_mapping,
                    "data": forecast_data
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 