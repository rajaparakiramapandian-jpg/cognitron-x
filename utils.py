import streamlit as st
import pandas as pd
import hashlib
import os
import json
from database import db

# --------------------------------------------------
# UI & CSS
# --------------------------------------------------
def apply_custom_css():
    st.markdown("""
<style>
/* GLOBAL: Clean, Professional Typography */
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    color: #000000 !important; /* Changed to Black as requested */
}

/* CRITICAL UI - DO NOT CHANGE */

/* CARDS: General */
.stCard, div[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    border-radius: 16px !important;
    padding: 30px !important;
}

/* BACKGROUND: Clean Professional Look */
.stApp {
    background-color: #f8f9fa !important;
    background-image: none !important;
}

/* BUTTONS: Solid, Professional Blue */
.stButton>button {
    background-color: #2c3e50 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}
/* ENSURE BUTTON TEXT IS VISIBLE */
.stButton>button p, .stButton>button span {
    color: white !important;
}
.stButton>button:hover {
    background-color: #1a252f !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
}

/* INPUTS: Clean Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    color: #000000 !important;
    border-radius: 8px !important;
    padding: 10px 15px !important;
}
::placeholder {
    color: rgba(0, 0, 0, 0.6) !important;
}

/* AI CHAT: Clean Bubbles */
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    border-radius: 8px !important;
}
[data-testid="stChatMessage"] p {
    color: #374151 !important;
}

/* METRICS: Clean Cards */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-left: 4px solid #2c3e50 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    padding: 15px !important;
    border-radius: 8px !important;
}

/* HEADERS: Dark Text, No Shadow */
h1, h2, h3, p, label, .stMarkdown, .stCaption {
    color: #000000 !important;
    text-shadow: none !important; /* Removed Shadow */
    font-weight: 600 !important;
}

/* REMOVE UNWANTED GLITCHES */
.stDeployButton {display:none;}
footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent !important;}

</style>
""", unsafe_allow_html=True)

def set_background():
    if st.session_state.page in ["login", "signup"]:
        st.markdown("""
        <style>
        .stApp {
            /* Cinematic Vignette & Depth of Field */
            background-image: 
                radial-gradient(circle at center, transparent 0%, rgba(0,0,0,0.4) 60%, rgba(0,0,0,0.8) 100%),
                linear-gradient(to bottom, rgba(10, 25, 50, 0.5), rgba(10, 25, 50, 0.8)),
                url('https://images.unsplash.com/photo-1441986300917-64674bd600d8?q=80&w=2070&auto=format&fit=crop') !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #f8f9fa !important;
            background-image: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# DATA CLEANING UTILITIES
# --------------------------------------------------
def clean_data(df):
    """
    Ensures the data is ready for analysis and ML.
    """
    # 1. Strip whitespace from columns
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. Try to fix Numeric Columns (strip $ and ,)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Remove currency symbols and commas, then convert
                cleaned = df[col].str.replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(cleaned)
            except:
                pass
                
    # 3. Handle Missing Values
    # Fill numeric with mean, categorical with mode
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            
    return df

def detect_sales_column(df):
    for col in df.columns:
        if col.lower() in ["sales", "revenue", "amount", "total_sales", "profit"]:
            return col
    return None

def detect_category_column(df):
    for col in df.columns:
        if col.lower() in ["category", "product", "item", "segment", "type"]:
            return col
    return None

@st.cache_data(show_spinner=False)
def calculate_kpis(df):
    sales_col = detect_sales_column(df)
    kpis = {
        "total_records": len(df),
        "total_revenue": df[sales_col].sum() if sales_col else 0,
        "avg_sale": df[sales_col].mean() if sales_col else 0,
        "max_sale": df[sales_col].max() if sales_col else 0,
        "sales_col_name": sales_col
    }
    return kpis

@st.cache_data(show_spinner=False)
def get_aggregated_sales(df, cat_col, sales_col):
    return df.groupby(cat_col)[sales_col].sum().reset_index().sort_values(by=sales_col, ascending=False)

@st.cache_data(show_spinner=False)
def get_ai_context(df):
    """
    Generates a lightweight string summary of the dataframe for the AI.
    Cached to prevent recalculating df.describe() on every chat message.
    """
    if df is None or df.empty:
        return {
            "columns": "N/A",
            "num_rows": 0,
            "sample_data": "No data uploaded.",
            "summary_stats": "N/A"
        }
        
    return {
        "columns": ", ".join(df.columns),
        "num_rows": len(df),
        "sample_data": df.head(5).to_string(),
        "summary_stats": df.describe().to_string()
    }

# --------------------------------------------------
# CACHED ML MODELS
# --------------------------------------------------
@st.cache_data(show_spinner="Training AI Models...", ttl=3600)
def run_forecasting_tournament(data_series):
    """
    Trains models and returns the best forecast.
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    y = np.array(data_series)
    x = np.arange(len(y)).reshape(-1, 1)
    
    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    
    # Models
    m1 = LinearRegression().fit(x_train, y_train)
    rmse1 = np.sqrt(mean_squared_error(y_test, m1.predict(x_test)))
    
    z = np.polyfit(x_train.flatten(), y_train, 2)
    rmse2 = np.sqrt(mean_squared_error(y_test, np.poly1d(z)(x_test.flatten())))
    
    m3 = RandomForestRegressor(n_estimators=50, random_state=42).fit(x_train, y_train)
    rmse3 = np.sqrt(mean_squared_error(y_test, m3.predict(x_test)))
    
    results = [
        {"name": "Linear Regression", "rmse": rmse1},
        {"name": "Polynomial (Curve)", "rmse": rmse2},
        {"name": "Random Forest AI", "rmse": rmse3}
    ]
    winner = min(results, key=lambda k: k['rmse'])
    
    # Refit
    future_steps = 30
    future_x = np.arange(len(y) + future_steps).reshape(-1, 1)
    
    if winner["name"] == "Linear Regression":
        forecast = LinearRegression().fit(x, y).predict(future_x)
    elif winner["name"] == "Random Forest AI":
        forecast = RandomForestRegressor(n_estimators=50, random_state=42).fit(x, y).predict(future_x)
    else:
        forecast = np.poly1d(np.polyfit(x.flatten(), y, 2))(future_x.flatten())
        
    return winner["name"], forecast, results

# --------------------------------------------------
# MIGRATION UTILS
# --------------------------------------------------
@st.cache_resource
def migrate_json_to_sql():
    """
    Migration utility: Moves users from users.json to SQL if it exists.
    Handles both plain text and hashed passwords safely.
    """
    JSON_FILE = "users.json"
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r") as f:
                users = json.load(f)
            
            migrated_count = 0
            for email, info in users.items():
                pwd = info["password"]
                
                # Check if it looks like a SHA256 hash (64 hex chars)
                # If not, hash it before storing
                if len(pwd) != 64 or not all(c in "0123456789abcdefABCDEF" for c in pwd):
                    pwd = hashlib.sha256(pwd.encode()).hexdigest()

                try:
                    with db._get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT OR IGNORE INTO users (email, password_hash, role) VALUES (?, ?, ?)",
                            (email, pwd, info["role"])
                        )
                        if cursor.rowcount > 0:
                            migrated_count += 1
                        conn.commit()
                except Exception as e:
                    print(f"Error migrating {email}: {e}")
            
            if migrated_count > 0:
                os.rename(JSON_FILE, JSON_FILE + ".bak")
                st.toast(f"Successfully migrated {migrated_count} users to SQL Database!", icon="🚀")
        except Exception as e:
            st.error(f"Migration error: {e}")
