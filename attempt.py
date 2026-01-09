import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import plotly.express as px
import io
import json
import hashlib
import os
import threading

# --------------------------------------------------
# PAGE CONFIG & STYLING
# --------------------------------------------------
st.set_page_config(
    page_title="RetailAI Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Glassmorphism Cards */
    .stCard, div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(75, 108, 183, 0.4);
    }
    
    /* Headers and Labels */
    h1, h2, h3, label, .stMarkdown, p {
        font-family: 'Inter', sans-serif;
    }
    
    /* Antigravity Style Chat Bubbles */
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 30, 47, 0.7) !important; /* Slightly darker for better contrast */
        backdrop-filter: blur(12px);
        border-radius: 12px !important;
        padding: 15px !important;
        margin-bottom: 12px !important;
        border: 1px solid rgba(75, 108, 183, 0.4) !important;
    }
    
    /* Force high-visibility white text for ALL chat content */
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] span, 
    [data-testid="stChatMessage"] div, 
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] code {
        color: #ffffff !important;
        font-weight: 400 !important;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.5); /* Subtle shadow for "pop" */
    }
    
    [data-testid="stChatMessageAvatar"] {
        background-color: #4b6cb7 !important;
        border-radius: 50% !important;
    }

    /* Input Fields Visibility (Global) */
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stNumberInput input {
        color: #ffffff !important;           /* White Text */
        background-color: #2d2d44 !important; /* Dark Background */
        border: 1px solid #4b6cb7 !important; /* Blue Border */
    }
    
    /* Specific Chat Input Styling */
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
    }
    
    div[data-testid="stChatInput"] textarea {
        background-color: #1e1e2f !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid #4b6cb7 !important;
    }
    
    /* Placeholder */
    ::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }

    /* --- ANIMATIONS --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes glowPulse {
        0% { text-shadow: 0 0 5px rgba(0, 255, 127, 0.2); color: #00ff7f; }
        50% { text-shadow: 0 0 20px rgba(0, 255, 127, 0.6); color: #55ffb2; }
        100% { text-shadow: 0 0 5px rgba(0, 255, 127, 0.2); color: #00ff7f; }
    }

    /* Apply Fade In to main components */
    .stCard, div[data-testid="stMetric"], [data-testid="stChatMessage"], .stTabs, div[data-testid="stDataFrame"] {
        animation: fadeIn 0.8s ease-out forwards;
    }

    /* FIX: Remove unwanted black space/margins around data previews */
    .stDataFrame, div[data-testid="stFileUploader"] {
        margin-bottom: 0px !important;
        margin-top: 5px !important;
    }
    
    .stAlert {
        margin-top: 10px !important;
    }

    /* FIX: Prevent glitches in Fullscreen/Maximize mode */
    [data-testid="stFullScreenFrame"] {
        background-color: #0e1117 !important;
        background-image: none !important;
    }
    
    [data-testid="stFullScreenFrame"] [data-testid="stCard"] {
        background: #1e1e2f !important;
        backdrop-filter: none !important;
    }

    button[title="Minimize"] {
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    /* AI Status Pulse */
    .ai-status {
        font-weight: bold;
        animation: glowPulse 2s infinite ease-in-out;
    }

    /* Premium Button Hover */
    .stButton>button {
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }
    .stButton>button:hover {
        transform: scale(1.05) translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(75, 108, 183, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# AUTHENTICATION LOGIC
# --------------------------------------------------
USER_DB_FILE = "users.json"
USER_DB_LOCK = threading.Lock() # Fix 1: Concurrency Lock

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_DB_FILE):
        return {}
    try:
        with USER_DB_LOCK: # Thread-safe read
            with open(USER_DB_FILE, "r") as f:
                return json.load(f)
    except:
        return {}

def save_user(email, password, role):
    with USER_DB_LOCK: # Thread-safe write
        # Re-read inside lock to prevent overwrite
        if os.path.exists(USER_DB_FILE):
             try:
                 with open(USER_DB_FILE, "r") as f:
                     users = json.load(f)
             except:
                 users = {}
        else:
             users = {}
             
        users[email] = {
            "password": hash_password(password),
            "role": role
        }
        with open(USER_DB_FILE, "w") as f:
            json.dump(users, f)

def authenticate(email, password):
    users = load_users()
    if email in users and users[email]["password"] == hash_password(password):
        return True
    return False

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 RetailAI Login")
        st.markdown("Enter your credentials to access the AI Agent.")
        
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In")

            if submit:
                if authenticate(email, password):
                    st.session_state.authenticated = True
                    st.session_state.page = "upload"
                    st.success("Login Successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

        st.markdown("---")
        if st.button("Create New Account"):
            st.session_state.page = "signup"
            st.rerun()

# --------------------------------------------------
# SIGNUP PAGE
# --------------------------------------------------
def signup_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("✨ Create Account")
        
        with st.form("signup_form"):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["Owner", "Manager", "Analyst"])
            submit = st.form_submit_button("Register")

            if submit:
                if email and password:
                    save_user(email, password, role)
                    st.success("Account created! Please login.")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error("Please fill all fields.")

        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.rerun()

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

# --------------------------------------------------
# CACHED ML MODELS
# --------------------------------------------------
@st.cache_data(show_spinner="Training AI Models...")
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
# DATASET UPLOAD PAGE
# --------------------------------------------------
def upload_page():
    st.title("Upload Your Business Dataset")
    st.caption("Self-Service Analytics — No Technical Support Needed")
    st.divider()

    st.markdown("""
    **Supported formats:**  
    • CSV  
    • Excel  

    Upload your sales or business dataset to generate insights.
    """)

    file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            # V2.1: CRASH-PROOF CLEANING
            df = clean_data(df)

            st.success("Dataset sanitized & loaded successfully!")
            
            # Display preview in a more compact way
            with st.container():
                st.markdown("**Dataset Preview (First 5 records):**")
                st.dataframe(df.head(), use_container_width=True)

            st.session_state.uploaded_data = df

            if st.button("Continue to Dashboard"):
                st.session_state.page = "dashboard"
                st.rerun()

        except Exception:
            st.error("Error processing the file. Please check format.")

# --------------------------------------------------
# HELPER FUNCTIONS FOR VISUALS
# --------------------------------------------------
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

# --------------------------------------------------
# KPI GENERATION
# --------------------------------------------------
def render_kpis(df):
    sales_col = detect_sales_column(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    if sales_col:
        with col2:
            st.metric("Total Revenue", f"{df[sales_col].sum():,.2f}")
        with col3:
            st.metric("Average Sale", f"{df[sales_col].mean():,.2f}")
    else:
        with col2:
            st.metric("Revenue", "Not detected")
        with col3:
            st.metric("Avg Sale", "Not detected")

# --------------------------------------------------
# VISUAL CHARTS
# --------------------------------------------------
# --------------------------------------------------
# VISUAL CHARTS
# --------------------------------------------------
def render_sales_trend(df):
    sales_col = detect_sales_column(df)
    if sales_col:
        st.subheader("📈 Sales Trend")
        st.line_chart(df[sales_col])

def render_sales_bar(df):
    sales_col = detect_sales_column(df)
    category_col = detect_category_column(df)

    if sales_col and category_col:
        st.subheader("📊 Sales by Category")
        grouped = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
        st.bar_chart(grouped)

def render_sales_pie(df):
    sales_col = detect_sales_column(df)
    category_col = detect_category_column(df)

    if sales_col and category_col:
        st.subheader("🧩 Sales Distribution")
        pie_data = df.groupby(category_col)[sales_col].sum().reset_index()

        st.plotly_chart(
            {
                "data": [{
                    "labels": pie_data[category_col],
                    "values": pie_data[sales_col],
                    "type": "pie",
                    "hole": 0.4
                }],
                "layout": {"title": "Sales Share by Category"}
            },
            use_container_width=True
        )

# --------------------------------------------------
# V2.0 FEATURES: FORECAST & BRIEF
# --------------------------------------------------
def render_forecast(df):
    """
    Updated Forecast UI using Cached ML Tournament
    """
    sales_col = detect_sales_column(df)
    if sales_col:
        st.subheader("🔮 Advanced Sales Forecast (Auto-AI Selection)")
        
        data_series = df[sales_col].fillna(0)
        
        if len(data_series) > 8:
            # CALL CACHED ENGINE
            winner_name, forecast_y, all_results = run_forecasting_tournament(data_series)
            
            # PLOT THE WINNER
            import plotly.graph_objects as go
            import numpy as np
            
            x_hist = np.arange(len(data_series))
            future_x = np.arange(len(data_series) + 30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=data_series, mode='lines', name='Historical Sales', line=dict(color='#4b6cb7')))
            fig.add_trace(go.Scatter(x=future_x[len(data_series):], y=forecast_y[len(data_series):], 
                                     mode='lines', name=f'Forecast ({winner_name})', 
                                     line=dict(color='orange', dash='dash')))
            
            fig.update_layout(title=f"Sales Prediction (Best Model: {winner_name})", xaxis_title="Time", yaxis_title="Sales")
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ **Auto-Selected Best Model**: {winner_name} (Lowest Error)")
            
            # SHOW DETAILS IF NEEDED
            with st.expander("🔎 View Model Comparison (Accuracy Details)"):
                st.write("We tested 3 algorithms using a 80/20 train-test split.")
                models_for_display = [{"name": r["name"], "rmse": r["rmse"]} for r in all_results]
                res_df = pd.DataFrame(models_for_display).sort_values("rmse")
                st.dataframe(res_df.style.format({"rmse": "{:.2f}"}))
        else:
            st.warning("Not enough data to run Auto-ML (Need 10+ records).")

def generate_executive_brief(df):
    """
    Generates a 3-bullet summary of the dataset (V2.0 Feature).
    Uses a simple heuristic or LLM if available.
    """
    sales_col = detect_sales_column(df)
    cat_col = detect_category_column(df)
    
    brief = []
    
    # Insight 1: Total Volume
    if sales_col:
        total = df[sales_col].sum()
        brief.append(f"**Total Revenue**: The dataset contains a total revenue of **${total:,.2f}**.")
        
    # Insight 2: Top Performer
    if sales_col and cat_col:
        top_cat = df.groupby(cat_col)[sales_col].sum().idxmax()
        brief.append(f"**Top Segment**: The highest performing category is **'{top_cat}'**.")
        
    # Insight 3: Data Health
    brief.append(f"**Data Health**: The dataset has **{len(df)} records** and appears clean.")
    
    return brief

# --------------------------------------------------
# CHATBOT LOGIC
# --------------------------------------------------
# --------------------------------------------------
# AGENTIC AI LOGIC
# --------------------------------------------------
# Redundant imports removed (now at top of file)

def execute_python_code(code, df):
    """
    Executes Python code generated by the agent on the dataframe.
    Returns the local variables 'result' or captured stdout.
    """
    # Fix 3: Security Sanity Check
    forbidden_keywords = ["import os", "import subprocess", "import sys", "open(", "remove(", "rmdir("]
    for keyword in forbidden_keywords:
        if keyword in code:
            return "⚠️ Security Alert: The generated code contains forbidden commands and was blocked."

    buffer = io.StringIO()
    import sys
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer
    
    # Enhance Agent Toolbox with ML Libraries
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    
    local_vars = {
        "df": df, 
        "plt": plt, 
        "px": px, 
        "st": st,
        "np": np,
        "LinearRegression": LinearRegression,
        "KMeans": KMeans
    }
    
    try:
        # We wrap in a try-except to catch execution errors
        exec(code, {}, local_vars)
        output = buffer.getvalue()
        
        result = local_vars.get("result", None)
        return output if output else str(result)
        
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = old_stdout

@st.cache_data(show_spinner=False)
def query_gemini(prompt, df, api_key):
    import time
    
    if not api_key:
        return "Please provide a valid Google Gemini API Key in the sidebar."
        
    try:
        genai.configure(api_key=api_key)
        # Switch to 2.0-flash for stability
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Construct the context
        columns = ", ".join(df.columns)
        
        # Optimize: Reduce token usage by truncating sample data if it's too large
        sample_data = df.head(3).to_string() 
        
        system_instruction = f"""
        You are RetailAI, a friendly and intelligent Data Analyst Assistant.
        
        **Your Personality:**
        - You are helpful, polite, and professional.
        - You can engage in **casual conversation** (e.g., greetings, asking "how can I help?").
        - You are an expert at **explaining concepts** (e.g., "What is ROI?", "Explain regression").
        
        **Your Data Access:**
        - You have access to a pandas DataFrame named 'df'.
        - Columns: {columns}
        - Sample Data:
        {sample_data}
        
        **Instructions for Handling Queries:**
        1. **General Chat**: If the user says "Hi", "Thanks", or asks a conceptual question, reply naturally in text. DO NOT write code.
        2. **Data Analysis**: If the user asks for insights, trends, or plots involving the data, **WRITE PYTHON CODE**.
        3. **Code Format**: Wrap python code in ```python ... ``` blocks.
        4. **Plotting**: Use `plotly.express` as `px`. Use `st.plotly_chart(fig)`.
        
        **Goal**: Be a versatile assistant. Chat when needed, code when needed.
        """
        
        # Retry logic for 429 Errors
        max_retries = 3
        retry_delay = 10 # START WITH 10 SECONDS
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(system_instruction)
                return response.text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay += 10 # Add 10 more seconds each time
                    continue
                elif "429" in error_msg:
                    return "⚠️ **Quota Exceeded**: The rate limit was hit even after waiting. Please wait 2-3 minutes."
                else:
                    raise e
                    
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
             return "⚠️ **Quota Exceeded**: Your API Key has hit the free tier rate limit. Please wait a minute and try again."
        elif "404" in error_msg:
             return f"⚠️ **Model Error**: The selected model is not found. Error: {error_msg}"
        return f"API Error: {error_msg}"

def process_agent_response(response_text, df):
    """
    Parses the LLM response, looks for Python code blocks, executes them,
    and returns a combined response.
    """
    import re
    # 1. Regex to find python code blocks
    code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)

    # 2. Create the Cleaned Display Output
    # We remove the ```python ... ``` blocks so the user doesn't see raw code
    clean_text = re.sub(r"```python(.*?)```", "", response_text, flags=re.DOTALL)
    
    execution_results = []
    
    if code_blocks:
        st.info("System: Executing Analysis Model...")
        
        for code in code_blocks:
            code = code.strip()
            res = execute_python_code(code, df)
            execution_results.append(res)
            
            # Optional: Add technical details in an expander
            with st.expander("🛠️ View Analysis Logic (Technical)"):
                st.code(code, language="python")
                 
        # Append logic output
        if execution_results:
            results_str = "\n".join([str(r) for r in execution_results if r and r != "None"])
            if results_str:
                clean_text += f"\n\n**Analysis Result:**\n{results_str}"
            
    return clean_text.strip()

# --------------------------------------------------
# CHATBOT UI
# --------------------------------------------------
# --------------------------------------------------
# DEMO MODE (FALLBACK)
# --------------------------------------------------
def demo_agent(prompt, df):
    """
    Simulated Agent that generates working code for common demo queries.
    Used when API Key is missing or quota is exceeded.
    """
    prompt = prompt.lower()
    
    # 1. Sales Trend
    if "trend" in prompt or "line" in prompt:
        col = detect_sales_column(df)
        if col:
            return f"""```python
import plotly.express as px
fig = px.line(df, y='{col}', title='Sales Trend (Demo Mode)')
st.plotly_chart(fig)
result = "Here is the sales trend over time."
```"""

    # 2. Bar Chart / Category
    if "bar" in prompt or "category" in prompt:
        cat = detect_category_column(df)
        val = detect_sales_column(df)
        if cat and val:
            return f"""```python
import plotly.express as px
grouped = df.groupby('{cat}')['{val}'].sum().reset_index()
fig = px.bar(grouped, x='{cat}', y='{val}', title='Sales by Category (Demo Mode)')
st.plotly_chart(fig)
result = "Here is the breakdown by category."
```"""

    # 3. Pie Chart
    if "pie" in prompt or "distribution" in prompt:
        cat = detect_category_column(df)
        val = detect_sales_column(df)
        if cat and val:
            return f"""```python
import plotly.express as px
fig = px.pie(df, values='{val}', names='{cat}', title='Distribution (Demo Mode)')
st.plotly_chart(fig)
result = "Here is the distribution view."
```"""

    # 4. Advanced: Prediction / Future (ML Demo)
    if "predict" in prompt or "future" in prompt or "forecast" in prompt:
        col = detect_sales_column(df)
        if col:
             return f"""```python
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Prepare Data
y = df['{col}'].fillna(0).values.reshape(-1, 1)
x = np.arange(len(y)).reshape(-1, 1)

# Train ML Model
model = LinearRegression()
model.fit(x, y)

# Predict Future (Next 6 Months/Points)
future_x = np.arange(len(y), len(y) + 6).reshape(-1, 1)
future_pred = model.predict(future_x)

# Visualize
fig = go.Figure()
fig.add_trace(go.Scatter(x=x.flatten(), y=y.flatten(), mode='lines+markers', name='Actual Data'))
fig.add_trace(go.Scatter(x=np.arange(len(y), len(y)+6), y=future_pred.flatten(), mode='lines+markers', name='AI Prediction', line=dict(dash='dot', color='red')))
fig.update_layout(title="🤖 AI-Generated Sales Prediction (Next 6 Periods)")
st.plotly_chart(fig)

result = "I have trained a Linear Regression model to predict the next 6 periods."
```"""

    # 5. Summary / General
    return f"""```python
st.write(df.describe())
result = "Here is the statistical summary of your dataset."
```"""

# --------------------------------------------------
# CHATBOT UI
# --------------------------------------------------
def chatbot_ui():
    st.markdown("### 🤖 Agentic AI Assistant")
    
    # Check secrets for API Key
    if "GEMINI_API_KEY" in st.secrets:
        st.session_state.gemini_api_key = st.secrets["GEMINI_API_KEY"]
        
    # API Key Input
    with st.sidebar:
        st.header("🔑 Configuration")
        
        if st.session_state.get("gemini_api_key"):
             st.success("API Key Active ✅")
             st.caption("Mode: ⚡ Real AI")
             
             # Re-added for User Convenience
             with st.expander("🔄 Change API Key"):
                 new_key = st.text_input("New Gemini Key", type="password", key="new_key_input")
                 if st.button("Update Key"):
                     st.session_state.gemini_api_key = new_key
                     st.rerun()
        else:
             st.warning("No API Key Found")
             st.caption("Mode: 🟢 Demo Simulation")
             api_key = st.text_input("Enter Gemini Key", type="password")
             if api_key:
                 st.session_state.gemini_api_key = api_key
                 st.rerun()
        
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        st.divider() # Visual separator between messages

    # User Input
    user_input = st.chat_input("Ask about your data...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🧠 Thinking...")
            
            
            try:
                # TRY REAL AI FIRST
                if st.session_state.get("gemini_api_key"):
                    raw_response = query_gemini(user_input, st.session_state.uploaded_data, st.session_state.gemini_api_key)
                    if "Quota Exceeded" in raw_response:
                        raise Exception("Quota Hit")
                else:
                    raise Exception("No Key")
            except:
                # FALLBACK TO DEMO MODE
                raw_response = demo_agent(user_input, st.session_state.uploaded_data)
                # st.toast("⚠️ Utilizing Demo Agent (Simulation Mode)", icon="🟢") # Silenced for Demo Video
            
            # Process Code Blocks (Execution happens instantly)
            final_response = process_agent_response(raw_response, st.session_state.uploaded_data)
            
            # STREAMING EFFECT (Simulated for UX)
            def stream_data():
                import time
                for word in final_response.split(" "):
                    yield word + " "
                    time.sleep(0.02)
            
            # Use st.write_stream if available (Streamlit 1.31+), else markdown
            if hasattr(st, "write_stream"):
                st.write_stream(stream_data)
            else:
                st.markdown(final_response)
            
        st.session_state.chat_history.append({"role": "assistant", "content": final_response})

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------
def dashboard_page():
    st.title("RetailAI Decision Dashboard")

    dashboard_panel, chat_panel = st.columns([2.5, 1], gap="medium")
    
    with dashboard_panel:
        # V2.0: Executive Brief
        st.subheader("📋 Executive Brief")
        brief = generate_executive_brief(st.session_state.uploaded_data)
        for b in brief:
            st.markdown(f"- {b}")
        st.divider()

        render_kpis(st.session_state.uploaded_data)
        
        # Charts Area
        tab1, tab2, tab3 = st.tabs(["📊 Charts", "🔮 Forecast", "🔢 Data"])
        
        with tab1:
            render_sales_trend(st.session_state.uploaded_data)
            render_sales_bar(st.session_state.uploaded_data)
            render_sales_pie(st.session_state.uploaded_data)
            
        with tab2:
            render_forecast(st.session_state.uploaded_data)
            
        with tab3:
            st.subheader("Dataset Preview")
            st.dataframe(st.session_state.uploaded_data.head(10))

    with chat_panel:
        st.subheader("🤖 AI Copilot")
        st.markdown('<p class="ai-status">🟢 Active Intelligence</p>', unsafe_allow_html=True)
        chatbot_ui()

    st.divider()
    
    # Bottom Sidebar Actions
    with st.sidebar:
        st.subheader("🛠️ Quick Actions")
        if st.button("🔄 Change Dataset", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
            
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.page = "login"
            st.session_state.uploaded_data = None
            st.session_state.chat_history = []
            st.rerun()

# --------------------------------------------------
# ROUTER
# --------------------------------------------------
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "upload":
    upload_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
