import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "T"
import streamlit as st
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import plotly.express as px
import io
import json
import hashlib
import os
import threading
import threading
from database import db
import login_page as lp
import utils

# --------------------------------------------------
# PAGE CONFIG & STYLING
# --------------------------------------------------
st.set_page_config(
    page_title="RetailAI Agent",
    page_icon="🤖",
    layout="wide"
)



# --------------------------------------------------
# AUTHENTICATION LOGIC (NOW SQL-BASED)
# --------------------------------------------------
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

# Run migration on startup
migrate_json_to_sql()

def authenticate(email, password):
    role = db.authenticate_user(email, password)
    if role:
        st.session_state.user_role = role
        st.session_state.user_email = email
        return True
    return False

def register_user(email, password, role):
    return db.add_user(email, password, role)

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

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "user_role" not in st.session_state:
    st.session_state.user_role = None

# --------------------------------------------------
# DYNAMIC BACKGROUND LOGIC
# --------------------------------------------------
# --------------------------------------------------
# DYNAMIC BACKGROUND LOGIC
# --------------------------------------------------
# Using utils.set_background() instead
utils.set_background()

# --------------------------------------------------
# NAVIGATION SIDEBAR
# --------------------------------------------------
def render_sidebar():
    """
    Unified Navigation for Authenticated Users
    """
    if st.session_state.authenticated:
        with st.sidebar:
            st.title("RetailAI Pro")
            st.caption(f"Logged in as: {st.session_state.user_email}")
            st.divider()
            
            # Smart Navigation
            page = st.radio("Navigate", ["Upload Data", "Dashboard", "AI Agent"], 
                            index=["upload", "dashboard", "chatbot"].index(st.session_state.page) if st.session_state.page in ["upload", "dashboard", "chatbot"] else 1)
            
            if page == "Upload Data":
                st.session_state.page = "upload"
            elif page == "Dashboard":
                st.session_state.page = "dashboard"
            elif page == "AI Agent":
                st.session_state.page = "chatbot"
            
            st.divider()
            
            if st.button("Log Out", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.page = "login"
                st.rerun()

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
# Login and Signup pages are now imported from login_page.py

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
# DATASET UPLOAD PAGE
# --------------------------------------------------
def upload_page():
    st.title("Upload Your Business Dataset")
    st.caption("Self-Service Analytics — No Technical Support Needed")
    st.divider()

    # Create a Card Container
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📤 Import Data")
    st.markdown("""
    **Supported formats:**  
    • CSV  
    • Excel  

    Upload your sales or business dataset to generate insights.
    """)

    file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    st.markdown('</div>', unsafe_allow_html=True)

    if file:
        try:
            if file.name.endswith(".csv"):
                try:
                    # Attempt 1: Default UTF-8
                    df = pd.read_csv(file)
                except UnicodeDecodeError:
                    try:
                        # Attempt 2: Common Windows/Excel encoding
                        file.seek(0)
                        df = pd.read_csv(file, encoding='cp1252')
                    except Exception:
                        # Attempt 3: Comprehensive detection using chardet
                        import chardet
                        file.seek(0)
                        raw_data = file.read(20000)  # Read a larger sample for accuracy
                        result = chardet.detect(raw_data)
                        encoding = result['encoding'] or 'latin1'
                        
                        file.seek(0)
                        df = pd.read_csv(file, encoding=encoding)
            else:
                df = pd.read_excel(file)

            # V2.1: CRASH-PROOF CLEANING
            df = clean_data(df)

            st.success("Dataset sanitized & loaded successfully!")
            
            # Save to Database
            if st.session_state.user_email:
                with st.spinner("Saving dataset to cloud storage..."):
                    db.save_dataset(st.session_state.user_email, file.name, df)
                st.info(f"💾 Dataset '{file.name}' has been saved to your account.")

            # Display preview
            with st.container():
                st.markdown("**Dataset Preview (First 5 records):**")
                st.dataframe(df.head(), use_container_width=True)

            st.session_state.uploaded_data = df

            if st.button("Continue to Dashboard"):
                st.session_state.page = "dashboard"
                st.rerun()

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

    # Load Previous Datasets UI
    if st.session_state.user_email:
        st.divider()
        st.subheader("📂 Your Saved Datasets")
        previous_datasets = db.get_datasets(st.session_state.user_email)
        
        if previous_datasets:
            cols = st.columns([3, 1])
            dataset_options = {f"{name} ({created})": ds_id for ds_id, name, created in previous_datasets}
            selected_ds_name = cols[0].selectbox("Select a previous dataset", options=list(dataset_options.keys()))
            
            if cols[1].button("Load Selected"):
                selected_id = dataset_options[selected_ds_name]
                with st.spinner("Retrieving dataset from database..."):
                    loaded_df = db.load_dataset(selected_id)
                if loaded_df is not None:
                    st.session_state.uploaded_data = loaded_df
                    st.success(f"Loaded {selected_ds_name}!")
                    st.session_state.page = "dashboard"
                    st.rerun()
                else:
                    st.error("Failed to load dataset.")
        else:
            st.info("You haven't saved any datasets yet.")

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

def render_kpis(df):
    data = calculate_kpis(df)
    sales_col = data["sales_col_name"]
    
    st.markdown("""
        <style>
            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                color: #374151;
            }
            [data-testid="stMetricLabel"] {
                color: #6b7280;
                font-size: 0.9rem;
            }
            [data-testid="stMetricValue"] {
                color: #1f2937;
                font-weight: 700;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📦 Total records", f"{data['total_records']:,}")

    if sales_col:
        with col2:
            st.metric("💰 Total Revenue", f"${data['total_revenue']:,.2f}")
        with col3:
            st.metric("📈 Average Sale", f"${data['avg_sale']:,.2f}")
        with col4:
            st.metric("🎯 Max Transaction", f"${data['max_sale']:,.2f}")
    else:
        with col2:
            st.metric("Revenue", "N/A")
        with col3:
            st.metric("Avg Sale", "N/A")
        with col4:
            st.metric("Max Sale", "N/A")

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
        fig = px.line(df, y=sales_col, title=f"Historical {sales_col} Over Time",
                     labels={"index": "Timeline (Records)", sales_col: f"Revenue ({sales_col})"},
                     template="plotly_white")
        fig.update_traces(line_color='#2c3e50')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Presentation Tip:** Explain that this chart shows the **momentum** of your business. The vertical (Y) axis represents your {sales_col}, while the horizontal (X) axis tracks the progression of transactions or time periods.")

@st.cache_data(show_spinner=False)
def get_aggregated_sales(df, cat_col, sales_col):
    return df.groupby(cat_col)[sales_col].sum().reset_index().sort_values(by=sales_col, ascending=False)

def render_sales_bar(df):
    sales_col = detect_sales_column(df)
    category_col = detect_category_column(df)

    if sales_col and category_col:
        st.subheader("📊 Sales by Category")
        grouped = get_aggregated_sales(df, category_col, sales_col)
        
        fig = px.bar(grouped, x=category_col, y=sales_col, 
                    title=f"Total {sales_col} per Category",
                    template="plotly_white",
                    color_discrete_sequence=['#2c3e50'])
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def render_sales_pie(df):
    sales_col = detect_sales_column(df)
    category_col = detect_category_column(df)

    if sales_col and category_col:
        st.subheader("🧩 Sales Distribution")
        # Reuse cached aggregation
        pie_data = get_aggregated_sales(df, category_col, sales_col)

        fig = px.pie(pie_data, names=category_col, values=sales_col, 
                    title="Revenue Share",
                    hole=0.4,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def render_product_treemap(df):
    sales_col = detect_sales_column(df)
    category_col = detect_category_column(df)
    
    sub_cat_col = None
    for col in df.columns:
        if col.lower() in ["sub-category", "product_name", "item_name", "product"]:
            if col != category_col:
                sub_cat_col = col
                break
    
    if sales_col and category_col:
        st.subheader("🌳 Product Hierarchy")
        path = [category_col]
        if sub_cat_col:
            path.append(sub_cat_col)
            
        fig = px.treemap(df, path=path, values=sales_col,
                        template="plotly_white",
                        color_continuous_scale='Blues')
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def render_profit_heatmap(df):
    sales_col = detect_sales_column(df)
    category_col = detect_category_column(df)
    
    if sales_col and category_col:
        st.subheader("🔥 Performance Heatmap")
        dim2 = None
        for col in df.columns:
            if col.lower() in ["region", "state", "city", "segment"]:
                if col != category_col:
                    dim2 = col
                    break
        
        if dim2:
            pivot_df = df.groupby([category_col, dim2])[sales_col].sum().unstack().fillna(0)
            fig = px.imshow(pivot_df, 
                            title=f"{category_col} vs {dim2}",
                            template="plotly_white",
                            color_continuous_scale="Blues")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

def render_kpi_gauges(df):
    sales_col = detect_sales_column(df)
    if sales_col:
        st.subheader("🎯 Performance Gauges")
        total_sales = df[sales_col].sum()
        avg_sales = df[sales_col].mean()
        max_sales = df[sales_col].max()
        
        import plotly.graph_objects as go
        
        fig1 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_sales,
            title = {'text': "Avg Transaction", 'font': {'size': 18}},
            gauge = {'axis': {'range': [0, max_sales], 'tickcolor': "#2c3e50"},
                    'bar': {'color': "#2c3e50"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#e0e0e0",
                    'steps' : [
                        {'range': [0, avg_sales * 0.8], 'color': "#f8d7da"},
                        {'range': [avg_sales * 0.8, avg_sales * 1.2], 'color': "#fff3cd"}]}))
        fig1.update_layout(template="plotly_white", height=250, margin=dict(t=50, b=20, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
            
        fig2 = go.Figure(go.Indicator(
            mode = "number+delta",
            value = total_sales,
            delta = {'reference': total_sales * 0.9, 'relative': True},
            title = {'text': "Current Revenue vs Target", 'font': {'size': 18}}))
        fig2.update_layout(template="plotly_white", height=220, margin=dict(t=50, b=20, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

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
            
            fig.update_layout(title=f"Sales Prediction (Best Model: {winner_name})", 
                              xaxis_title="Time Steps (Future 30 Days)", 
                              yaxis_title=f"Predicted {sales_col}",
                              template="plotly_white",
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ **Auto-Selected Best Model**: {winner_name} (Lowest Error)")
            st.info("**Presentation Tip:** This is the **Forward-Looking** part of the AI. Historical data handles the past, but this model predicts the next 30 steps. Explain that 'Random Forest' or 'Linear Regression' was chosen automatically based on which one fit your specific data best.")
            
            # V2.1: Restored Accuracy Details
            with st.expander("🔎 View Model Comparison (Accuracy Details)"):
                st.write("We tested 3 algorithms using a 80/20 train-test split.")
                models_for_display = [{"name": r["name"], "rmse": r["rmse"]} for r in all_results]
                res_df = pd.DataFrame(models_for_display).sort_values("rmse")
                st.dataframe(res_df.style.format({"rmse": "{:.2f}"}), use_container_width=True)
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

def execute_python_code(code, df, global_context=None):
    """
    Executes Python code generated by the agent on the dataframe.
    Returns the local variables 'result' or captured stdout.
    """
    # Fix 3: Security Sanity Check
    forbidden_keywords = ["import os", "import subprocess", "import sys", "remove(", "rmdir("]
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
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    
    # Use provided context or initialize new
    if global_context is None:
        local_vars = {}
    else:
        local_vars = global_context
        
    # Inject standard variables if not present
    default_vars = {
        "df": df, 
        "pd": pd,
        "plt": plt, 
        "px": px, 
        "st": st,
        "np": np,
        "LinearRegression": LinearRegression,
        "KMeans": KMeans
    }
    
    # Update only if not already set (preserve state)
    for k, v in default_vars.items():
        if k not in local_vars:
            local_vars[k] = v
    
    try:
        # Pass local_vars as both globals and locals for consistency in exec
        exec(code, local_vars, local_vars)
        output = buffer.getvalue().strip()
        result = local_vars.get("result", None)
        
        # Combine stdout and result variable
        final_bits = []
        if output:
            final_bits.append(output)
        if result is not None and str(result) not in output:
            final_bits.append(str(result))
            
        return "\n".join(final_bits) if final_bits else None
        
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = old_stdout

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key):
    return OpenAI(api_key=api_key)

def query_openai_stream(prompt, df, api_key, model="gpt-5-nano"):
    if not api_key:
        yield "Please provide a valid OpenAI API Key in the sidebar."
        return
        
    try:
        client = get_openai_client(api_key)
        
        # Enhanced Context (Cached)
        context = utils.get_ai_context(df)
        
        system_instruction = f"""
        You are 'RetailAI Pro', a versatile AI Assistant powered by OpenAI. 
        You function as a world-class Business Strategy Consultant and Data Scientist, but you are also capable of answering ANY general question (like ChatGPT).

        **CRITICAL: DATA ACCESS CONFIRMED**
        - You HAVE access to the full dataframe in the variable `df`.
        - The `df` is LOADED in memory.
        - You CAN execute Python code to analyze it.
        - **NEVER** say "I don't have access to the full data".
        - **ALWAYS** write Python code to answer data questions.

        **Your Mission:**
        1. **General Assistance**: Answer any question helpfully—from general knowledge to logical reasoning.
        2. **Data Intelligence**: When the query relates to the provided business data (columns: {context['columns']}), perform high-precision analysis.

        **Protocol for Data Queries:**
        If the user asks about the dataframe 'df':
        - **Format:** Keep your text response extremely brief (e.g., "Calculating...").
        - **Analysis:** Write Python code inside ```python ... ``` blocks to calculate the answer.
        - **Output:** You MUST use `print()` to display the final answer in the code block.
        - **Restrictions:**
            - DO NOT explain the code.
            - DO NOT tell the user to run the code.
            - DO NOT say "Data-Driven Conclusion".
            - Just let the code result be the answer.

        **Protocol for General Queries:**
        If the user asks a general question (e.g., "Who is the Prime Minister of India?"), answer directly and conversationally. DO NOT follow the data protocol or write code for non-data questions.

        **Data Context (for reference):**
        - Rows: {context['num_rows']} | Columns: {context['columns']}
        - Sample Data: {context['sample_data']}
        """
        
        # Compatibility Adjustments for newer models (o1, gpt-5)
        completion_args = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }

        # Newer models (o1/gpt-5) sometimes prefer 'user' over 'system' for initial instructions
        # or require 'max_completion_tokens' instead of 'max_tokens'
        if "o1" in model or "gpt-5" in model:
            # For o1-preview/o1-mini, early versions didn't support system messages
            # Converting system instruction to a user message for reliability
            completion_args["messages"] = [
                {"role": "user", "content": f"SYSTEM INSTRUCTION:\n{system_instruction}\n\nUSER QUERY:\n{prompt}"}
            ]
            # completion_args["max_completion_tokens"] = 10000 # Default safe ceiling

        response = client.chat.completions.create(**completion_args)
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                    
    except Exception as e:
        raise e

def process_agent_response(response_text, df):
    """
    Parses the LLM response, looks for Python code blocks, executes them,
    and returns a combined response.
    """
    import re
    # 1. Regex to find python code blocks (flexible on spacing/newlines)
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response_text, re.DOTALL)

    # 2. Create the Cleaned Display Output
    # We remove the code blocks so the user doesn't see raw code
    clean_text = re.sub(r"```(?:python)?\s*(.*?)```", "", response_text, flags=re.DOTALL)
    
    execution_results = []
    
    # Create a shared execution context for multi-block dependencies
    execution_context = {}
    
    if code_blocks:
        st.info("System: Executing Analysis Model...")
        
        for code in code_blocks:
            code = code.strip()
            # Pass and update the shared context
            res = execute_python_code(code, df, global_context=execution_context)
            execution_results.append(res)
            
            # Optional: Add technical details in an expander
            with st.expander("🛠️ View Analysis Logic (Technical)"):
                st.code(code, language="python")
                 
        # Append logic output
        if execution_results:
            results_str = "\n".join([str(r) for r in execution_results if r and r != "None"])
            if results_str:
                clean_text += f"\n\n{results_str}"
            
    return clean_text.strip()

# --------------------------------------------------
# CHATBOT UI
# --------------------------------------------------
def chatbot_ui():
    st.markdown("### 🤖 Agentic AI Assistant")
    
    # Check secrets for API Key
    if "OPENAI_API_KEY" in st.secrets:
        st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
        
    # API Key Input
    with st.sidebar:
        st.header("🔑 Configuration")
        
        if st.session_state.get("openai_api_key"):
            st.success("API Key Active ✅")
            st.caption("Mode: ⚡ OpenAI AI")
             
            # Model Selector
            st.divider()
            st.caption("🤖 Model Selection")
            available_models = ["gpt-5-nano", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
            selected_model = st.selectbox("Choose AI Model", available_models, index=0)
            st.session_state.active_model = selected_model
            
            # Re-added for User Convenience
            with st.expander("🔄 Change API Key"):
                new_key = st.text_input("New OpenAI Key", type="password", key="new_key_input")
                if st.button("Update Key"):
                    st.session_state.openai_api_key = new_key
                    st.rerun()

            st.info(f"⚡ Powered by: **{selected_model.upper()}**")
        else:
            st.warning("No API Key Found")
            api_key = st.text_input("Enter OpenAI Key", type="password")
            if api_key:
                st.session_state.openai_api_key = api_key
                st.rerun()
        
        st.divider()
        st.subheader("🧹 Maintenance")
        if st.button("Clear System Cache"):
            st.cache_data.clear()
            st.success("System cache cleared!")
            st.rerun()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            if st.session_state.user_email:
                db.save_chat_history(st.session_state.user_email, "default", [])
            st.rerun()

    # Load History from DB on first run in dashboard
    if st.session_state.user_email and not st.session_state.chat_history:
        st.session_state.chat_history = db.load_chat_history(st.session_state.user_email, "default")

    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        st.divider() 

    # User Input
    user_input = st.chat_input("Ask about your data...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            # UI: Immediate Response (Performance Optimized)
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # API Key Check
                if st.session_state.get("openai_api_key"):
                    # Use selected model or default to gpt-5-nano
                    active_model = st.session_state.get("active_model", "gpt-5-nano")
                    
                    # Use the restored OpenAI streaming function
                    stream = query_openai_stream(
                        user_input, 
                        st.session_state.uploaded_data, 
                        st.session_state.openai_api_key,
                        model=active_model
                    )
                    
                    buffer_counter = 0
                    for chunk in stream:
                        full_response += chunk
                        buffer_counter += 1
                        # Update UI every 5 chunks to prevent lag
                        if buffer_counter % 5 == 0:
                            message_placeholder.markdown(full_response + "▌")
                    
                    # Final update to ensure everything is shown
                    message_placeholder.markdown(full_response)
                else:
                    st.error("⚠️ OpenAI API Key is required to use the AI Copilot.")
                    st.info("Please enter your API key in the sidebar configuration.")
                    return # Stop further processing if no key
            except Exception as e:
                error_msg = str(e).lower()
                
                # Enhanced error diagnostics for GPT-5 / newer models
                if "429" in error_msg:
                    st.error("🚀 **API Quota Reached**")
                    st.warning(f"Your OpenAI account has reached its limit for **{active_model}**.")
                    st.info("💡 **Why is this happening?** \nGPT-5 is a premium model and often requires 'Tier 4' billing status and enough pre-paid credits. Your account is currently blocking this request.")
                    
                    if active_model == "gpt-5-nano":
                        st.divider()
                        st.subheader("🛠️ Emergency Solution")
                        st.write("If you need to chat right now, you can switch to a more available model:")
                        if st.button("🔄 Emergency Switch to gpt-4o-mini"):
                            st.session_state.emergency_fallback_model = "gpt-4o-mini"
                            st.success("Fallback active! Please try your question again.")
                            st.rerun()
                    else:
                        st.info("💡 **What to do?** \n1. Wait a moment and try again. \n2. Check your [OpenAI Usage Dashboard](https://platform.openai.com/usage). \n3. Ensure your API key has credits.")
                elif "404" in error_msg or "model_not_found" in error_msg:
                    st.error(f"❌ **Model Access Denied: {active_model}**")
                    st.warning(f"Your API key does not appear to have access to the model: `{active_model}`.")
                    st.info("💡 **Note:** GPT-5 models are often rolled out in tiers. Please ensure your account has 'Tier 4' or higher access in the OpenAI Dashboard.")
                elif "503" in error_msg:
                    st.error("🚧 **OpenAI Model is Busy**")
                    st.warning("The selected model is currently experiencing high load.")
                    st.info("Please try again in a few seconds or switch to a faster model like `gpt-4o-mini`.")
                else:
                    st.error(f"⚠️ OpenAI Connection Issue: {str(e)}")
                    st.info(f"Technical Reason: {error_msg}")
                    st.info("Please check your API key / model selection or network connection.")
                return
            
            # Post-Process: Execute Code Blocks (instantly)
            final_display = process_agent_response(full_response, st.session_state.uploaded_data)
            
            # If the cleaner removed code blocks, we update the message
            if final_display != full_response:
                message_placeholder.markdown(final_display)

        st.session_state.chat_history.append({"role": "assistant", "content": final_display})
        
        # Save to Database
        if st.session_state.user_email:
            db.save_chat_history(st.session_state.user_email, "default", st.session_state.chat_history)

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------
def dashboard_page():
    if st.session_state.uploaded_data is None:
        st.warning("Please upload a dataset first.")
        st.session_state.page = "upload"
        st.rerun()

    df = st.session_state.uploaded_data
    
    # Dashboard Header: Clean, Minimalist
    st.title("Command Center")
    st.markdown(f"<p style='color: #666; font-size: 0.9rem;'>{st.session_state.user_role} access | {st.session_state.user_email}</p>", unsafe_allow_html=True)
    
    st.divider()

    # Layout Split: Full Width Dashboard
    
    render_kpis(df)
    st.divider()
    
    tab_bi, tab_ai, tab_raw = st.tabs(["Market Dynamics", "AI Intelligence", "Data Registry"])
    
    with tab_bi:
        # Row 1: Sales Trend (Full Width)
        render_sales_trend(df)
        
        st.divider()

        # Row 2
        col3, col4 = st.columns(2)
        with col3:
            render_sales_bar(df)
        with col4:
            render_sales_pie(df)
            
        # Row 3
        col5, col6 = st.columns(2)
        with col5:
            render_product_treemap(df)
        with col6:
            render_profit_heatmap(df)

        st.divider()
        
        # Row 4: Performance Gauges (Moved to Bottom)
        render_kpi_gauges(df)

    with tab_ai:
        render_forecast(df)
        st.divider()
        st.subheader("Strategy Brief")
        brief = generate_executive_brief(df)
        for b in brief:
            st.write(f"• {b}")

    with tab_raw:
        st.subheader("Data Explorer")
        st.dataframe(df, use_container_width=True)

    # Footer
    st.markdown("<br><hr><div style='text-align: center; color: #999; font-size: 0.8rem;'>RetailAI v2.5 Enterprise — Restricted Access Control</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ROUTER
# --------------------------------------------------
render_sidebar()

if st.session_state.page == "login":
    lp.login_page()
elif st.session_state.page == "signup":
    lp.signup_page()
elif st.session_state.page == "upload":
    upload_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
elif st.session_state.page == "chatbot":
    st.title("AI Intelligence Agent")
    chatbot_ui()
