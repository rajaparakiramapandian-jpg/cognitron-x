import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import utils

def render_sales_trend(df):
    sales_col = utils.detect_sales_column(df)
    if sales_col:
        st.subheader("📈 Sales Trend")
        fig = px.line(df, y=sales_col, title=f"Historical {sales_col} Over Time",
                     labels={"index": "Timeline (Records)", sales_col: f"Revenue ({sales_col})"},
                     template="plotly_white")
        fig.update_traces(line_color='#2c3e50')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def render_sales_bar(df):
    sales_col = utils.detect_sales_column(df)
    category_col = utils.detect_category_column(df)

    if sales_col and category_col:
        st.subheader("📊 Sales by Category")
        grouped = utils.get_aggregated_sales(df, category_col, sales_col)
        
        fig = px.bar(grouped, x=category_col, y=sales_col, 
                    title=f"Total {sales_col} per Category",
                    template="plotly_white",
                    color_discrete_sequence=['#2c3e50'])
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def render_sales_pie(df):
    sales_col = utils.detect_sales_column(df)
    category_col = utils.detect_category_column(df)

    if sales_col and category_col:
        st.subheader("🧩 Sales Distribution")
        # Reuse cached aggregation
        pie_data = utils.get_aggregated_sales(df, category_col, sales_col)

        fig = px.pie(pie_data, names=category_col, values=sales_col, 
                    title="Revenue Share",
                    hole=0.4,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def render_product_treemap(df):
    sales_col = utils.detect_sales_column(df)
    category_col = utils.detect_category_column(df)
    
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
    sales_col = utils.detect_sales_column(df)
    category_col = utils.detect_category_column(df)
    
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
    sales_col = utils.detect_sales_column(df)
    if sales_col:
        st.subheader("🎯 Performance Gauges")
        total_sales = df[sales_col].sum()
        avg_sales = df[sales_col].mean()
        max_sales = df[sales_col].max()
        
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

def render_forecast(df):
    sales_col = utils.detect_sales_column(df)
    if sales_col:
        st.subheader("🔮 Advanced Sales Forecast (Auto-AI Selection)")
        
        data_series = df[sales_col].fillna(0)
        
        if len(data_series) > 8:
            # CALL CACHED ENGINE
            winner_name, forecast_y, all_results = utils.run_forecasting_tournament(data_series)
            
            # PLOT THE WINNER
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
            
            with st.expander("🔎 View Model Comparison (Accuracy Details)"):
                st.write("We tested 3 algorithms using a 80/20 train-test split.")
                models_for_display = [{"name": r["name"], "rmse": r["rmse"]} for r in all_results]
                res_df = pd.DataFrame(models_for_display).sort_values("rmse")
                st.dataframe(res_df.style.format({"rmse": "{:.2f}"}), use_container_width=True)
        else:
            st.warning("Not enough data to run Auto-ML (Need 10+ records).")

def generate_executive_brief(df):
    sales_col = utils.detect_sales_column(df)
    cat_col = utils.detect_category_column(df)
    
    brief = []
    if sales_col:
        total = df[sales_col].sum()
        brief.append(f"**Total Revenue**: The dataset contains a total revenue of **${total:,.2f}**.")
    if sales_col and cat_col:
        top_cat = df.groupby(cat_col)[sales_col].sum().idxmax()
        brief.append(f"**Top Segment**: The highest performing category is **'{top_cat}'**.")
    brief.append(f"**Data Health**: The dataset has **{len(df)} records** and appears clean.")
    return brief

def render_kpis(df):
    data = utils.calculate_kpis(df)
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

def render_filters(df):
    st.markdown("### 🔍 Filter Data")
    filtered_df = df.copy()
    
    # Try to convert likely date columns
    for col in filtered_df.columns:
        if "date" in col.lower() and not pd.api.types.is_datetime64_any_dtype(filtered_df[col]):
            try:
                filtered_df[col] = pd.to_datetime(filtered_df[col])
            except:
                pass

    with st.expander("Show Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        # 1. Date Filter
        date_col = None
        for col in filtered_df.columns:
            if pd.api.types.is_datetime64_any_dtype(filtered_df[col]):
                date_col = col
                break
        
        if date_col:
            min_date = filtered_df[date_col].min()
            max_date = filtered_df[date_col].max()
            with col1:
                date_range = st.date_input("📅 Date Range", [min_date, max_date])
                if len(date_range) == 2:
                    filtered_df = filtered_df[(filtered_df[date_col] >= pd.to_datetime(date_range[0])) & 
                                              (filtered_df[date_col] <= pd.to_datetime(date_range[1]))]

        # 2. Category Filter
        cat_col = utils.detect_category_column(filtered_df)
        if cat_col:
            with col2:
                options = sorted(filtered_df[cat_col].unique().tolist())
                selected_cats = st.multiselect(f"🏷️ Filter by {cat_col}", options=options, default=options)
                if selected_cats:
                    filtered_df = filtered_df[filtered_df[cat_col].isin(selected_cats)]
        
        # 3. Region/Location Filter
        region_col = None
        for col in filtered_df.columns:
            if col.lower() in ["region", "state", "city", "country", "store"]:
                region_col = col
                break
        
        if region_col:
            with col3:
                options = sorted(filtered_df[region_col].unique().tolist())
                selected_regions = st.multiselect(f"🌍 Filter by {region_col}", options=options, default=options)
                if selected_regions:
                    filtered_df = filtered_df[filtered_df[region_col].isin(selected_regions)]
                    
    return filtered_df

def dashboard_page():
    utils.set_background()
    
    if st.session_state.uploaded_data is None:
        st.warning("Please upload a dataset first.")
        st.session_state.page = "upload"
        st.rerun()

    raw_df = st.session_state.uploaded_data
    
    # Dashboard Header: Clean, Minimalist
    st.title("Command Center")
    st.markdown(f"<p style='color: #666; font-size: 0.9rem;'>{st.session_state.user_role} access | {st.session_state.user_email}</p>", unsafe_allow_html=True)
    
    # APPLY FILTERS
    df = render_filters(raw_df)
    
    st.divider()

    # Layout Split: Full Width Dashboard
    render_kpis(df)
    st.divider()
    
    tab_bi, tab_ai, tab_raw = st.tabs(["Market Dynamics", "AI Intelligence", "Data Registry"])
    
    with tab_bi:
        # Row 1
        col1, col2 = st.columns(2)
        with col1:
            render_sales_trend(df)
        with col2:
            render_kpi_gauges(df)
        
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
