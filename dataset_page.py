import streamlit as st
import pandas as pd
import utils
from database import db

def upload_page():
    utils.set_background()
    
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
    
    st.markdown("""
    <div style="text-align: center; margin: 10px 0;">
        <span style="color: #666; font-size: 0.8rem;">— OR —</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Load Demo Data (Test Drive)", use_container_width=True):
        # Generate Sample Data
        data = {
            "Date": pd.date_range(start="2024-01-01", periods=100),
            "Category": ["Electronics", "Clothing", "Home", "Toys"] * 25,
            "Product": [f"Product {i}" for i in range(100)],
            "Sales": [x * 10 + 50 for x in range(100)],
            "Profit": [x * 2 + 10 for x in range(100)],
            "Region": ["North", "South", "East", "West"] * 25
        }
        df = pd.DataFrame(data)
        st.session_state.uploaded_data = df
        st.success("Demo Data Loaded! Redirecting to Dashboard...")
        st.session_state.page = "dashboard"
        st.rerun()

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

            # CLEAN
            df = utils.clean_data(df)

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
