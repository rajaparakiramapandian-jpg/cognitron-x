import streamlit as st
import utils
from database import db

def authenticate(email, password):
    role = db.authenticate_user(email, password)
    if role:
        st.session_state.user_role = role
        st.session_state.user_email = email
        return True
    return False

def register_user(email, password, role):
    return db.add_user(email, password, role)

def login_page():
    utils.set_background()
    
    # Custom CSS - CINEMATIC ENTERPRISE STYLE (Stripe/OpenAI inspired)
    st.markdown("""
<style>
/* -------------------------------------------------------------------------
   GLOBAL RESET & FONTS
------------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Main Background handled in utils.py */

/* Animation Keyframes */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes inputFocus {
    from { width: 0%; }
    to { width: 100%; }
}

/* -------------------------------------------------------------------------
   CONTAINER: COMPACT GLASS CARD
------------------------------------------------------------------------- */
[data-testid="stVerticalBlockBorder"] {
    /* Dimensions & Layout */
    max-width: 420px !important;
    margin: 40px auto !important; /* Center horizontally */
    padding: 40px 30px !important;
    
    /* Glassmorphism */
    background: rgba(15, 23, 42, 0.65) !important; /* Navy-Black Glass */
    backdrop-filter: blur(25px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
    
    /* Borders & Rounding */
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.15) !important; /* Top Highlight */
    
    /* Elevation & Glow */
    box-shadow: 
        0 20px 40px -10px rgba(0, 0, 0, 0.6), /* Deep Drop Shadow */
        0 0 0 1px rgba(0, 0, 0, 0.3), /* Border Definition */
        0 0 20px rgba(59, 130, 246, 0.15); /* Subtle Blue Glow */
        
    animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

/* -------------------------------------------------------------------------
   TYPOGRAPHY: HIGH CONTRAST
------------------------------------------------------------------------- */
[data-testid="stVerticalBlockBorder"] h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    color: #ffffff !important;
    text-align: center;
    letter-spacing: -0.03em;
    margin-bottom: 8px !important;
    
    /* Requested "Highlight with Black" */
    text-shadow: 
        0 2px 4px rgba(0,0,0,0.8), 
        0 1px 2px rgba(0,0,0,1) !important;
}

[data-testid="stVerticalBlockBorder"] p {
    color: #cbd5e1 !important; /* Light Grey */
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    text-align: center;
    margin-bottom: 30px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.8) !important;
}

[data-testid="stVerticalBlockBorder"] label {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    margin-bottom: 4px !important;
    letter-spacing: 0.02em;
    text-shadow: 0 1px 2px rgba(0,0,0,0.9) !important; /* Max readability */
}

/* -------------------------------------------------------------------------
   INPUT FIELDS: REFINED & FLOATING-LIKE
------------------------------------------------------------------------- */
/* Floating Label Simulation: Position label to look integrated */
[data-testid="stVerticalBlockBorder"] label {
    margin-bottom: -10px !important;
    position: relative;
    top: 15px; /* Push label down */
    left: 10px;
    z-index: 10;
    font-size: 0.75rem !important;
    color: #94a3b8 !important; /* Soft grey */
    font-weight: 500 !important;
    text-shadow: none !important;
    background: transparent;
    padding: 0 4px;
    pointer-events: none;
    transition: all 0.2s ease;
}

.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background: rgba(10, 15, 30, 0.4) !important; /* Darker background */
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 1.25rem 1rem 0.5rem 1rem !important; /* Extra top padding for label */
    font-size: 0.95rem !important;
    transition: all 0.2s ease;
}

.stTextInput input:hover {
    border-color: rgba(255, 255, 255, 0.25) !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

.stTextInput input:focus {
    background: rgba(10, 15, 30, 0.6) !important;
    border-color: #3b82f6 !important; /* Royal Blue Focus */
    box-shadow: 
        0 4px 12px rgba(0, 0, 0, 0.2),
        inset 0 0 0 1px #3b82f6 !important;
    outline: none;
}

/* -------------------------------------------------------------------------
   BUTTONS: CINEMATIC GRADIENT
------------------------------------------------------------------------- */
div[data-testid="stButton"] > button:first-child:not(.secondary-btn) {
    /* Premium Gradient: Royal Blue -> Deep Navy/Purple */
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
    
    color: white !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important; /* Rounded pill-like */
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important; /* Compact padding */
    font-size: 1rem !important;
    width: auto !important; /* Auto width */
    min-width: 150px !important;
    margin: 15px auto 0 auto !important; /* Center button */
    display: block !important;
    
    box-shadow: 
        0 4px 12px rgba(30, 64, 175, 0.4),
        inset 0 1px 0 rgba(255,255,255,0.2) !important;
        
    transition: transform 0.2s, box-shadow 0.2s !important;
}

div[data-testid="stButton"] > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 
        0 8px 20px rgba(30, 64, 175, 0.6),
        inset 0 1px 0 rgba(255,255,255,0.3) !important;
    filter: brightness(110%);
}

div[data-testid="stButton"] > button:first-child:active {
    transform: translateY(0);
}

/* Secondary Button (Link) */
div[data-testid="stButton"] > button.secondary-btn {
    color: #94a3b8 !important;
    background: transparent !important;
    border: none !important;
    margin-top: 10px !important;
    font-weight: 500 !important;
}
div[data-testid="stButton"] > button.secondary-btn:hover {
    color: #ffffff !important;
    text-shadow: 0 0 5px rgba(255,255,255,0.5);
    text-decoration: none !important;
}

/* Center Align the Button Container Hack */
div.stButton {
    text-align: center;
}

/* Utils */
footer {visibility: hidden;}
.stDeployButton {display:none;}
[data-testid="stHeader"] {background: transparent !important;}
</style>
    """, unsafe_allow_html=True)
    
    # Layout: Center the card
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        # The Card Container
        with st.container(border=True):
            # Professional Logo / Title (Mockup has logo text inside)
            st.markdown('<h1 style="text-align: center; font-size: 3rem !important; text-shadow: 2px 2px 8px #000000; font-weight: 800;"><span style="color: #FFFFFF !important;">Retail</span><i style="color: #FFFFFF !important;">AI</i></h1>', unsafe_allow_html=True)

            # Inputs (Mockup has icons inside, we use emojis in label as substitute)
            email = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Secret Password", type="password", placeholder="Enter your password")
            
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            
            # Primary Action: Log In
            if st.button("Log In 🚀", type="primary", use_container_width=True):
                if authenticate(email, password):
                    st.session_state.authenticated = True
                    st.session_state.page = "upload" # Default landing
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            
            # Secondary Action: Create Account
            st.markdown("<div style='text-align: center; margin-top: 15px;'>", unsafe_allow_html=True)
            if st.button("Create New Account ✨", key="goto_signup"):
                st.session_state.page = "signup"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

def signup_page():
    utils.set_background()
    
    # Custom CSS - CINEMATIC ENTERPRISE STYLE (Consistency) - Refined
    st.markdown("""
<style>
/* -------------------------------------------------------------------------
   GLOBAL RESET & FONTS
------------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Main Background handled in utils.py */

/* Animation Keyframes */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* -------------------------------------------------------------------------
   CONTAINER: COMPACT GLASS CARD (Sign Up Version)
------------------------------------------------------------------------- */
[data-testid="stVerticalBlockBorder"] {
    max-width: 480px !important; /* Slightly wider for form */
    margin: 40px auto !important;
    padding: 40px 30px !important;
    
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(25px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
    
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.15) !important;
    
    box-shadow: 
        0 20px 40px -10px rgba(0, 0, 0, 0.6),
        0 0 0 1px rgba(0, 0, 0, 0.3),
        0 0 20px rgba(59, 130, 246, 0.15);
        
    animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

/* -------------------------------------------------------------------------
   TYPOGRAPHY
------------------------------------------------------------------------- */
[data-testid="stVerticalBlockBorder"] h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    color: #ffffff !important;
    text-align: center;
    margin-bottom: 8px !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.8) !important;
}

[data-testid="stVerticalBlockBorder"] p {
    color: #cbd5e1 !important;
    font-family: 'Inter', sans-serif !important;
    text-align: center;
    margin-bottom: 25px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.8) !important;
}

/* Floating Label Simulation */
[data-testid="stVerticalBlockBorder"] label {
    margin-bottom: -10px !important;
    position: relative;
    top: 15px; 
    left: 10px;
    z-index: 10;
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    text-shadow: none !important;
    background: transparent;
    padding: 0 4px;
    pointer-events: none;
    transition: all 0.2s ease;
}

/* -------------------------------------------------------------------------
   INPUT FIELDS
------------------------------------------------------------------------- */
.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background: rgba(10, 15, 30, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 1.25rem 1rem 0.5rem 1rem !important; /* Extra top padding */
    transition: all 0.2s ease;
}

.stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
    background: rgba(10, 15, 30, 0.6) !important;
    border-color: #60a5fa !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2), inset 0 0 0 1px #60a5fa !important;
    outline: none;
}

/* Center Align Buttons */
div.stButton { text-align: center; }
div[data-testid="stButton"] > button:first-child {
    width: auto !important;
    min-width: 150px !important;
    margin: 0 auto !important;
}
</style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        with st.container(border=True):
            st.markdown("<h1 style='text-align: center; color: white !important; text-shadow: 0 0 10px rgba(0,0,0,0.8);'>✨ Join RetailAI</h1>", unsafe_allow_html=True)
            # Subtitle
            st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); margin-bottom: 30px;'>Create your professional account</p>", unsafe_allow_html=True)
            
            with st.form("signup_form"):
                email = st.text_input("Email Address", placeholder="name@company.com")
                password = st.text_input("Set Password", type="password")
                role = st.selectbox("Role", ["Owner", "Manager", "Analyst"])
                
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
                submit = st.form_submit_button("Register Account", use_container_width=True)

                if submit:
                    if email and password:
                        if register_user(email, password, role):
                            st.success("Account created! Please login.")
                            st.session_state.page = "login"
                            st.rerun()
                        else:
                            st.error("Email already registered.")
                    else:
                        st.error("Please fill all fields.")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬅ Back to Login", key="goto_login", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()
