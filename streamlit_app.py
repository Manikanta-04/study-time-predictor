"""
Study Time Recommendation - Streamlit App with Fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Study Time Recommender",
    page_icon="📚",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .recommendation-number {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Try to load model, use fallback if not available
@st.cache_resource
def load_model():
    """Load the trained model or return fallback function"""
    try:
        if os.path.exists('models/latest_model.pkl') and os.path.exists('models/scaler.pkl'):
            model = joblib.load('models/latest_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            st.success("✅ Model loaded successfully!")
            return model, scaler, "ml"
        else:
            st.warning("⚠️ Using rule-based predictions (ML model files not found)")
            return None, None, "rule"
    except Exception as e:
        st.warning(f"⚠️ Using rule-based predictions: {e}")
        return None, None, "rule"

# Load model or fallback
model, scaler, mode = load_model()

# Rule-based prediction function
def rule_based_prediction(gpa, difficulty, past_performance, available_hours):
    """Fallback prediction using simple rules"""
    base_hours = 2.0
    
    # GPA factor
    if gpa < 2.0:
        base_hours += 2.0
    elif gpa < 2.5:
        base_hours += 1.5
    elif gpa < 3.0:
        base_hours += 1.0
    elif gpa > 3.5:
        base_hours -= 0.5
    
    # Difficulty factor
    base_hours += (difficulty - 3) * 0.5
    
    # Past performance factor
    if past_performance < 60:
        base_hours += 1.0
    elif past_performance < 70:
        base_hours += 0.5
    elif past_performance > 90:
        base_hours -= 0.5
    
    # Cap by available hours
    prediction = min(base_hours, available_hours)
    prediction = max(0.5, prediction)
    return round(prediction * 2) / 2

# Header
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem;'>
    <h1>📚 Study Time Recommender</h1>
    <p>Get personalized daily study time recommendations based on your profile</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Individual Prediction", "📊 Batch Prediction", "ℹ️ Info"])

with tab1:
    st.markdown("### Enter Your Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gpa = st.number_input("📈 GPA (0-4.0)", 0.0, 4.0, 3.0, 0.1)
        difficulty = st.selectbox("🎯 Subject Difficulty", [1,2,3,4,5], 
                                 format_func=lambda x: ["Very Easy","Easy","Moderate","Hard","Very Hard"][x-1])
    
    with col2:
        past_performance = st.number_input("📊 Past Performance (0-100%)", 0, 100, 75)
        available_hours = st.number_input("⏰ Available Hours", 0.5, 24.0, 4.0, 0.5)
    
    if st.button("🎯 Get Recommendation", use_container_width=True):
        if mode == "ml" and model is not None:
            # ML prediction
            input_data = np.array([[gpa, difficulty, past_performance, available_hours]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
        else:
            # Rule-based fallback
            prediction = rule_based_prediction(gpa, difficulty, past_performance, available_hours)
        
        prediction = round(prediction * 2) / 2
        
        # Display
        st.markdown("---")
        st.markdown(f"""
        <div class="recommendation-box">
            <h2>Recommended Study Time</h2>
            <div class="recommendation-number">{prediction} hours/day</div>
            <p>Based on your inputs</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tips
        tips = []
        if difficulty >= 4:
            tips.append("🎯 Practice active recall for difficult subjects")
        if gpa < 2.5:
            tips.append("📝 Focus on foundational concepts first")
        if prediction > 5:
            tips.append("📚 Break into 45-min sessions with breaks")
        tips.append("✅ Review within 24 hours")
        tips.append("⏰ Study during peak energy times")
        
        st.markdown("### 💡 Tips")
        for tip in tips:
            st.info(tip)

with tab2:
    st.markdown("### Batch Prediction")
    st.markdown("Upload CSV with columns: `gpa, difficulty, past_performance, available_hours`")
    
    uploaded = st.file_uploader("Choose CSV", type="csv")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        if all(c in df.columns for c in ['gpa', 'difficulty', 'past_performance', 'available_hours']):
            
            # Make predictions
            predictions = []
            for _, row in df.iterrows():
                if mode == "ml" and model is not None:
                    inp = np.array([[row['gpa'], row['difficulty'], row['past_performance'], row['available_hours']]])
                    inp_scaled = scaler.transform(inp)
                    pred = model.predict(inp_scaled)[0]
                else:
                    pred = rule_based_prediction(row['gpa'], row['difficulty'], 
                                                row['past_performance'], row['available_hours'])
                predictions.append(round(pred * 2) / 2)
            
            df['recommended_hours'] = predictions
            
            st.success(f"✅ Processed {len(df)} students")
            st.dataframe(df)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
        else:
            st.error("Missing required columns!")

with tab3:
    st.markdown("### About")
    if mode == "ml":
        st.success("✅ Using ML model for predictions")
        if os.path.exists('models/model_metadata.csv'):
            meta = pd.read_csv('models/model_metadata.csv').iloc[0]
            st.metric("Model", meta['model_name'])
            st.metric("R² Score", f"{float(meta['r2_score']):.4f}")
    else:
        st.info("ℹ️ Using rule-based predictions (ML model files not found)")
        st.markdown("""
        **How to add ML model:**
        1. Train model locally with `train_model.py`
        2. Commit model files to GitHub:
           ```bash
           git add -f models/*.pkl
           git commit -m "Add model files"
           git push
           ```
        3. Redeploy on Streamlit Cloud
        """)

st.markdown("---")
st.caption("Made with ❤️ for better learning")