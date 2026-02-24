"""
Study Time Recommendation - Streamlit App
Run this for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Study Time Recommender",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
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
    .tip-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('models/latest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Load metadata if available
        metadata = None
        if os.path.exists('models/model_metadata.csv'):
            metadata = pd.read_csv('models/model_metadata.csv').iloc[0]
        
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load the model
model, scaler, metadata = load_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 Study Time Recommender</h1>
    <p>Get personalized daily study time recommendations based on your profile</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Individual Prediction", "📊 Batch Prediction", "ℹ️ Model Info"])

with tab1:
    st.markdown("### Enter Your Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gpa = st.number_input("📈 GPA (0-4.0)", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
        difficulty = st.selectbox(
            "🎯 Subject Difficulty",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: ["1 - Very Easy", "2 - Easy", "3 - Moderate", "4 - Hard", "5 - Very Hard"][x-1]
        )
    
    with col2:
        past_performance = st.number_input("📊 Past Performance (0-100%)", min_value=0, max_value=100, value=75)
        available_hours = st.number_input("⏰ Available Hours Per Day", min_value=0.5, max_value=24.0, value=4.0, step=0.5)
    
    if st.button("🎯 Get Recommendation", use_container_width=True):
        if model is not None and scaler is not None:
            try:
                # Prepare input
                input_data = np.array([[gpa, difficulty, past_performance, available_hours]])
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                prediction = round(prediction * 2) / 2
                prediction = max(0.5, min(available_hours, prediction))
                
                # Generate tips
                tips = []
                if prediction >= 5:
                    tips.append("📚 Break study time into 45-50 min sessions with 10 min breaks")
                elif prediction >= 3:
                    tips.append("📚 Use Pomodoro technique: 25 min study, 5 min break")
                else:
                    tips.append("📚 Focus on high-yield topics and active recall")
                
                if difficulty >= 4:
                    tips.append("🎯 Practice active recall and teach others")
                
                if gpa < 2.5:
                    tips.append("📝 Master foundational concepts first")
                elif gpa > 3.5:
                    tips.append("🌟 Challenge yourself with advanced problems")
                
                tips.append("✅ Review within 24 hours for better retention")
                tips.append("⏰ Study during peak energy times")
                
                # Display recommendation
                st.markdown("---")
                st.markdown(f"""
                <div class="recommendation-box">
                    <h2>Your Personalized Recommendation</h2>
                    <div class="recommendation-number">{prediction} hours/day</div>
                    <p>Based on your inputs, this is the optimal study time.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display tips
                st.markdown("### 💡 Study Tips Just For You")
                for tip in tips:
                    st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
                
                # Show input summary
                with st.expander("📋 View Input Summary"):
                    summary_df = pd.DataFrame({
                        "Metric": ["GPA", "Difficulty", "Past Performance", "Available Hours"],
                        "Value": [gpa, f"{difficulty}/5", f"{past_performance}%", f"{available_hours} hours"]
                    })
                    st.dataframe(summary_df, hide_index=True)
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            st.error("Model not loaded. Please check the models directory.")

with tab2:
    st.markdown("### Batch Prediction")
    st.markdown("Upload a CSV file with multiple students to get recommendations for all at once.")
    
    # Show sample format
    with st.expander("📁 View Required CSV Format"):
        sample_df = pd.DataFrame({
            'gpa': [3.8, 2.5, 3.2],
            'difficulty': [2, 4, 5],
            'past_performance': [85, 60, 70],
            'available_hours': [5, 3, 2]
        })
        st.dataframe(sample_df)
        st.caption("Your CSV must have exactly these column names: gpa, difficulty, past_performance, available_hours")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['gpa', 'difficulty', 'past_performance', 'available_hours']
            if all(col in df.columns for col in required_cols):
                
                if st.button("🚀 Process Batch", use_container_width=True):
                    with st.spinner("Processing..."):
                        # Make predictions
                        predictions = []
                        for _, row in df.iterrows():
                            input_data = np.array([[
                                row['gpa'], row['difficulty'], 
                                row['past_performance'], row['available_hours']
                            ]])
                            input_scaled = scaler.transform(input_data)
                            pred = model.predict(input_scaled)[0]
                            pred = round(pred * 2) / 2
                            predictions.append(pred)
                        
                        df['recommended_hours'] = predictions
                        
                        # Display results
                        st.success(f"✅ Processed {len(df)} students successfully!")
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average", f"{df['recommended_hours'].mean():.1f} hrs")
                        with col2:
                            st.metric("Minimum", f"{df['recommended_hours'].min():.1f} hrs")
                        with col3:
                            st.metric("Maximum", f"{df['recommended_hours'].max():.1f} hrs")
                        
                        # Show results
                        st.markdown("### Results")
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Visualize
                        if len(df) > 1:
                            fig = px.bar(df, x=df.index, y='recommended_hours', 
                                       title='Recommended Hours Distribution',
                                       labels={'index': 'Student', 'recommended_hours': 'Hours'})
                            st.plotly_chart(fig, use_container_width=True)
                            
            else:
                missing = set(required_cols) - set(df.columns)
                st.error(f"Missing columns: {missing}. Please check the format.")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.markdown("### Model Information")
    
    if metadata is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Details")
            st.metric("Model Type", metadata['model_name'])
            st.metric("R² Score", f"{float(metadata['r2_score']):.4f}")
            st.metric("RMSE", f"{float(metadata['rmse']):.4f}")
            st.metric("MAE", f"{float(metadata['mae']):.4f}")
        
        with col2:
            st.markdown("#### Training Details")
            st.metric("Training Samples", int(metadata['training_samples']))
            st.metric("Test Samples", int(metadata['test_samples']))
            st.markdown(f"**Training Date:** {metadata['training_date']}")
    
    # Feature importance
    if os.path.exists('models/feature_importance.png'):
        st.markdown("#### Feature Importance")
        st.image('models/feature_importance.png', use_container_width=True)
    
    # Correlation matrix
    if os.path.exists('models/correlation_matrix.png'):
        st.markdown("#### Feature Correlation")
        st.image('models/correlation_matrix.png', use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ for personalized learning</p>",
    unsafe_allow_html=True
)