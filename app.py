from flask import Flask, request, jsonify, render_template, send_file
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os  # <-- This was missing!
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('models/latest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"⚠️ Model not found. Please train the model first. Error: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        # Get data from request
        data = request.json
        
        # Extract features
        gpa = float(data['gpa'])
        difficulty = int(data['difficulty'])
        past_performance = float(data['past_performance'])
        available_hours = float(data['available_hours'])
        
        # Validate inputs
        if not (0 <= gpa <= 4.0):
            return jsonify({'error': 'GPA must be between 0 and 4.0'}), 400
        if not (1 <= difficulty <= 5):
            return jsonify({'error': 'Difficulty must be between 1 and 5'}), 400
        if not (0 <= past_performance <= 100):
            return jsonify({'error': 'Past performance must be between 0 and 100'}), 400
        if not (0.5 <= available_hours <= 24):
            return jsonify({'error': 'Available hours must be between 0.5 and 24'}), 400
        
        # Prepare input
        input_data = np.array([[gpa, difficulty, past_performance, available_hours]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Round to nearest 0.5
        prediction = round(prediction * 2) / 2
        prediction = max(0.5, min(available_hours, prediction))
        
        # Generate tips
        tips = generate_study_tips(prediction, gpa, difficulty)
        
        return jsonify({
            'success': True,
            'recommended_hours': prediction,
            'tips': tips,
            'input_summary': {
                'gpa': gpa,
                'difficulty': difficulty,
                'past_performance': past_performance,
                'available_hours': available_hours
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get CSV file
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Define required columns and their possible variations
        column_mappings = {
            'gpa': ['gpa', 'GPA', 'grade_point_average'],
            'difficulty': ['difficulty', 'difficulty_level', 'subject_difficulty', 'diff_level'],
            'past_performance': ['past_performance', 'past performance', 'previous_score', 'past_score'],
            'available_hours': ['available_hours', 'available hours', 'hours_available', 'study_hours_available']
        }
        
        # Find the actual column names in the uploaded file
        selected_columns = {}
        missing_columns = []
        
        for required_col, possible_names in column_mappings.items():
            found = False
            for possible_name in possible_names:
                if possible_name in df.columns:
                    selected_columns[required_col] = possible_name
                    found = True
                    break
            if not found:
                missing_columns.append(required_col)
        
        # If columns are missing, return helpful error
        if missing_columns:
            return jsonify({
                'error': f'CSV missing required columns. Need: {list(column_mappings.keys())}. Found: {list(df.columns)}'
            }), 400
        
        # Make predictions
        predictions = []
        for _, row in df.iterrows():
            input_data = np.array([[
                row[selected_columns['gpa']], 
                row[selected_columns['difficulty']], 
                row[selected_columns['past_performance']], 
                row[selected_columns['available_hours']]
            ]])
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            pred = round(pred * 2) / 2
            predictions.append(pred)
        
        # Add predictions to dataframe
        df['recommended_hours'] = predictions
        
        # Save to CSV
        output_file = f'batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False)
        
        return jsonify({
            'success': True,
            'message': f'✅ Processed {len(df)} students. Predictions saved!',
            'download_url': f'/download/{output_file}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'File not found: {e}'}), 404

@app.route('/model_info')
def model_info():
    if os.path.exists('models/model_metadata.csv'):
        metadata = pd.read_csv('models/model_metadata.csv').to_dict('records')[0]
        return jsonify(metadata)
    return jsonify({'error': 'Model info not found'})

@app.route('/visualize')
def visualize():
    """Generate and return visualization of model performance"""
    if not os.path.exists('models/model_comparison.csv'):
        return jsonify({'error': 'Model comparison data not found'})
    
    try:
        # Read model comparison data
        comparison_df = pd.read_csv('models/model_comparison.csv', index_col=0)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        comparison_df['R² Score'].plot(kind='bar', color='skyblue')
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({'image': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_study_tips(hours, gpa, difficulty):
    """Generate personalized study tips"""
    tips = []
    
    if hours >= 5:
        tips.append("📚 Break your study time into 45-50 minute sessions with 10-minute breaks")
    elif hours >= 3:
        tips.append("📚 Use the Pomodoro technique: 25 min study, 5 min break")
    else:
        tips.append("📚 Make the most of limited time by focusing on high-yield topics")
    
    if difficulty >= 4:
        tips.append("🎯 For difficult subjects, practice active recall and teach others")
    
    if gpa < 2.5:
        tips.append("📝 Start with foundational concepts before moving to advanced topics")
    elif gpa > 3.5:
        tips.append("🌟 Challenge yourself with additional practice problems")
    
    tips.append("✅ Review material within 24 hours for better retention")
    tips.append("⏰ Study at your peak energy times (morning/evening)")
    
    return tips

# Create HTML template with DARK THEME and HOVER EFFECTS
os.makedirs('templates', exist_ok=True)

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Study Time Recommender</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 20px;
            color: #e0e0e0;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(30, 30, 46, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        h1 {
            color: #fff;
            margin-bottom: 10px;
            text-align: center;
            font-size: 2.5em;
            text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
            letter-spacing: 1px;
        }
        
        .subtitle {
            color: #b0b0b0;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .nav-links {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
            padding: 10px;
            background: rgba(20, 20, 35, 0.6);
            border-radius: 50px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .nav-link {
            color: #b0b0b0;
            text-decoration: none;
            font-weight: 500;
            padding: 8px 20px;
            border-radius: 30px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .nav-link:hover {
            color: #fff;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .nav-link::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .nav-link:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .form-group {
            margin-bottom: 25px;
            position: relative;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #d0d0d0;
            font-weight: 500;
            font-size: 0.95em;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        
        .form-group:hover label {
            color: #667eea;
            transform: translateX(5px);
        }
        
        input, select {
            width: 100%;
            padding: 14px;
            background: rgba(20, 20, 35, 0.8);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            font-size: 16px;
            color: #fff;
            transition: all 0.3s ease;
            outline: none;
        }
        
        input:hover, select:hover {
            border-color: #667eea;
            background: rgba(30, 30, 50, 0.9);
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
        }
        
        input:focus, select:focus {
            border-color: #764ba2;
            box-shadow: 0 0 30px rgba(118, 75, 162, 0.3);
            background: rgba(35, 35, 55, 0.95);
        }
        
        button {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
            letter-spacing: 1px;
            text-transform: uppercase;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.5);
        }
        
        button:active {
            transform: translateY(-1px) scale(0.98);
        }
        
        button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        button:hover::before {
            width: 400px;
            height: 400px;
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 16px;
            background: rgba(20, 20, 35, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
            display: none;
            animation: glowPulse 2s infinite;
        }
        
        .result.show {
            display: block;
            animation: slideIn 0.5s ease, glowPulse 2s infinite;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes glowPulse {
            0% {
                box-shadow: 0 0 5px rgba(102, 126, 234, 0.3);
            }
            50% {
                box-shadow: 0 0 20px rgba(102, 126, 234, 0.6);
            }
            100% {
                box-shadow: 0 0 5px rgba(102, 126, 234, 0.3);
            }
        }
        
        .recommendation {
            font-size: 48px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin: 20px 0;
            text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-5px);
            }
            100% {
                transform: translateY(0px);
            }
        }
        
        .tips {
            margin-top: 20px;
        }
        
        .tips h3 {
            color: #d0d0d0;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
            padding-bottom: 8px;
        }
        
        .tip-item {
            padding: 15px;
            margin: 10px 0;
            background: rgba(30, 30, 46, 0.6);
            border-radius: 10px;
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
            cursor: default;
            animation: fadeInTip 0.5s ease;
            animation-fill-mode: both;
        }
        
        .tip-item:hover {
            transform: translateX(10px) scale(1.02);
            background: rgba(40, 40, 60, 0.8);
            border-left-color: #764ba2;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
        }
        
        @keyframes fadeInTip {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .tip-item:nth-child(1) { animation-delay: 0.1s; }
        .tip-item:nth-child(2) { animation-delay: 0.2s; }
        .tip-item:nth-child(3) { animation-delay: 0.3s; }
        .tip-item:nth-child(4) { animation-delay: 0.4s; }
        .tip-item:nth-child(5) { animation-delay: 0.5s; }
        
        .loader {
            border: 5px solid rgba(255, 255, 255, 0.1);
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite, glow 1s ease-in-out infinite;
            margin: 30px auto;
            display: none;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes glow {
            0% { box-shadow: 0 0 5px #667eea; }
            50% { box-shadow: 0 0 30px #667eea; }
            100% { box-shadow: 0 0 5px #667eea; }
        }
        
        .error {
            color: #ff6b6b;
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 107, 107, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(255, 107, 107, 0.3);
            display: none;
            animation: shake 0.5s ease;
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
        
        .batch-upload {
            margin-top: 40px;
            padding: 25px;
            border-top: 2px solid rgba(255, 255, 255, 0.1);
            background: rgba(20, 20, 35, 0.4);
            border-radius: 16px;
            transition: all 0.3s ease;
        }
        
        .batch-upload:hover {
            background: rgba(25, 25, 40, 0.6);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .batch-upload h3 {
            margin-bottom: 15px;
            color: #d0d0d0;
            font-size: 1.4em;
        }
        
        .batch-upload p {
            color: #a0a0a0;
            margin-bottom: 15px;
            font-size: 0.95em;
        }
        
        .file-input {
            margin-bottom: 20px;
            position: relative;
        }
        
        .file-input input[type=file] {
            padding: 20px;
            background: rgba(20, 20, 35, 0.8);
            border: 2px dashed rgba(255, 255, 255, 0.2);
            cursor: pointer;
        }
        
        .file-input input[type=file]:hover {
            border-color: #667eea;
            background: rgba(30, 30, 50, 0.9);
        }
        
        .file-input input[type=file]::file-selector-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin-right: 15px;
            transition: all 0.3s ease;
        }
        
        .file-input input[type=file]::file-selector-button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .batch-button {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            box-shadow: 0 10px 20px rgba(40, 167, 69, 0.3);
        }
        
        .batch-button:hover {
            background: linear-gradient(135deg, #34ce57 0%, #2fd9a8 100%);
            box-shadow: 0 20px 40px rgba(40, 167, 69, 0.5);
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(20, 20, 35, 0.8);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        /* Selection styling */
        ::selection {
            background: rgba(102, 126, 234, 0.3);
            color: #fff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Study Time Recommender</h1>
        <p class="subtitle">Get personalized daily study time recommendations</p>
        
        <div class="nav-links">
            <a href="/" class="nav-link">🏠 Home</a>
            <a href="/model_info" class="nav-link" target="_blank">📊 Model Info</a>
        </div>
        
        <div class="form-group">
            <label for="gpa">📈 GPA (0-4.0)</label>
            <input type="number" id="gpa" step="0.1" min="0" max="4" value="3.0">
        </div>
        
        <div class="form-group">
            <label for="difficulty">🎯 Subject Difficulty (1-5)</label>
            <select id="difficulty">
                <option value="1">1 - Very Easy</option>
                <option value="2">2 - Easy</option>
                <option value="3" selected>3 - Moderate</option>
                <option value="4">4 - Hard</option>
                <option value="5">5 - Very Hard</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="past_performance">📊 Past Performance (0-100%)</label>
            <input type="number" id="past_performance" step="1" min="0" max="100" value="75">
        </div>
        
        <div class="form-group">
            <label for="available_hours">⏰ Available Hours Per Day</label>
            <input type="number" id="available_hours" step="0.5" min="0.5" max="24" value="4">
        </div>
        
        <button onclick="predict()">✨ Get Recommendation ✨</button>
        
        <div class="loader" id="loader"></div>
        
        <div class="error" id="error"></div>
        
        <div class="result" id="result">
            <h2>Your Personalized Recommendation</h2>
            <div class="recommendation" id="recommendedHours">0 hours/day</div>
            
            <div class="tips">
                <h3>💡 Study Tips Just For You</h3>
                <div id="tipsList"></div>
            </div>
        </div>
        
        <div class="batch-upload">
            <h3>📁 Batch Prediction</h3>
            <p>Upload a CSV file with columns: gpa, difficulty, past_performance, available_hours</p>
            <div class="file-input">
                <input type="file" id="fileInput" accept=".csv">
            </div>
            <button onclick="batchPredict()" class="batch-button">🚀 Process Batch</button>
        </div>
    </div>

    <script>
        async function predict() {
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result').classList.remove('show');
            document.getElementById('error').style.display = 'none';
            
            const data = {
                gpa: parseFloat(document.getElementById('gpa').value),
                difficulty: parseInt(document.getElementById('difficulty').value),
                past_performance: parseFloat(document.getElementById('past_performance').value),
                available_hours: parseFloat(document.getElementById('available_hours').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                document.getElementById('loader').style.display = 'none';
                
                if (result.success) {
                    document.getElementById('recommendedHours').textContent = 
                        `${result.recommended_hours} hours/day`;
                    
                    const tipsList = document.getElementById('tipsList');
                    tipsList.innerHTML = '';
                    result.tips.forEach(tip => {
                        const tipDiv = document.createElement('div');
                        tipDiv.className = 'tip-item';
                        tipDiv.textContent = tip;
                        tipsList.appendChild(tipDiv);
                    });
                    
                    document.getElementById('result').classList.add('show');
                } else {
                    document.getElementById('error').textContent = result.error;
                    document.getElementById('error').style.display = 'block';
                }
            } catch (error) {
                document.getElementById('loader').style.display = 'none';
                document.getElementById('error').textContent = 'Network error. Please try again.';
                document.getElementById('error').style.display = 'block';
            }
        }
        
        async function batchPredict() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a file first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('loader').style.display = 'block';
            
            try {
                const response = await fetch('/batch_predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                document.getElementById('loader').style.display = 'none';
                
                if (result.success) {
                    alert('✅ ' + result.message);
                    if (result.download_url) {
                        window.location.href = result.download_url;
                    }
                } else {
                    alert('❌ Error: ' + result.error);
                }
            } catch (error) {
                document.getElementById('loader').style.display = 'none';
                alert('❌ Network error. Please try again.');
            }
        }
    </script>
</body>
</html>
    """)

# Add this missing piece - the main block to run the app
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 STARTING STUDY TIME RECOMMENDER WEB APP")
    print("=" * 60)
    print("\n📱 Open your browser and go to: http://127.0.0.1:5000")
    print("⚠️  Press CTRL+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)