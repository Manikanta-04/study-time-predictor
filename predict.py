"""
Study Time Recommendation - Prediction Script
Use this to make predictions with your trained model
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

class StudyTimePredictor:
    def __init__(self, model_path='models/latest_model.pkl', scaler_path='models/scaler.pkl'):
        """
        Initialize the predictor with trained model and scaler
        """
        print("📚 Initializing Study Time Predictor...")
        
        # Check if model files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")
        
        # Load model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata if available
        if os.path.exists('models/model_metadata.csv'):
            self.metadata = pd.read_csv('models/model_metadata.csv').iloc[0]
            print(f"✅ Loaded model: {self.metadata['model_name']}")
            print(f"✅ R² Score: {self.metadata['r2_score']:.4f}")
        else:
            print("✅ Model and scaler loaded successfully")
        
        self.features = ['gpa', 'difficulty_level', 'past_performance', 'available_hours']
    
    def predict(self, gpa, difficulty, past_performance, available_hours):
        """
        Predict recommended study hours
        
        Parameters:
        - gpa: float (0-4.0)
        - difficulty: int (1-5)
        - past_performance: float (0-100)
        - available_hours: float (hours per day)
        
        Returns:
        - recommended_hours: float
        - confidence: str
        """
        
        # Validate inputs
        self._validate_inputs(gpa, difficulty, past_performance, available_hours)
        
        # Prepare input
        input_data = np.array([[gpa, difficulty, past_performance, available_hours]])
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        
        # Round to nearest 0.5 hour
        prediction = round(prediction * 2) / 2
        
        # Ensure within reasonable bounds
        prediction = max(0.5, min(available_hours, prediction))
        
        # Calculate confidence (simple heuristic)
        confidence = self._calculate_confidence(gpa, difficulty, past_performance)
        
        return prediction, confidence
    
    def predict_batch(self, students_data):
        """
        Predict for multiple students
        
        Parameters:
        - students_data: list of dictionaries with student info
        
        Returns:
        - DataFrame with predictions
        """
        results = []
        
        for student in students_data:
            pred, conf = self.predict(
                student['gpa'],
                student['difficulty'],
                student['past_performance'],
                student['available_hours']
            )
            
            results.append({
                'student_id': student.get('student_id', 'N/A'),
                'gpa': student['gpa'],
                'difficulty': student['difficulty'],
                'past_performance': student['past_performance'],
                'available_hours': student['available_hours'],
                'recommended_hours': pred,
                'confidence': conf
            })
        
        return pd.DataFrame(results)
    
    def _validate_inputs(self, gpa, difficulty, past_performance, available_hours):
        """Validate input ranges"""
        if not 0 <= gpa <= 4.0:
            raise ValueError("GPA must be between 0 and 4.0")
        if not 1 <= difficulty <= 5:
            raise ValueError("Difficulty level must be between 1 and 5")
        if not 0 <= past_performance <= 100:
            raise ValueError("Past performance must be between 0 and 100")
        if not 0.5 <= available_hours <= 24:
            raise ValueError("Available hours must be between 0.5 and 24")
    
    def _calculate_confidence(self, gpa, difficulty, past_performance):
        """Calculate prediction confidence"""
        # Simple confidence based on input values
        if (2.0 <= gpa <= 4.0 and 
            1 <= difficulty <= 5 and 
            50 <= past_performance <= 100):
            return "High"
        elif (1.0 <= gpa <= 4.0 and 
              1 <= difficulty <= 5 and 
              30 <= past_performance <= 100):
            return "Medium"
        else:
            return "Low"
    
    def get_study_tips(self, prediction, gpa, difficulty):
        """Generate personalized study tips"""
        tips = []
        
        if prediction >= 5:
            tips.append("📚 Consider breaking study sessions into 45-minute blocks with 10-minute breaks")
        
        if difficulty >= 4:
            tips.append("🎯 For difficult subjects, try active recall and practice problems")
        
        if gpa < 2.5:
            tips.append("📝 Focus on understanding core concepts before moving to advanced topics")
        
        if prediction > 0:
            tips.append("⏰ Use the Pomodoro technique: 25 min study, 5 min break")
        
        tips.append("✅ Review material within 24 hours of learning for better retention")
        
        return tips

# Command-line interface
def main():
    print("\n" + "=" * 60)
    print("STUDY TIME RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = StudyTimePredictor()
        
        while True:
            print("\n" + "-" * 40)
            print("1. Make a prediction")
            print("2. Batch predict from CSV")
            print("3. Exit")
            print("-" * 40)
            
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                # Single prediction
                print("\n📝 Enter student details:")
                
                try:
                    gpa = float(input("GPA (0-4.0): "))
                    difficulty = int(input("Subject difficulty (1-5): "))
                    past_perf = float(input("Past performance (0-100): "))
                    available = float(input("Available hours per day: "))
                    
                    prediction, confidence = predictor.predict(gpa, difficulty, past_perf, available)
                    tips = predictor.get_study_tips(prediction, gpa, difficulty)
                    
                    print("\n" + "=" * 40)
                    print(f"📊 RECOMMENDED STUDY TIME: {prediction} hours/day")
                    print(f"📈 Confidence: {confidence}")
                    print("=" * 40)
                    
                    print("\n💡 Study Tips:")
                    for tip in tips:
                        print(f"  {tip}")
                    
                except ValueError as e:
                    print(f"❌ Error: {e}")
            
            elif choice == '2':
                # Batch prediction
                csv_path = input("Enter path to CSV file: ").strip()
                
                try:
                    df = pd.read_csv(csv_path)
                    required_cols = ['gpa', 'difficulty', 'past_performance', 'available_hours']
                    
                    if not all(col in df.columns for col in required_cols):
                        print("❌ CSV must contain columns: gpa, difficulty, past_performance, available_hours")
                        continue
                    
                    students_data = df.to_dict('records')
                    results = predictor.predict_batch(students_data)
                    
                    # Save results
                    output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    results.to_csv(output_file, index=False)
                    
                    print(f"\n✅ Predictions saved to {output_file}")
                    print("\n📊 Sample predictions:")
                    print(results.head())
                    
                except Exception as e:
                    print(f"❌ Error processing file: {e}")
            
            elif choice == '3':
                print("\n👋 Thank you for using Study Time Recommender!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
    
    except Exception as e:  # Fixed indentation here
        print(f"\n❌ Error: {e}")
        print("Please make sure you've trained the model first by running train_model.py")

if __name__ == "__main__":
    main()