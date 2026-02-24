"""
Study Time Recommendation - Dataset Generator
Run this script in VS Code to generate and save your custom dataset
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 60)
print("STUDY TIME RECOMMENDATION - DATASET GENERATOR")
print("=" * 60)

# Configuration
NUM_SAMPLES = 2000  # Number of student records to generate
OUTPUT_FILE = "study_time_dataset.csv"

def generate_study_time_dataset(n_samples=2000):
    """
    Generate a realistic dataset for study time prediction
    """
    
    # Initialize empty lists for each feature
    data = {
        'student_id': [],
        'gpa': [],
        'difficulty_level': [],
        'past_performance': [],
        'available_hours': [],
        'study_hours_per_week': [],
        'num_courses': [],
        'sleep_hours': [],
        'extracurricular_hours': [],
        'part_time_job': [],
        'motivation_level': [],
        'attendance_percentage': [],
        'assignment_completion_rate': [],
        'recommended_study_hours': []  # Target variable
    }
    
    print(f"\nGenerating {n_samples} student records...")
    
    for i in range(n_samples):
        # Generate student ID
        student_id = f"STU{str(i+1).zfill(5)}"
        
        # Generate GPA (realistic distribution)
        # Most students between 2.5 and 3.8
        gpa = round(np.random.normal(3.0, 0.5), 2)
        gpa = max(0.0, min(4.0, gpa))  # Clip to valid range
        
        # Difficulty level (1-5)
        difficulty_level = random.randint(1, 5)
        
        # Past performance (0-100)
        past_performance = round(np.random.normal(75, 15), 1)
        past_performance = max(0, min(100, past_performance))
        
        # Available hours per day (1-10)
        available_hours = round(np.random.uniform(1, 10), 1)
        
        # Study hours per week (derived from available hours)
        study_hours_per_week = round(available_hours * random.uniform(3, 6), 1)
        
        # Number of courses (3-7)
        num_courses = random.randint(3, 7)
        
        # Sleep hours (4-9)
        sleep_hours = round(random.uniform(4, 9), 1)
        
        # Extracurricular hours (0-4 per day)
        extracurricular_hours = round(random.uniform(0, 4), 1)
        
        # Part-time job (0 = no, 1 = yes)
        part_time_job = random.choice([0, 1])
        
        # Motivation level (1-10)
        motivation_level = random.randint(1, 10)
        
        # Attendance percentage (40-100)
        attendance_percentage = round(random.uniform(40, 100), 1)
        
        # Assignment completion rate (50-100)
        assignment_completion_rate = round(random.uniform(50, 100), 1)
        
        # Calculate RECOMMENDED STUDY HOURS (Target Variable)
        # This is a formula-based approach to create realistic recommendations
        
        # Base hours (2-4 hours depending on courses)
        base_hours = 2 + (num_courses - 3) * 0.5
        
        # Adjustments
        if gpa < 2.5:
            gpa_factor = 1.5  # Need more study
        elif gpa < 3.0:
            gpa_factor = 1.0
        elif gpa < 3.5:
            gpa_factor = 0.5
        else:
            gpa_factor = 0.0  # Good students need less
        
        # Difficulty factor
        difficulty_factor = (difficulty_level - 3) * 0.3
        
        # Past performance factor
        if past_performance < 60:
            performance_factor = 1.0
        elif past_performance < 75:
            performance_factor = 0.5
        else:
            performance_factor = 0.0
        
        # Motivation factor (less motivated need more structured time)
        motivation_factor = (10 - motivation_level) * 0.1
        
        # Calculate recommended hours
        recommended = base_hours + gpa_factor + difficulty_factor + performance_factor + motivation_factor
        
        # Add some random noise to make it realistic
        noise = np.random.normal(0, 0.3)
        recommended += noise
        
        # Ensure within reasonable bounds (1-8 hours)
        recommended = max(1, min(8, recommended))
        
        # Consider available hours (can't recommend more than available)
        recommended = min(recommended, available_hours)
        
        # Round to nearest 0.5
        recommended = round(recommended * 2) / 2
        
        # Append all values
        data['student_id'].append(student_id)
        data['gpa'].append(gpa)
        data['difficulty_level'].append(difficulty_level)
        data['past_performance'].append(past_performance)
        data['available_hours'].append(available_hours)
        data['study_hours_per_week'].append(study_hours_per_week)
        data['num_courses'].append(num_courses)
        data['sleep_hours'].append(sleep_hours)
        data['extracurricular_hours'].append(extracurricular_hours)
        data['part_time_job'].append(part_time_job)
        data['motivation_level'].append(motivation_level)
        data['attendance_percentage'].append(attendance_percentage)
        data['assignment_completion_rate'].append(assignment_completion_rate)
        data['recommended_study_hours'].append(recommended)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def add_realistic_correlations(df):
    """
    Add realistic correlations between variables
    """
    # GPA correlation with past performance
    df['past_performance'] = df.apply(
        lambda row: row['past_performance'] * (0.7 + 0.1 * row['gpa']), 
        axis=1
    )
    
    # Recommended hours correlation with available hours
    df['recommended_study_hours'] = df.apply(
        lambda row: min(row['recommended_study_hours'], row['available_hours'] * 0.9),
        axis=1
    )
    
    return df

def save_dataset(df, filename):
    """
    Save dataset to CSV file
    """
    # Create datasets directory if it doesn't exist
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    filepath = os.path.join('datasets', filename)
    df.to_csv(filepath, index=False)
    print(f"\n✅ Dataset saved successfully to: {filepath}")
    
    # Also save a copy in current directory
    df.to_csv(filename, index=False)
    print(f"✅ Dataset also saved to current directory: {filename}")
    
    return filepath

def display_dataset_info(df):
    """
    Display information about the generated dataset
    """
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Column Names:")
    for col in df.columns:
        print(f"   - {col}")
    
    print("\n📈 Data Types:")
    print(df.dtypes)
    
    print("\n🔍 First 5 Records:")
    print(df.head())
    
    print("\n📊 Statistical Summary:")
    print(df.describe())
    
    print("\n🎯 Target Variable Distribution (Recommended Study Hours):")
    print(df['recommended_study_hours'].value_counts().sort_index().head(10))
    
    # Check for missing values
    print("\n⚠️ Missing Values:")
    print(df.isnull().sum())

def create_data_dictionary():
    """
    Create a data dictionary explaining all features
    """
    data_dict = {
        'Feature': [
            'student_id',
            'gpa',
            'difficulty_level',
            'past_performance',
            'available_hours',
            'study_hours_per_week',
            'num_courses',
            'sleep_hours',
            'extracurricular_hours',
            'part_time_job',
            'motivation_level',
            'attendance_percentage',
            'assignment_completion_rate',
            'recommended_study_hours'
        ],
        'Description': [
            'Unique student identifier',
            'Grade Point Average (0-4.0 scale)',
            'Subject difficulty level (1-5, where 5 is hardest)',
            'Past academic performance percentage (0-100)',
            'Hours available for study per day',
            'Actual study hours per week',
            'Number of courses enrolled',
            'Average sleep hours per day',
            'Hours spent on extracurricular activities',
            'Whether student has part-time job (0=No, 1=Yes)',
            'Self-reported motivation level (1-10)',
            'Class attendance percentage',
            'Rate of assignment completion',
            'RECOMMENDED study hours per day (TARGET)'
        ],
        'Data Type': [
            'string',
            'float',
            'integer',
            'float',
            'float',
            'float',
            'integer',
            'float',
            'float',
            'integer',
            'integer',
            'float',
            'float',
            'float'
        ]
    }
    
    dict_df = pd.DataFrame(data_dict)
    return dict_df

# Main execution
if __name__ == "__main__":
    print("\n🚀 Starting dataset generation process...")
    
    # Generate the dataset
    df = generate_study_time_dataset(NUM_SAMPLES)
    
    # Add realistic correlations
    df = add_realistic_correlations(df)
    
    # Display dataset information
    display_dataset_info(df)
    
    # Save the dataset
    filename = f"study_time_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = save_dataset(df, filename)
    
    # Create and save data dictionary
    dict_df = create_data_dictionary()
    dict_df.to_csv('datasets/data_dictionary.csv', index=False)
    print("\n📚 Data dictionary saved to: datasets/data_dictionary.csv")
    
    print("\n" + "=" * 60)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print("\n📁 Files created:")
    print(f"   1. {filename} - Main dataset")
    print(f"   2. datasets/data_dictionary.csv - Data dictionary")
    print(f"   3. study_time_dataset.csv - Copy in current directory")
    
    print("\n💡 Next Steps:")
    print("   1. Load this dataset in your ML model")
    print("   2. Use 'pd.read_csv(\"study_time_dataset.csv\")' to load")
    print("   3. Features: gpa, difficulty_level, past_performance, available_hours")
    print("   4. Target: recommended_study_hours")
    
    # Sample code to load the dataset
    print("\n📝 Sample code to load the dataset:")
    print("""
import pandas as pd

# Load the dataset
df = pd.read_csv('study_time_dataset.csv')

# Prepare features and target
features = ['gpa', 'difficulty_level', 'past_performance', 'available_hours']
X = df[features]
y = df['recommended_study_hours']

print("Dataset loaded successfully!")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
    """)