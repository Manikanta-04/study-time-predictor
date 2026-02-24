# 📚 Study Time Recommender

> A machine learning application that helps students optimize their study schedules through personalized, data-driven recommendations.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://study-time-predictor-manikanta.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Individual Prediction** | Personalized recommendations based on GPA, difficulty, and available time |
| **Batch Processing** | Upload CSV files to process multiple students at once |
| **Study Tips** | Custom strategies generated from your academic profile |
| **Visualizations** | Interactive charts for data patterns and model performance |
| **Fallback Mode** | Rule-based predictions when ML model is unavailable |

---

## 📊 Model Performance

The app uses a **Gradient Boosting Regressor** with the following metrics:

| Metric | Value |
|---|---|
| R² Score | **0.8094** (81% variance explained) |
| RMSE | 0.6114 hours |
| MAE | 0.4117 hours |
| Cross-validation Score | 0.7986 |

### Feature Importance

```
Available Hours     ████████████████████  84.6%
Past Performance    ██                     5.7%
Difficulty Level    ██                     5.0%
GPA                 ██                     4.8%
```

---

## 🛠️ Tech Stack

`Python 3.12` · `Streamlit` · `Scikit-learn` · `Pandas` · `NumPy` · `Plotly` · `Matplotlib` · `Joblib`

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/manikanta-04/study-time-predictor.git
cd study-time-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Generate dataset & train model
python generate_dataset.py
python train_model.py

# 4. Run the app
streamlit run streamlit_app.py
```

---

## 💡 Usage

### Individual Prediction
1. Open the **Individual Prediction** tab
2. Enter your GPA, subject difficulty, past performance, and available hours
3. Click **Get Recommendation**

### Batch Prediction
1. Open the **Batch Prediction** tab
2. Upload a CSV with the following columns:

```csv
gpa,difficulty,past_performance,available_hours
3.8,2,85,5
2.5,4,60,3
3.2,5,70,2
```

3. Download the results with recommended study hours

---

## 📁 Project Structure

```
study-time-predictor/
├── models/
│   ├── latest_model.pkl          # Best performing model
│   ├── gradient_boosting_model.pkl
│   ├── scaler.pkl                # Feature scaler
│   ├── model_metadata.csv        # Performance metrics
│   ├── feature_importance.png
│   └── correlation_matrix.png
├── datasets/                     # Generated datasets
├── .streamlit/                   # Streamlit config
├── streamlit_app.py              # Main application
├── train_model.py                # Model training
├── generate_dataset.py           # Dataset generation
├── predict.py                    # CLI prediction tool
├── requirements.txt
└── README.md
```

---

## ⚙️ Model Details

```python
Best Model: Gradient Boosting Regressor
Parameters: {
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_split': 5,
    'n_estimators': 50
}
Training samples: 1600
Test samples:     400
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| Model not found | Run `python train_model.py` first |
| Import errors | Run `pip install -r requirements.txt` |
| Large file upload fails | Increase `maxUploadSize` in `.streamlit/config.toml` |
| Slow predictions | Use batch mode for multiple students |

---

## 🤝 Contributing

1. Fork the repository
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👨‍💻 Author

**Manikanta** — [@manikanta-04](https://github.com/manikanta-04)

🔗 **Live App**: [study-time-predictor-manikanta.streamlit.app](https://study-time-predictor-manikanta.streamlit.app)  
📦 **Repository**: [github.com/manikanta-04/study-time-predictor](https://github.com/manikanta-04/study-time-predictor)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with ❤️ using Streamlit & Scikit-learn</p>
