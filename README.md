# 🎓 SmartGrade (ASPIRE)
> **Academic & Social Pattern Integrated Result Engine**  
> An intelligent, machine-learning-powered web platform for predicting student performance, calculating SGPA/CGPA, analyzing grade trends, and delivering personalized study recommendations.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
  - [Student Features](#student-features)
  - [Administrator Features](#administrator-features)
- [Machine Learning Engine](#-machine-learning-engine)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started & Setup](#-getting-started--setup)
  - [Prerequisites](#1-prerequisites)
  - [Installation & Environment Setup](#2-installation--environment-setup)
  - [Training the ML Model](#3-training-the-ml-model)
  - [Running the Backend API](#4-running-the-backend-api)
  - [Launching the Frontend UI](#5-launching-the-frontend-ui)
- [Demo Credentials](#-demo-credentials)
- [API Reference Summary](#-api-reference-summary)
- [Grading System Matrix](#-grading-system-matrix)
- [License](#-license)

---

## 🌟 Overview

**SmartGrade (ASPIRE)** is a full-stack academic analytics and performance prediction engine built for students and educational administrators. Built with Ramdeobaba University (RBU) evaluation guidelines in mind, SmartGrade leverages machine learning to forecast end-semester exam marks based on continuous academic inputs, study habits, attendance, and subject difficulty.

Instead of reactive grade tracking, SmartGrade empowers students to proactively target areas for improvement, calculate real-time SGPA/CGPA, simulate Target CGPA trajectories, and view personalized study guidance. For institutions, SmartGrade provides administrative insights into academic distribution, department performance, and early identification of at-risk students.

---

## ✨ Key Features

### 👨‍🎓 Student Features

* **🤖 ML-Powered End-Sem Mark Predictor**: Predicts end-semester exam marks (out of 50) using trained Random Forest models based on 5 core academic indicators.
* **🧮 Live SGPA & CGPA Grade Calculator**: Real-time calculation of SGPA with automatic credit point weighting based on the RBU 10-point grading scale (`AA` to `FF`).
* **🔄 Seamless Predictor-to-Calculator Integration**: Auto-fills predicted marks directly into the Grade Calculator with visual badges indicating ML estimates.
* **📊 Comprehensive Analytics Dashboard**:
  * Multi-semester SGPA progression line charts.
  * Subject performance heatmaps and grade distribution breakdown charts.
  * Internal vs. ML-predicted score comparison.
  * **Target CGPA Simulator**: Interactive tool to set a goal CGPA and calculate the required average SGPA across remaining semesters.
* **📚 AI-Powered Study Guidance**: Automated subject-level focus recommendation engine categorizing subjects into *High*, *Medium*, or *Low* priority based on grade risk.
* **💡 Syllabus Topic Suggestions**: High-yield topic recommendations tailored to subject area and current academic standing.
* **🏆 Department Leaderboards**: Department-wise and overall student rankings with opt-in/opt-out privacy controls.
* **📋 Historical Record Tracking**: View all saved past semesters with dynamic cumulative CGPA recalculations.

### 🛡️ Administrator Features

* **📊 Institution-Wide Analytics**: Overview of student counts, average CGPA across all departments, distinction ratios, and at-risk student tracking (CGPA < 5.0).
* **👥 Student Search & Directory**: Searchable directory with filtering by department (CSE, IT, ECE, MECH, CIVIL) and classification (*Distinction*, *First Class*, *Pass*, *At Risk*).
* **🤖 ML Pipeline & Model Monitoring**: Metrics dashboard showing model evaluation stats ($R^2$, RMSE, Cross-Validation, Feature Importance).

---

## 🧠 Machine Learning Engine

SmartGrade features a standalone data science training pipeline (`ml/train.py`) trained on student academic records.

### Trained Models:
1. **Random Forest Regressor** (`rf.pkl`): Primary regression model for predicting exact end-semester marks ($0 - 50$).
2. **Decision Tree Classifier** (`dt.pkl`): Risk assessment classification model (*At Risk*, *On Track*, *Good*).
3. **Linear Regression**: Baseline regression model for performance benchmark comparison.

### Key Metrics Summary:
| Metric | Random Forest Regressor | Linear Regression | Decision Tree Classifier |
| :--- | :---: | :---: | :---: |
| **$R^2$ Score** | `0.8117` | `0.8581` | — |
| **RMSE** | `3.1954` | `2.7740` | — |
| **Accuracy** | — | — | `73.33%` |
| **5-Fold CV Mean $R^2$** | `0.7424 ± 0.0424` | — | — |

### Feature Importance Weights:
* **Internal Marks (out of 50)**: `48.66%`
* **Daily Study Hours**: `19.74%`
* **Subject Difficulty**: `12.26%`
* **Attendance Percentage**: `10.13%`
* **Days Before Exam**: `9.21%`

---

## 🛠️ Tech Stack

* **Backend Framework**: Python 3.x, Flask, Flask-SQLAlchemy (SQLite ORM), Flask-CORS
* **Authentication**: JSON Web Tokens (PyJWT), Bcrypt password hashing
* **Machine Learning & Data Science**: Scikit-Learn, Joblib, NumPy, Pandas
* **Frontend**: HTML5, Vanilla CSS3 (Custom Design System, Flexbox, CSS Grid), Vanilla JavaScript ES6+
* **Data Visualization**: Chart.js v4.4
* **Database**: SQLite3 (`aspire.db`)

---

## 📁 Project Directory Structure

```
SmartGrade/
└── Backend/
    ├── app.py              # Flask API application server & routing logic
    ├── aspire.db           # SQLite database storing users, semesters & predictions
    ├── ml/
    │   ├── train.py        # ML dataset generator & model training pipeline script
    │   ├── dataset.csv     # Generated dataset (300 student records)
    │   ├── rf.pkl          # Serialized Random Forest Regressor model
    │   ├── dt.pkl          # Serialized Decision Tree Classifier model
    │   └── metrics.json    # Model evaluation metrics & feature importances
    └── Frontend/
        └── index.html      # Responsive Single-Page Web Application UI & Client Logic
```

---

## 🚀 Getting Started & Setup

### 1. Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 2. Installation & Environment Setup

Navigate to the project directory:
```bash
cd SmartGrade/Backend
```

Install the required Python dependencies:
```bash
pip install flask flask-cors flask-sqlalchemy scikit-learn joblib pandas numpy pyjwt bcrypt
```

### 3. Training the ML Model
Before running the server for the first time, execute the ML training pipeline to generate `rf.pkl`, `dt.pkl`, `metrics.json`, and `dataset.csv`:

```bash
python ml/train.py
```

### 4. Running the Backend API
Start the Flask development backend server (runs on `http://127.0.0.1:5000`):

```bash
python app.py
```

*Note: On initial execution, `app.py` automatically initializes the database schema and seeds the default administrator account.*

### 5. Launching the Frontend UI
Simply open `Backend/Frontend/index.html` in any modern web browser, or serve it using a local static file server:

```bash
# Optional: Using Python built-in HTTP server
python -m http.server 8000 --directory Frontend
```
Then navigate to `http://localhost:8000` in your browser.

---

## 🔑 Demo Credentials

### Administrator Account:
* **Email**: `admin@rbu.ac.in`
* **Password**: `admin123`

### Student Account:
You can self-register a new student account directly from the login/register screen in the web app interface.

---

## 🔗 API Reference Summary

| Endpoint | Method | Access | Description |
| :--- | :---: | :---: | :--- |
| `/api/register` | `POST` | Public | Register a new student account |
| `/api/login` | `POST` | Public | Authenticate user & return JWT token |
| `/api/me` | `GET` | Authenticated | Retrieve current user profile & CGPA details |
| `/api/semesters` | `GET` / `POST` | Authenticated | Get or save semester marks and calculate SGPA |
| `/api/semesters/<id>` | `DELETE` | Authenticated | Remove a saved semester record |
| `/api/predict` | `POST` | Authenticated | Predict end-sem marks & assess academic risk |
| `/api/predictions` | `GET` | Authenticated | List all past predictions for the user |
| `/api/leaderboard` | `GET` | Authenticated | Retrieve student leaderboard (opt-in only) |
| `/api/leaderboard/toppers`| `GET` | Authenticated | Retrieve top student per department |
| `/api/leaderboard/opt` | `POST` | Authenticated | Toggle user opt-in status on leaderboard |
| `/api/admin/students` | `GET` | Admin | List all registered students with CGPA |
| `/api/admin/analytics` | `GET` | Admin | Overall institution analytics & ML metrics |

---

## 📊 Grading System Matrix

SmartGrade implements the Ramdeobaba University (RBU) credit-based grade point system:

| Total Marks (out of 100) | Grade | Grade Points ($GP$) | Description |
| :---: | :---: | :---: | :--- |
| $90 - 100$ | **AA** | `10.0` | Outstanding |
| $80 - 89$ | **AB** | `9.0` | Excellent |
| $70 - 79$ | **BB** | `8.0` | Very Good |
| $60 - 69$ | **BC** | `7.0` | Good |
| $50 - 59$ | **CC** | `6.0` | Above Average |
| $40 - 49$ | **CD** | `5.0` | Average |
| $30 - 39$ | **DD** | `4.0` | Below Average / Pass |
| $< 30$ | **FF** | `0.0` | Fail |

$$\text{SGPA} = \frac{\sum (\text{Course Credits} \times \text{Grade Points})}{\sum \text{Course Credits}}$$

---

## 📜 License

This project is developed for educational and academic tracking purposes.