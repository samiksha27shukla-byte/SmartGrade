"""
ASPIRE - ML training script
===========================
Predicts END-SEM marks (out of 50) from 5 academic features.

Usage
-----
  python ml/train.py                      # trains on SIMULATED data (default)
  python ml/train.py --csv real_data.csv  # trains on REAL collected data

A real CSV must contain these columns (see real_data_template.csv):
  internal_marks, attendance_pct, study_hrs_day, days_before_exam, difficulty, end_sem_marks

Outputs (written next to this file):
  model.pkl         - the model selected by 5-fold cross-validation
  risk_model.pkl    - decision-tree risk classifier (At Risk / On Track / Good)
  metrics.json      - evaluation results shown in the app
  dataset_synthetic.csv - only when simulated data is used
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['internal_marks', 'attendance_pct', 'study_hrs_day',
            'days_before_exam', 'difficulty']
TARGET = 'end_sem_marks'

SUBJECTS = ['Data Structures', 'DBMS', 'Operating Systems', 'Computer Networks', 'DAA',
            'OOP Java', 'Software Engineering', 'Discrete Mathematics', 'Web Technologies',
            'Python Programming', 'Maths I', 'Maths II', 'Physics', 'C Programming']
DEPTS = ['CSE', 'IT', 'ECE', 'MECH', 'CIVIL']


def make_synthetic(n=300, seed=42):
    """SIMULATED dataset. End-sem marks are generated from a hand-written formula
    plus noise, so a model trained on it only learns that formula."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        internal = rng.uniform(20, 50)
        att = rng.uniform(50, 100)
        study = rng.uniform(0.5, 8)
        diff = int(rng.choice([1, 2, 3]))
        days = int(rng.uniform(0, 60))
        factor = {1: 1.05, 2: 0.95, 3: 0.85}[diff]
        raw = ((internal / 50) * 0.50 + (att / 100) * 0.18 +
               min(study / 6, 1) * 0.22 + min(days / 30, 1) * 0.10) * 50 * factor
        end_sem = float(np.clip(raw + rng.normal(0, 2.5), 0, 50))
        rows.append({
            'student_id': i + 1,
            'department': str(rng.choice(DEPTS)),
            'year': int(rng.choice([1, 2, 3, 4])),
            'subject': str(rng.choice(SUBJECTS)),
            'difficulty': diff,
            'internal_marks': round(internal, 1),
            'attendance_pct': round(att, 1),
            'study_hrs_day': round(study, 1),
            'days_before_exam': days,
            'end_sem_marks': round(end_sem, 1),
        })
    return pd.DataFrame(rows)


def load_data(csv_path):
    if csv_path:
        df = pd.read_csv(csv_path)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            sys.exit(f'CSV is missing columns: {missing}')
        df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors='coerce').dropna()
        df = df[df['internal_marks'].between(0, 50) & df[TARGET].between(0, 50) &
                df['attendance_pct'].between(0, 100) & df['difficulty'].isin([1, 2, 3])]
        if len(df) < 30:
            sys.exit(f'Only {len(df)} valid rows found - need at least 30 to train.')
        return df.reset_index(drop=True), 'real'
    df = make_synthetic()
    df.to_csv(os.path.join(HERE, 'dataset_synthetic.csv'), index=False)
    return df, 'synthetic'


def risk_label(internal, end_sem):
    total = internal + end_sem
    return 0 if total < 50 else (1 if total < 70 else 2)   # 0 At Risk, 1 On Track, 2 Good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', help='path to a CSV of REAL student records')
    args = ap.parse_args()

    df, dtype = load_data(args.csv)
    print(f'Dataset: {len(df)} rows ({dtype})')

    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
        'Linear Regression': LinearRegression(),
    }
    results = {}
    for name, mdl in candidates.items():
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        cv = cross_val_score(clone(mdl), X, y, cv=5, scoring='r2')
        results[name] = {
            'test_r2': round(float(r2_score(yte, pred)), 4),
            'test_rmse': round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
            'cv_r2_mean': round(float(cv.mean()), 4),
            'cv_r2_std': round(float(cv.std()), 4),
        }
        print(f"  {name:<18} test R2={results[name]['test_r2']}  RMSE={results[name]['test_rmse']}  "
              f"CV R2={results[name]['cv_r2_mean']}+/-{results[name]['cv_r2_std']}")

    best = max(results, key=lambda k: results[k]['cv_r2_mean'])
    print(f'Selected model (best 5-fold CV R2): {best}')

    pi = permutation_importance(candidates[best], Xte, yte, n_repeats=20,
                                random_state=42, scoring='r2')
    imp = np.clip(pi.importances_mean, 0, None)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    importance = {f: round(float(v), 4) for f, v in
                  sorted(zip(FEATURES, imp), key=lambda t: -t[1])}

    final = clone(candidates[best]).fit(X, y)          # refit on all data for deployment

    # Risk classifier (decision tree) on the final total = internal + end sem
    risk = np.array([risk_label(i, e) for i, e in zip(df['internal_marks'], df[TARGET])])
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, risk, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xr_tr, yr_tr)
    dt_acc = round(float(accuracy_score(yr_te, dt.predict(Xr_te))), 4)
    risk_final = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, risk)
    print(f'Risk classifier (Decision Tree) accuracy: {dt_acc}')

    joblib.dump(final, os.path.join(HERE, 'model.pkl'))
    joblib.dump(risk_final, os.path.join(HERE, 'risk_model.pkl'))

    metrics = {
        'dataset_type': dtype,
        'n_total': int(len(df)), 'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
        'features': FEATURES,
        'feature_ranges': {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES},
        'selected_model': best,
        'models': results,
        'risk_classifier': {'name': 'Decision Tree', 'accuracy': dt_acc,
                            'classes': ['At Risk (<50)', 'On Track (50-69)', 'Good (>=70)']},
        'feature_importance': importance,
    }
    with open(os.path.join(HERE, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\nPermutation importance of selected model:')
    for k, v in importance.items():
        print(f"  {'#' * int(v * 40):<40} {k} ({v * 100:.1f}%)")
    print('\nSaved: model.pkl, risk_model.pkl, metrics.json')
    if dtype == 'synthetic':
        print('NOTE: trained on SIMULATED data. Collect real records and run: '
              'python ml/train.py --csv real_data.csv')


if __name__ == '__main__':
    main()
