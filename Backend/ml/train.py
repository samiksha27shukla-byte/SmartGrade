"""
ASPIRE - ML Training Script
Run ONCE before starting app.py
Outputs: rf.pkl, dt.pkl, metrics.json, dataset.csv
"""
import pandas as pd, numpy as np, json, joblib, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

np.random.seed(42)
print("Generating 300 student records...")

subjects = ['Data Structures','DBMS','Operating Systems','Computer Networks','DAA',
            'OOP Java','Software Engineering','Discrete Mathematics','Web Technologies',
            'Python Programming','Maths I','Maths II','Physics','C Programming']
depts = ['CSE','IT','ECE','MECH','CIVIL']

rows = []
for i in range(300):
    internal = np.random.uniform(20, 50)
    att      = np.random.uniform(50, 100)
    study    = np.random.uniform(0.5, 8)
    diff     = np.random.choice([1, 2, 3])
    days     = np.random.uniform(0, 60)
    dept     = np.random.choice(depts)
    subj     = np.random.choice(subjects)
    year     = np.random.choice([1, 2, 3, 4])
    has_lab  = np.random.choice([True, False])
    df2      = {1:1.05, 2:0.95, 3:0.85}[diff]
    raw      = ((internal/50)*0.50+(att/100)*0.18+min(study/6,1)*0.22+min(days/30,1)*0.10)*50*df2+np.random.normal(0,2.5)
    end_sem  = round(float(np.clip(raw, 0, 50)), 1)
    lab_val  = round(float(np.clip(np.random.uniform(30,50)+np.random.normal(0,3), 0, 50)), 1) if has_lab else None
    rows.append({
        'student_id':i+1,'department':dept,'year':year,'subject':subj,
        'has_lab':has_lab,'difficulty':diff,'internal_marks':round(internal,1),
        'lab_marks':lab_val,'attendance_pct':round(att,1),
        'study_hrs_day':round(study,1),'days_before_exam':int(days),'end_sem_marks':end_sem
    })

df = pd.DataFrame(rows)
OUT = os.path.dirname(os.path.abspath(__file__))
df.to_csv(os.path.join(OUT,'dataset.csv'), index=False)
print(f"Dataset: {len(df)} rows saved as dataset.csv")

FEATURES = ['internal_marks','attendance_pct','study_hrs_day','days_before_exam','difficulty']
X,y = df[FEATURES],df['end_sem_marks']
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

print("Training Random Forest Regressor...")
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(Xtr,ytr); yp=rf.predict(Xte)
rf_r2=round(r2_score(yte,yp),4); rf_rmse=round(float(np.sqrt(mean_squared_error(yte,yp))),4)

print("Training Linear Regression...")
lr=LinearRegression(); lr.fit(Xtr,ytr); ylp=lr.predict(Xte)
lr_r2=round(r2_score(yte,ylp),4); lr_rmse=round(float(np.sqrt(mean_squared_error(yte,ylp))),4)

print("Training Decision Tree Classifier...")
df['risk']=df['end_sem_marks'].apply(lambda v:0 if v<25 else 1 if v<35 else 2)
Xrtr,Xrte,yrtr,yrte=train_test_split(df[FEATURES],df['risk'],test_size=0.2,random_state=42)
dt=DecisionTreeClassifier(max_depth=5,random_state=42); dt.fit(Xrtr,yrtr)
dt_acc=round(accuracy_score(yrte,dt.predict(Xrte)),4)

print("5-fold cross validation...")
cv=cross_val_score(rf,X,y,cv=5,scoring='r2')
fi=dict(zip(FEATURES,rf.feature_importances_))

metrics={
    'rf_r2':rf_r2,'rf_rmse':rf_rmse,'lr_r2':lr_r2,'lr_rmse':lr_rmse,'dt_acc':dt_acc,
    'cv_mean':round(float(cv.mean()),4),'cv_std':round(float(cv.std()),4),
    'feature_importance':{k:round(float(v),4) for k,v in sorted(fi.items(),key=lambda x:-x[1])},
    'features':FEATURES,'train_size':len(Xtr),'test_size':len(Xte),'total':len(df)
}
joblib.dump(rf,os.path.join(OUT,'rf.pkl'))
joblib.dump(dt,os.path.join(OUT,'dt.pkl'))
with open(os.path.join(OUT,'metrics.json'),'w') as f: json.dump(metrics,f,indent=2)

print(f"\n=== RESULTS ===")
print(f"RF  R2={rf_r2}  RMSE={rf_rmse}")
print(f"LR  R2={lr_r2}  RMSE={lr_rmse}")
print(f"DT  Accuracy={dt_acc}")
print(f"CV  R2={metrics['cv_mean']} +/-{metrics['cv_std']}")
print(f"\nFeature Importance:")
for k,v in metrics['feature_importance'].items():
    print(f"  {'x'*int(v*40)} {k} ({v*100:.1f}%)")
print("\nFiles saved: rf.pkl  dt.pkl  metrics.json  dataset.csv")
print("Now run:  python app.py")