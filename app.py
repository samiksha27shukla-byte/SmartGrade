"""
ASPIRE backend (Flask + SQLite + scikit-learn)
Run:  python app.py      then open http://127.0.0.1:5000
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from functools import wraps

import bcrypt
import jwt
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text

BASE = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE, 'ml')

app = Flask(__name__)
CORS(app, resources={r'/api/*': {'origins': '*',
                                 'allow_headers': ['Content-Type', 'Authorization'],
                                 'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']}})

DB_PATH = os.environ.get('ASPIRE_DB_PATH', os.path.join(BASE, 'aspire.db')).replace('\\', '/')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
SECRET = os.environ.get('ASPIRE_SECRET_KEY', 'aspire-dev-secret-change-this-before-deploying-0123456789')
db = SQLAlchemy(app)

# ───────────────────────── constants ─────────────────────────
GRADE_PTS = {'AA': 10, 'AB': 9, 'BB': 8, 'BC': 7, 'CC': 6, 'CD': 5, 'DD': 4, 'FF': 0}
THRESH = {'AA': 90, 'AB': 80, 'BB': 70, 'BC': 60, 'CC': 50, 'CD': 40, 'DD': 30}
DEPARTMENTS = ['CSE', 'IT', 'ECE', 'MECH', 'CIVIL']
SECURITY_QUESTIONS = [
    'What was the name of your first school?',
    "What is your favourite teacher's name?",
    'What is the name of your childhood best friend?',
    'In which city were you born?',
    'What was the name of your first pet?',
]
RISK_LABELS = {0: 'At Risk', 1: 'On Track', 2: 'Good'}


# ───────────────────────── model loading ─────────────────────────
def ensure_model():
    need = [os.path.join(ML_DIR, f) for f in ('model.pkl', 'risk_model.pkl', 'metrics.json')]
    if all(os.path.exists(p) for p in need):
        return
    print('Model files not found - training now (first run only)...')
    subprocess.run([sys.executable, os.path.join(ML_DIR, 'train.py')], check=True)


ensure_model()
model = joblib.load(os.path.join(ML_DIR, 'model.pkl'))
risk_model = joblib.load(os.path.join(ML_DIR, 'risk_model.pkl'))
with open(os.path.join(ML_DIR, 'metrics.json')) as _f:
    META = json.load(_f)
FEATURES = META['features']
RANGES = META.get('feature_ranges', {})
FEATURE_LABELS = {'internal_marks': 'Internal marks', 'attendance_pct': 'Attendance',
                  'study_hrs_day': 'Study hours/day', 'days_before_exam': 'Days before exam',
                  'difficulty': 'Difficulty'}


def model_predict(rows):
    X = pd.DataFrame(rows, columns=FEATURES)
    return np.clip(model.predict(X), 0, 50)


# ───────────────────────── database models ─────────────────────────
def now():
    return datetime.now(timezone.utc)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), default='student')
    department = db.Column(db.String(50), default='CSE')
    year = db.Column(db.Integer, default=1)
    roll_no = db.Column(db.String(20), default='')
    show_lb = db.Column(db.Boolean, default=True)
    security_q = db.Column(db.Integer, nullable=True)
    security_a = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=now)


class Semester(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sem_no = db.Column(db.Integer, nullable=False)
    sgpa = db.Column(db.Float, nullable=False)
    credits = db.Column(db.Integer, nullable=False)
    saved_at = db.Column(db.DateTime, default=now)
    subjects = db.relationship('Subject', backref='semester',
                               cascade='all, delete-orphan', lazy=True)


class Subject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sem_id = db.Column(db.Integer, db.ForeignKey('semester.id'), nullable=False)
    name = db.Column(db.String(120))
    credits = db.Column(db.Integer)
    kind = db.Column(db.String(10), default='theory')     # theory | lab
    internal = db.Column(db.Float)                         # for labs: lab marks /50
    end_sem = db.Column(db.Float)
    is_predicted = db.Column(db.Boolean, default=False)
    total = db.Column(db.Float)                            # out of 100
    grade = db.Column(db.String(5))
    gp = db.Column(db.Float)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    subject = db.Column(db.String(120))
    credits = db.Column(db.Integer, default=3)
    internal = db.Column(db.Float)
    attendance = db.Column(db.Float)
    study_hrs = db.Column(db.Float)
    days_left = db.Column(db.Integer)
    difficulty = db.Column(db.Integer)
    pred_end = db.Column(db.Float)
    grade = db.Column(db.String(5))
    risk = db.Column(db.String(20))
    saved_at = db.Column(db.DateTime, default=now)


# Adds columns to an existing aspire.db so you never have to delete it again.
MIGRATIONS = {
    'user': [('security_q', 'INTEGER'), ('security_a', 'VARCHAR(200)')],
    'subject': [('kind', "VARCHAR(10) DEFAULT 'theory'")],
}


def ensure_columns():
    insp = inspect(db.engine)
    for table, cols in MIGRATIONS.items():
        if not insp.has_table(table):
            continue
        existing = {c['name'] for c in insp.get_columns(table)}
        for name, ddl in cols:
            if name not in existing:
                with db.engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN {name} {ddl}'))
                print(f'Migrated: added {table}.{name}')


# ───────────────────────── helpers ─────────────────────────
class ApiError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.msg, self.code = msg, code


@app.errorhandler(ApiError)
def _api_error(e):
    return jsonify({'error': e.msg}), e.code


def get_grade(total):
    for g, t in (('AA', 90), ('AB', 80), ('BB', 70), ('BC', 60), ('CC', 50), ('CD', 40), ('DD', 30)):
        if total >= t:
            return g
    return 'FF'


def classify(c):
    if c >= 9: return 'Distinction'
    if c >= 7: return 'First Class'
    if c >= 5: return 'Pass'
    if c > 0: return 'At Risk'
    return 'No Data'


def num(d, key, lo, hi, default=None, required=False):
    v = d.get(key)
    if v is None or v == '':
        if required:
            raise ApiError(f'{key} is required')
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ApiError(f'{key} must be a number')
    if f != f or f < lo or f > hi:
        raise ApiError(f'{key} must be between {lo:g} and {hi:g}')
    return f


def hash_text(s):
    return bcrypt.hashpw(s.encode(), bcrypt.gensalt()).decode()


def check_text(s, h):
    try:
        return bcrypt.checkpw(s.encode(), h.encode())
    except Exception:
        return False


def check_password_rules(pw):
    if not isinstance(pw, str) or len(pw) < 6:
        raise ApiError('Password must be at least 6 characters')
    if len(pw.encode()) > 72:
        raise ApiError('Password is too long (max 72 bytes)')


def make_token(uid, role):
    t = jwt.encode({'uid': uid, 'role': role, 'exp': now() + timedelta(days=7)}, SECRET, algorithm='HS256')
    return t.decode() if isinstance(t, bytes) else t


def auth(role=None):
    def deco(f):
        @wraps(f)
        def inner(*a, **k):
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            try:
                data = jwt.decode(token, SECRET, algorithms=['HS256'])
            except jwt.PyJWTError:
                return jsonify({'error': 'Unauthorized'}), 401
            if role and data.get('role') != role:
                return jsonify({'error': 'Forbidden'}), 403
            request.uid, request.role = data['uid'], data['role']
            return f(*a, **k)
        return inner
    return deco


def udict(u):
    return {'id': u.id, 'name': u.name, 'email': u.email, 'role': u.role, 'department': u.department,
            'year': u.year, 'roll_no': u.roll_no, 'show_lb': bool(u.show_lb),
            'has_security': u.security_q is not None}


def all_cgpas():
    acc = {}
    for s in Semester.query.all():
        a = acc.setdefault(s.user_id, [0.0, 0])
        a[0] += s.sgpa * s.credits
        a[1] += s.credits
    return {uid: (round(a[0] / a[1], 2) if a[1] else 0.0) for uid, a in acc.items()}


def cgpa_of(uid):
    return all_cgpas().get(uid, 0.0)


def range_warnings(vals):
    out = []
    for f, v in zip(FEATURES, vals):
        lo, hi = RANGES.get(f, (None, None))
        if lo is not None and (v < lo - 1e-9 or v > hi + 1e-9):
            out.append(f'{FEATURE_LABELS.get(f, f)} ({v:g}) is outside the range the model was trained on '
                       f'({lo:g}-{hi:g}); treat this result as a rough estimate.')
    return out


# ───────────────────────── public routes ─────────────────────────
def find_frontend():
    for d in ('frontend', 'Frontend', os.path.join('..', 'frontend'), os.path.join('..', 'Frontend')):
        p = os.path.join(BASE, d)
        if os.path.exists(os.path.join(p, 'index.html')):
            return p
    return None


@app.route('/')
def root():
    fe = find_frontend()
    if fe:
        return send_from_directory(fe, 'index.html')
    return jsonify({'message': 'ASPIRE backend running', 'status': 'ok'})


@app.route('/api/ping')
def ping():
    return jsonify({'pong': True})


@app.route('/api/config')
def config():
    return jsonify({'departments': DEPARTMENTS, 'security_questions': SECURITY_QUESTIONS,
                    'thresholds': THRESH})


# ───────────────────────── auth ─────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    d = request.get_json(silent=True) or {}
    name = str(d.get('name', '')).strip()[:100]
    email = str(d.get('email', '')).strip().lower()[:120]
    if not name or '@' not in email or '.' not in email.split('@')[-1]:
        raise ApiError('Enter your name and a valid email address')
    check_password_rules(d.get('password'))
    dept = d.get('department', 'CSE')
    if dept not in DEPARTMENTS:
        raise ApiError('Invalid department')
    year = int(num(d, 'year', 1, 4, default=1))
    sq = d.get('security_q')
    sa = str(d.get('security_a', '')).strip().lower()
    if not isinstance(sq, int) or not (0 <= sq < len(SECURITY_QUESTIONS)) or len(sa) < 2:
        raise ApiError('Choose a security question and give an answer (needed for password recovery)')
    if User.query.filter_by(email=email).first():
        raise ApiError('Email already registered', 409)
    u = User(name=name, email=email, password=hash_text(d['password']), role='student',
             department=dept, year=year, roll_no=str(d.get('roll_no', '')).strip()[:20],
             security_q=sq, security_a=hash_text(sa))
    db.session.add(u)
    db.session.commit()
    return jsonify({'token': make_token(u.id, u.role), 'user': udict(u)}), 201


@app.route('/api/login', methods=['POST'])
def login():
    d = request.get_json(silent=True) or {}
    u = User.query.filter_by(email=str(d.get('email', '')).strip().lower()).first()
    if not u or not check_text(str(d.get('password', '')), u.password):
        raise ApiError('Invalid email or password', 401)
    return jsonify({'token': make_token(u.id, u.role), 'user': udict(u)})


@app.route('/api/me')
@auth()
def me():
    u = db.session.get(User, request.uid)
    if not u:
        return jsonify({'error': 'Unauthorized'}), 401
    d = udict(u)
    c = cgpa_of(u.id)
    d.update({'cgpa': c, 'classification': classify(c),
              'sem_count': Semester.query.filter_by(user_id=u.id).count()})
    return jsonify(d)


# Forgot password: security-question based (no email server needed)
RESET_ATTEMPTS = {}


def reset_blocked(email):
    rec = RESET_ATTEMPTS.get(email)
    if rec and time.time() - rec[1] > 900:
        RESET_ATTEMPTS.pop(email, None)
        return False
    return bool(rec and rec[0] >= 5)


def note_reset_fail(email):
    rec = RESET_ATTEMPTS.get(email)
    RESET_ATTEMPTS[email] = [(rec[0] + 1) if rec else 1, rec[1] if rec else time.time()]


@app.route('/api/forgot/question', methods=['POST'])
def forgot_question():
    email = str((request.get_json(silent=True) or {}).get('email', '')).strip().lower()
    u = User.query.filter_by(email=email, role='student').first()
    if not u or u.security_q is None:
        raise ApiError('No recovery question found for this email. Please contact your administrator.', 404)
    return jsonify({'question': SECURITY_QUESTIONS[u.security_q]})


@app.route('/api/forgot/reset', methods=['POST'])
def forgot_reset():
    d = request.get_json(silent=True) or {}
    email = str(d.get('email', '')).strip().lower()
    if reset_blocked(email):
        raise ApiError('Too many wrong attempts. Try again in 15 minutes or contact your administrator.', 429)
    u = User.query.filter_by(email=email, role='student').first()
    if not u or u.security_q is None:
        raise ApiError('No recovery question found for this email.', 404)
    check_password_rules(d.get('new_password'))
    if not check_text(str(d.get('answer', '')).strip().lower(), u.security_a):
        note_reset_fail(email)
        raise ApiError('Incorrect answer', 403)
    u.password = hash_text(d['new_password'])
    db.session.commit()
    RESET_ATTEMPTS.pop(email, None)
    return jsonify({'message': 'Password updated. You can sign in now.'})


@app.route('/api/me/security', methods=['POST'])
@auth()
def set_security():
    d = request.get_json(silent=True) or {}
    sq, sa = d.get('security_q'), str(d.get('security_a', '')).strip().lower()
    if not isinstance(sq, int) or not (0 <= sq < len(SECURITY_QUESTIONS)) or len(sa) < 2:
        raise ApiError('Choose a question and give an answer')
    u = db.session.get(User, request.uid)
    u.security_q, u.security_a = sq, hash_text(sa)
    db.session.commit()
    return jsonify({'message': 'Recovery question saved'})


@app.route('/api/me/password', methods=['POST'])
@auth()
def change_password():
    d = request.get_json(silent=True) or {}
    u = db.session.get(User, request.uid)
    if not check_text(str(d.get('current', '')), u.password):
        raise ApiError('Current password is incorrect', 403)
    check_password_rules(d.get('new'))
    u.password = hash_text(d['new'])
    db.session.commit()
    return jsonify({'message': 'Password changed'})


# ───────────────────────── semesters ─────────────────────────
def sem_dict(s, cgpa_so_far):
    return {'id': s.id, 'sem_no': s.sem_no, 'sgpa': s.sgpa, 'credits': s.credits,
            'cgpa_so_far': cgpa_so_far,
            'subjects': [{'id': x.id, 'name': x.name, 'credits': x.credits, 'kind': x.kind or 'theory',
                          'internal': x.internal, 'end_sem': x.end_sem, 'is_predicted': bool(x.is_predicted),
                          'total': x.total, 'grade': x.grade, 'gp': x.gp} for x in s.subjects]}


@app.route('/api/semesters')
@auth()
def get_semesters():
    uid = request.uid
    if request.role == 'admin' and request.args.get('uid'):
        uid = int(request.args['uid'])
    tp = tc = 0
    out = []
    for s in Semester.query.filter_by(user_id=uid).order_by(Semester.sem_no).all():
        tp += s.sgpa * s.credits
        tc += s.credits
        out.append(sem_dict(s, round(tp / tc, 2)))
    return jsonify(out)


@app.route('/api/semesters', methods=['POST'])
@auth()
def save_semester():
    d = request.get_json(silent=True) or {}
    sem_no = int(num(d, 'sem_no', 1, 8, required=True))
    subs = d.get('subjects')
    if not isinstance(subs, list) or not (1 <= len(subs) <= 20):
        raise ApiError('Add between 1 and 20 subjects')
    clean, tp, tc = [], 0.0, 0
    for s in subs:
        if not isinstance(s, dict):
            raise ApiError('Invalid subject data')
        kind = 'lab' if s.get('kind') == 'lab' else 'theory'
        credits = int(num(s, 'credits', 1, 6, default=3))
        internal = num(s, 'internal', 0, 50, required=True)
        end = 0.0 if kind == 'lab' else num(s, 'end_sem', 0, 50, required=True)
        total = internal * 2 if kind == 'lab' else internal + end   # lab is graded out of 50 -> scaled to 100
        grade = get_grade(total)
        tp += GRADE_PTS[grade] * credits
        tc += credits
        clean.append(Subject(name=(str(s.get('name') or 'Subject').strip()[:120] or 'Subject'),
                             credits=credits, kind=kind, internal=internal, end_sem=end,
                             is_predicted=bool(s.get('is_predicted')) and kind == 'theory',
                             total=round(total, 1), grade=grade, gp=GRADE_PTS[grade]))
    sgpa = round(tp / tc, 2)                       # SGPA is always computed server-side
    old = Semester.query.filter_by(user_id=request.uid, sem_no=sem_no).first()
    if old:
        db.session.delete(old)
        db.session.flush()
    sem = Semester(user_id=request.uid, sem_no=sem_no, sgpa=sgpa, credits=tc, subjects=clean)
    db.session.add(sem)
    db.session.commit()
    return jsonify({'message': f'Semester {sem_no} saved', 'sem_no': sem_no, 'sgpa': sgpa,
                    'cgpa': cgpa_of(request.uid)}), 201


@app.route('/api/semesters/<int:sid>', methods=['DELETE'])
@auth()
def del_semester(sid):
    s = db.session.get(Semester, sid)
    if not s or s.user_id != request.uid:
        raise ApiError('Not found', 404)
    db.session.delete(s)
    db.session.commit()
    return jsonify({'message': 'Deleted', 'cgpa': cgpa_of(request.uid)})


# ───────────────────────── prediction ─────────────────────────
@app.route('/api/predict', methods=['POST'])
@auth()
def predict():
    d = request.get_json(silent=True) or {}
    internal = num(d, 'internal', 0, 50, required=True)
    att = num(d, 'attendance', 0, 100, default=75)
    study = num(d, 'study_hrs', 0, 16, default=3)
    days = num(d, 'days_left', 0, 120, default=21)
    diff = int(num(d, 'difficulty', 1, 3, default=2))
    vals = [internal, att, study, days, diff]

    pred = round(float(model_predict([vals])[0]), 1)
    total = internal + pred
    grade = get_grade(total)
    risk = RISK_LABELS[int(risk_model.predict(pd.DataFrame([vals], columns=FEATURES))[0])]

    what_if = {}
    if study + 1 <= 16:
        what_if['study_plus_1h'] = round(float(model_predict([[internal, att, study + 1, days, diff]])[0]) - pred, 1)
    if att + 10 <= 100:
        what_if['attendance_plus_10'] = round(float(model_predict([[internal, att + 10, study, days, diff]])[0]) - pred, 1)

    if d.get('log'):
        db.session.add(Prediction(user_id=request.uid, subject=str(d.get('subject', ''))[:120],
                                  credits=int(num(d, 'credits', 1, 6, default=3)), internal=internal,
                                  attendance=att, study_hrs=study, days_left=int(days), difficulty=diff,
                                  pred_end=pred, grade=grade, risk=risk))
        db.session.commit()
    return jsonify({'pred_end': pred, 'total': round(total, 1), 'grade': grade, 'gp': GRADE_PTS[grade],
                    'risk': risk, 'what_if': what_if, 'warnings': range_warnings(vals),
                    'model': META['selected_model'], 'dataset_type': META['dataset_type']})


@app.route('/api/reverse', methods=['POST'])
@auth()
def reverse():
    """Target planner: what do I need to reach grade X?"""
    d = request.get_json(silent=True) or {}
    target = d.get('target_grade', 'AA')
    if target not in THRESH:
        raise ApiError('target_grade must be one of ' + ', '.join(THRESH))
    thr = THRESH[target]
    diff = int(num(d, 'difficulty', 1, 3, default=2))
    att = num(d, 'attendance', 0, 100, default=75)
    study = num(d, 'study_hrs', 0, 16, default=3)
    days = num(d, 'days_left', 0, 120, default=21)
    internal = num(d, 'internal', 0, 50, default=None)

    lo, hi = RANGES.get('internal_marks', [20, 50])
    grid = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1, 1.0)
    preds = model_predict([[i, att, study, days, diff] for i in grid])
    totals = grid + preds
    meets = totals >= thr
    min_internal = float(grid[meets][0]) if meets.any() else None
    table = [{'internal': float(i), 'pred_end': round(float(p), 1), 'total': round(float(t), 1),
              'grade': get_grade(t), 'meets': bool(t >= thr)}
             for i, p, t in zip(grid, preds, totals) if int(i) % 5 == 0 or i == grid[-1]]

    out = {'target': target, 'threshold': thr, 'min_internal': min_internal, 'table': table,
           'warnings': range_warnings([lo, att, study, days, diff])}   # internal is gridded inside its range
    if internal is not None:
        required_end = max(0.0, thr - internal)
        cur = float(model_predict([[internal, att, study, days, diff]])[0])
        s_lo, s_hi = RANGES.get('study_hrs_day', [0.5, 8])
        hours = np.arange(0, s_hi + 0.01, 0.5)
        sp = model_predict([[internal, att, h, days, diff] for h in hours])
        ok = (internal + sp) >= thr
        out.update({'internal': internal, 'required_end_sem': round(required_end, 1),
                    'feasible': required_end <= 50, 'max_possible_total': round(internal + 50, 1),
                    'current_pred_end': round(cur, 1), 'current_total': round(internal + cur, 1),
                    'current_grade': get_grade(internal + cur),
                    'min_study_hrs': float(hours[ok][0]) if ok.any() else None, 'study_cap': float(s_hi)})
    return jsonify(out)


@app.route('/api/predictions')
@auth()
def get_predictions():
    rows = Prediction.query.filter_by(user_id=request.uid).order_by(Prediction.saved_at.desc()).limit(100).all()
    return jsonify([{'id': p.id, 'subject': p.subject, 'internal': p.internal, 'pred_end': p.pred_end,
                     'grade': p.grade, 'risk': p.risk, 'date': p.saved_at.strftime('%d %b %Y')} for p in rows])


@app.route('/api/model/info')
@auth()
def model_info():
    return jsonify(META)


# ───────────────────────── department stats / leaderboard ─────────────────────────
@app.route('/api/dept/stats')
@auth()
def dept_stats():
    cg = all_cgpas()
    buckets = {}
    for u in User.query.filter_by(role='student').all():
        c = cg.get(u.id, 0)
        if c > 0:
            buckets.setdefault(u.department, []).append(c)
    me_u = db.session.get(User, request.uid)
    return jsonify({
        'departments': {d: {'avg': round(sum(v) / len(v), 2), 'count': len(v)} for d, v in buckets.items()},
        'my_dept': me_u.department, 'my_cgpa': cg.get(me_u.id, 0.0)})


@app.route('/api/leaderboard')
@auth()
def leaderboard():
    dept = request.args.get('dept', 'ALL')
    cg = all_cgpas()
    q = User.query.filter_by(role='student', show_lb=True)
    if dept != 'ALL':
        q = q.filter_by(department=dept)
    board = [{'id': u.id, 'name': u.name, 'department': u.department, 'year': u.year,
              'roll_no': u.roll_no, 'cgpa': cg[u.id]} for u in q.all() if cg.get(u.id, 0) > 0]
    board.sort(key=lambda x: -x['cgpa'])
    for i, b in enumerate(board):
        b['rank'] = i + 1
    return jsonify(board)


@app.route('/api/leaderboard/toppers')
@auth()
def toppers():
    cg = all_cgpas()
    out = {}
    for u in User.query.filter_by(role='student', show_lb=True).all():
        c = cg.get(u.id, 0)
        if c > 0 and (u.department not in out or c > out[u.department]['cgpa']):
            out[u.department] = {'name': u.name, 'cgpa': c, 'roll_no': u.roll_no, 'year': u.year}
    return jsonify(out)


@app.route('/api/leaderboard/opt', methods=['POST'])
@auth()
def lb_opt():
    u = db.session.get(User, request.uid)
    u.show_lb = bool((request.get_json(silent=True) or {}).get('show', True))
    db.session.commit()
    return jsonify({'show_lb': u.show_lb})


# ───────────────────────── admin ─────────────────────────
@app.route('/api/admin/students')
@auth('admin')
def admin_students():
    cg = all_cgpas()
    counts = {}
    for s in Semester.query.all():
        counts[s.user_id] = counts.get(s.user_id, 0) + 1
    out = [{'id': u.id, 'name': u.name, 'email': u.email, 'department': u.department, 'year': u.year,
            'roll_no': u.roll_no, 'cgpa': cg.get(u.id, 0.0), 'classification': classify(cg.get(u.id, 0.0)),
            'sems': counts.get(u.id, 0), 'has_security': u.security_q is not None}
           for u in User.query.filter_by(role='student').all()]
    return jsonify(sorted(out, key=lambda x: -x['cgpa']))


@app.route('/api/admin/analytics')
@auth('admin')
def admin_analytics():
    cg = all_cgpas()
    students = User.query.filter_by(role='student').all()
    active = [cg[u.id] for u in students if cg.get(u.id, 0) > 0]
    depts = {}
    for u in students:
        if cg.get(u.id, 0) > 0:
            depts.setdefault(u.department, []).append(cg[u.id])
    return jsonify({'total': len(students), 'with_data': len(active),
                    'avg_cgpa': round(sum(active) / len(active), 2) if active else 0,
                    'at_risk': sum(1 for c in active if c < 5),
                    'distinction': sum(1 for c in active if c >= 9),
                    'first_class': sum(1 for c in active if 7 <= c < 9),
                    'pass_class': sum(1 for c in active if 5 <= c < 7),
                    'dept_avg': {d: round(sum(v) / len(v), 2) for d, v in depts.items()}})


@app.route('/api/admin/reset-password', methods=['POST'])
@auth('admin')
def admin_reset_password():
    d = request.get_json(silent=True) or {}
    u = db.session.get(User, int(d.get('user_id', 0)))
    if not u or u.role != 'student':
        raise ApiError('Student not found', 404)
    check_password_rules(d.get('new_password'))
    u.password = hash_text(d['new_password'])
    db.session.commit()
    return jsonify({'message': f'Password reset for {u.name}'})


# ───────────────────────── startup ─────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        ensure_columns()
        admin_email = os.environ.get('ASPIRE_ADMIN_EMAIL', 'admin@rbu.ac.in')
        if not User.query.filter_by(email=admin_email).first():
            admin_pw = os.environ.get('ASPIRE_ADMIN_PASSWORD', 'admin123')
            db.session.add(User(name='Admin RBU', email=admin_email, password=hash_text(admin_pw),
                                role='admin', department='Admin', year=0))
            db.session.commit()
            print(f'Admin created: {admin_email}' + ('  (default password - change it!)' if admin_pw == 'admin123' else ''))


init_db()

if __name__ == '__main__':
    # Local run. On Render, gunicorn starts the app instead (see render.yaml).
    port = int(os.environ.get('PORT', 5000))
    print(f'ASPIRE running on http://127.0.0.1:{port}')
    app.run(debug=os.environ.get('FLASK_DEBUG', '1') == '1', host='127.0.0.1', port=port)
