from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import bcrypt, jwt, joblib, json, numpy as np, os
from datetime import datetime, timedelta
from functools import wraps

app = Flask(__name__)
CORS(app, resources={r"/*":{"origins":"*","methods":["GET","POST","PUT","DELETE","OPTIONS"],"allow_headers":["Content-Type","Authorization"]}})

BASE = os.path.dirname(__file__)
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{BASE}/aspire.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'aspire-rbu-2025-secret'
db = SQLAlchemy(app)

ML = os.path.join(BASE, 'ml')
rf = joblib.load(os.path.join(ML,'rf.pkl'))
dt = joblib.load(os.path.join(ML,'dt.pkl'))
with open(os.path.join(ML,'metrics.json')) as f: ML_METRICS = json.load(f)

GRADE_PTS = {'AA':10,'AB':9,'BB':8,'BC':7,'CC':6,'CD':5,'DD':4,'FF':0}

def get_grade(total):
    if total>=90: return 'AA'
    if total>=80: return 'AB'
    if total>=70: return 'BB'
    if total>=60: return 'BC'
    if total>=50: return 'CC'
    if total>=40: return 'CD'
    if total>=30: return 'DD'
    return 'FF'

def classify(c):
    if c>=9:  return 'Distinction'
    if c>=7:  return 'First Class'
    if c>=5:  return 'Pass'
    if c>0:   return 'At Risk'
    return 'No Data'

# ── Database Models ────────────────────────────────────────
class User(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    role       = db.Column(db.String(10), default='student')
    department = db.Column(db.String(50), default='CSE')
    year       = db.Column(db.Integer, default=1)
    roll_no    = db.Column(db.String(20), default='')
    show_lb    = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Semester(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    user_id  = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sem_no   = db.Column(db.Integer, nullable=False)
    sgpa     = db.Column(db.Float, nullable=False)
    credits  = db.Column(db.Integer, nullable=False)
    saved_at = db.Column(db.DateTime, default=datetime.utcnow)
    subjects = db.relationship('Subject', backref='semester', cascade='all,delete', lazy=True)

class Subject(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    sem_id       = db.Column(db.Integer, db.ForeignKey('semester.id'), nullable=False)
    name         = db.Column(db.String(120))
    credits      = db.Column(db.Integer)
    internal     = db.Column(db.Float)
    lab_marks    = db.Column(db.Float, nullable=True)
    has_lab      = db.Column(db.Boolean, default=False)
    end_sem      = db.Column(db.Float)
    is_predicted = db.Column(db.Boolean, default=False)
    total        = db.Column(db.Float)
    grade        = db.Column(db.String(5))
    gp           = db.Column(db.Float)

class Prediction(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'))
    subject     = db.Column(db.String(120))
    credits     = db.Column(db.Integer, default=3)
    has_lab     = db.Column(db.Boolean, default=False)
    internal    = db.Column(db.Float)
    lab_marks   = db.Column(db.Float, nullable=True)
    attendance  = db.Column(db.Float)
    study_hrs   = db.Column(db.Float)
    days_left   = db.Column(db.Integer)
    difficulty  = db.Column(db.Integer)
    pred_end    = db.Column(db.Float)
    grade       = db.Column(db.String(5))
    risk        = db.Column(db.String(20))
    saved_at    = db.Column(db.DateTime, default=datetime.utcnow)

# ── Auth helpers ───────────────────────────────────────────
def make_token(uid, role):
    return jwt.encode(
        {'uid':uid,'role':role,'exp':datetime.utcnow()+timedelta(days=7)},
        app.config['SECRET_KEY'], algorithm='HS256'
    )

def auth(f):
    @wraps(f)
    def inner(*a, **k):
        t = request.headers.get('Authorization','').replace('Bearer ','')
        try:
            d = jwt.decode(t, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.uid  = d['uid']
            request.role = d['role']
        except: return jsonify({'error':'Unauthorized'}), 401
        return f(*a, **k)
    return inner

def admin_only(f):
    @wraps(f)
    def inner(*a, **k):
        t = request.headers.get('Authorization','').replace('Bearer ','')
        try:
            d = jwt.decode(t, app.config['SECRET_KEY'], algorithms=['HS256'])
            if d['role'] != 'admin': return jsonify({'error':'Admin only'}), 403
            request.uid  = d['uid']
            request.role = d['role']
        except: return jsonify({'error':'Unauthorized'}), 401
        return f(*a, **k)
    return inner

def cgpa_of(uid):
    sems = Semester.query.filter_by(user_id=uid).all()
    if not sems: return 0.0
    tp = sum(s.sgpa * s.credits for s in sems)
    tc = sum(s.credits for s in sems)
    return round(tp/tc, 2) if tc else 0.0

def udict(u):
    return {'id':u.id,'name':u.name,'email':u.email,'role':u.role,
            'department':u.department,'year':u.year,'roll_no':u.roll_no,'show_lb':u.show_lb}

# ── Utility ────────────────────────────────────────────────
@app.route('/')
def root(): return jsonify({'message':'ASPIRE backend running','status':'ok'})

@app.route('/api/ping')
def ping(): return jsonify({'pong':True})

# ── Auth ───────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    d = request.json
    if not d.get('email') or not d.get('password') or not d.get('name'):
        return jsonify({'error':'Name, email and password required'}), 400
    if d.get('role') == 'admin':
        return jsonify({'error':'Admin accounts cannot be self-registered. Contact administrator.'}), 403
    if User.query.filter_by(email=d['email']).first():
        return jsonify({'error':'Email already registered'}), 409
    pw = bcrypt.hashpw(d['password'].encode(), bcrypt.gensalt()).decode()
    u  = User(name=d['name'], email=d['email'], password=pw, role='student',
              department=d.get('department','CSE'), year=int(d.get('year',1)),
              roll_no=d.get('roll_no',''))
    db.session.add(u); db.session.commit()
    return jsonify({'token':make_token(u.id, u.role), 'user':udict(u)}), 201

@app.route('/api/login', methods=['POST'])
def login():
    d = request.json
    u = User.query.filter_by(email=d.get('email','')).first()
    if not u or not bcrypt.checkpw(d.get('password','').encode(), u.password.encode()):
        return jsonify({'error':'Invalid email or password'}), 401
    return jsonify({'token':make_token(u.id, u.role), 'user':udict(u)})

@app.route('/api/me', methods=['GET'])
@auth
def me():
    u = User.query.get(request.uid)
    d = udict(u)
    d['cgpa']           = cgpa_of(u.id)
    d['classification'] = classify(d['cgpa'])
    d['sem_count']      = Semester.query.filter_by(user_id=u.id).count()
    return jsonify(d)

# ── Semesters ──────────────────────────────────────────────
@app.route('/api/semesters', methods=['GET'])
@auth
def get_semesters():
    uid = int(request.args.get('uid', request.uid))
    if uid != request.uid and request.role != 'admin': uid = request.uid
    sems = Semester.query.filter_by(user_id=uid).order_by(Semester.sem_no).all()
    tp = tc = 0; out = []
    for s in sems:
        tp += s.sgpa * s.credits; tc += s.credits
        out.append({
            'id':s.id,'sem_no':s.sem_no,'sgpa':s.sgpa,'credits':s.credits,
            'cgpa_so_far':round(tp/tc,2),
            'subjects':[{
                'id':sub.id,'name':sub.name,'credits':sub.credits,
                'internal':sub.internal,'lab_marks':sub.lab_marks,'has_lab':sub.has_lab,
                'end_sem':sub.end_sem,'is_predicted':sub.is_predicted,
                'total':sub.total,'grade':sub.grade,'gp':sub.gp
            } for sub in s.subjects]
        })
    return jsonify(out)

@app.route('/api/semesters', methods=['POST'])
@auth
def save_semester():
    d = request.json
    ex = Semester.query.filter_by(user_id=request.uid, sem_no=d['sem_no']).first()
    if ex: db.session.delete(ex); db.session.commit()
    sem = Semester(user_id=request.uid, sem_no=d['sem_no'], sgpa=round(d['sgpa'],2), credits=d['credits'])
    db.session.add(sem); db.session.flush()
    for sub in d['subjects']:
        total = sub['internal'] + sub['end_sem']
        grade = get_grade(total)
        db.session.add(Subject(
            sem_id=sem.id, name=sub['name'], credits=sub['credits'],
            internal=sub['internal'], lab_marks=sub.get('lab_marks'), has_lab=sub.get('has_lab',False),
            end_sem=sub['end_sem'], is_predicted=sub.get('is_predicted',False),
            total=round(total,1), grade=grade, gp=GRADE_PTS[grade]
        ))
    db.session.commit()
    return jsonify({'message':f'Semester {d["sem_no"]} saved','sgpa':round(d['sgpa'],2),'cgpa':cgpa_of(request.uid)}), 201

@app.route('/api/semesters/<int:sid>', methods=['DELETE'])
@auth
def del_semester(sid):
    s = Semester.query.filter_by(id=sid, user_id=request.uid).first_or_404()
    db.session.delete(s); db.session.commit()
    return jsonify({'message':'Deleted','cgpa':cgpa_of(request.uid)})

# ── Prediction ─────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
@auth
def predict():
    d = request.json
    X = np.array([[
        float(d.get('internal',0)),
        float(d.get('attendance',75)),
        float(d.get('study_hrs',3)),
        float(d.get('days_left',21)),
        int(d.get('difficulty',2))
    ]])
    pred_end  = float(np.clip(rf.predict(X)[0], 0, 50))
    risk_code = int(dt.predict(X)[0])
    risk_map  = {0:'At Risk', 1:'On Track', 2:'Good'}
    total     = float(d.get('internal',0)) + pred_end
    grade     = get_grade(total)
    p = Prediction(
        user_id=request.uid, subject=d.get('subject',''), credits=int(d.get('credits',3)),
        has_lab=bool(d.get('has_lab',False)), internal=d.get('internal',0),
        lab_marks=d.get('lab_marks'), attendance=d.get('attendance',75),
        study_hrs=d.get('study_hrs',3), days_left=d.get('days_left',21),
        difficulty=d.get('difficulty',2), pred_end=round(pred_end,1),
        grade=grade, risk=risk_map[risk_code]
    )
    db.session.add(p); db.session.commit()
    return jsonify({
        'pred_end':round(pred_end,1),'total':round(total,1),'grade':grade,
        'gp':GRADE_PTS[grade],'risk':risk_map[risk_code],'risk_code':risk_code,
        'metrics':ML_METRICS, 'prediction_id': p.id
    })

@app.route('/api/predictions', methods=['GET'])
@auth
def get_predictions():
    preds = Prediction.query.filter_by(user_id=request.uid).order_by(Prediction.saved_at.desc()).all()
    return jsonify([{
        'id':p.id,'subject':p.subject,'credits':p.credits,'has_lab':p.has_lab,
        'internal':p.internal,'lab_marks':p.lab_marks,'attendance':p.attendance,
        'study_hrs':p.study_hrs,'pred_end':p.pred_end,'grade':p.grade,'risk':p.risk,
        'date':p.saved_at.strftime('%d %b %Y')
    } for p in preds])

# ── Leaderboard ────────────────────────────────────────────
@app.route('/api/leaderboard', methods=['GET'])
@auth
def leaderboard():
    dept  = request.args.get('dept','ALL')
    q     = User.query.filter_by(role='student', show_lb=True)
    if dept != 'ALL': q = q.filter_by(department=dept)
    board = []
    for u in q.all():
        c = cgpa_of(u.id)
        if c > 0: board.append({'id':u.id,'name':u.name,'department':u.department,'year':u.year,'roll_no':u.roll_no,'cgpa':c})
    board.sort(key=lambda x: x['cgpa'], reverse=True)
    for i,b in enumerate(board): b['rank'] = i+1
    return jsonify(board)

@app.route('/api/leaderboard/toppers', methods=['GET'])
@auth
def dept_toppers():
    result = {}
    for dept in ['CSE','IT','ECE','MECH','CIVIL']:
        best=None; bc=0
        for u in User.query.filter_by(role='student',department=dept,show_lb=True).all():
            c = cgpa_of(u.id)
            if c > bc: bc=c; best=u
        if best and bc>0: result[dept]={'name':best.name,'cgpa':bc,'roll_no':best.roll_no,'year':best.year}
    return jsonify(result)

@app.route('/api/leaderboard/opt', methods=['POST'])
@auth
def lb_opt():
    u = User.query.get(request.uid)
    u.show_lb = bool(request.json.get('show',True))
    db.session.commit()
    return jsonify({'show_lb':u.show_lb})

# ── Admin ──────────────────────────────────────────────────
@app.route('/api/admin/students', methods=['GET'])
@admin_only
def admin_students():
    students = User.query.filter_by(role='student').all()
    out = [{'id':u.id,'name':u.name,'email':u.email,'department':u.department,'year':u.year,
            'roll_no':u.roll_no,'cgpa':cgpa_of(u.id),'classification':classify(cgpa_of(u.id)),
            'sems':Semester.query.filter_by(user_id=u.id).count()} for u in students]
    return jsonify(sorted(out, key=lambda x: x['cgpa'], reverse=True))

@app.route('/api/admin/analytics', methods=['GET'])
@admin_only
def admin_analytics():
    students = User.query.filter_by(role='student').all()
    cgpas    = [cgpa_of(u.id) for u in students]
    active   = [c for c in cgpas if c > 0]
    depts    = {}
    for u in students:
        d = u.department or 'Unknown'
        if d not in depts: depts[d] = {'count':0,'sum':0}
        depts[d]['count'] += 1; depts[d]['sum'] += cgpa_of(u.id)
    dept_avg = {d: round(v['sum']/v['count'],2) if v['count'] else 0 for d,v in depts.items()}
    return jsonify({
        'total':len(students),
        'avg_cgpa':round(sum(active)/len(active),2) if active else 0,
        'at_risk':sum(1 for c in active if c<5),
        'distinction':sum(1 for c in active if c>=9),
        'first_class':sum(1 for c in active if 7<=c<9),
        'pass_class':sum(1 for c in active if 5<=c<7),
        'dept_avg':dept_avg,'model_metrics':ML_METRICS
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email='admin@rbu.ac.in').first():
            pw = bcrypt.hashpw(b'admin123', bcrypt.gensalt()).decode()
            db.session.add(User(name='Admin RBU',email='admin@rbu.ac.in',password=pw,role='admin',department='Admin',year=0))
            db.session.commit()
            print('Admin seeded: admin@rbu.ac.in / admin123')
    print('ASPIRE running on http://127.0.0.1:5000')
    app.run(debug=True, port=5000)