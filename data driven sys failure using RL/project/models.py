from datetime import datetime
from database import db


class Metric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    cpu = db.Column(db.Float, nullable=False)
    memory = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(16), default='Healthy', index=True)


class Action(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    action = db.Column(db.String(32), nullable=False)
    result = db.Column(db.String(16), nullable=False)
    reward = db.Column(db.Float, nullable=False)
    recovery_time = db.Column(db.Float, nullable=True)  # seconds to recover
