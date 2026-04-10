from flask import Flask, jsonify, request, render_template
from datetime import datetime
import threading
import time
import os

from database import db, init_db
from models import Metric, Action
from simulator import read_current_metrics, simulate_failure_metric
from rl_agent import RLAgent


app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'system.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
init_db(app)

rl_agent = RLAgent(model_path=os.path.join(os.path.dirname(__file__), 'model.zip'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Read current metrics and persist
    cpu, mem, status = read_current_metrics()
    metric = Metric(timestamp=datetime.utcnow(), cpu=cpu, memory=mem, status=status)
    db.session.add(metric)
    db.session.commit()

    # Return last 50 metrics for charting
    last_metrics = (
        Metric.query.order_by(Metric.id.desc()).limit(50).all()
    )
    last_metrics = list(reversed(last_metrics))
    return jsonify({
        'status': status,
        'metrics': [
            {
                'timestamp': m.timestamp.isoformat() + 'Z',
                'cpu': m.cpu,
                'memory': m.memory,
                'status': m.status,
            } for m in last_metrics
        ]
    })


@app.route('/detect', methods=['POST'])
def detect_failure():
    # Simple threshold-based detection as a baseline
    data = request.get_json(silent=True) or {}
    cpu = data.get('cpu')
    memory = data.get('memory')
    if cpu is None or memory is None:
        cpu, memory, _ = read_current_metrics()
    status = 'Failed' if cpu >= 90 or memory >= 90 else 'Healthy'
    metric = Metric(timestamp=datetime.utcnow(), cpu=cpu, memory=memory, status=status)
    db.session.add(metric)
    db.session.commit()
    return jsonify({'detected': status == 'Failed', 'status': status})


@app.route('/recover', methods=['POST'])
def recover():
    # Build state from recent metrics
    last_metric = Metric.query.order_by(Metric.id.desc()).first()
    failure_count = Action.query.filter_by(result='failed').count()
    if last_metric is None:
        cpu, mem, status = read_current_metrics()
    else:
        cpu, mem, status = last_metric.cpu, last_metric.memory, last_metric.status

    state = [cpu / 100.0, mem / 100.0, min(failure_count, 10) / 10.0]
    action_name = rl_agent.select_action(state)

    # Apply action effect (simulated probability) as prior
    _ = rl_agent.apply_action_effect(action_name, cpu, mem, status)

    # Validate recovery within a timeout window by sampling metrics
    start = time.time()
    baseline_cpu, baseline_mem = cpu, mem
    recovered = False
    sample_interval = 0.5
    max_wait = 5.0
    observed_recovery_time = None
    while time.time() - start < max_wait:
        cur_cpu, cur_mem, cur_status = read_current_metrics()
        if cur_status == 'Healthy' and (cur_cpu <= baseline_cpu or cur_mem <= baseline_mem):
            recovered = True
            observed_recovery_time = time.time() - start
            break
        time.sleep(sample_interval)

    # Reward shaping: fast recovery positive; slow/no recovery negative
    if recovered:
        # success_bonus decays with time, bounded
        success_bonus = max(2.0, 10.0 - (observed_recovery_time or max_wait) * 2.0)
        reward = success_bonus
        result = 'recovered'
    else:
        reward = -10.0
        result = 'failed'

    # Log action with recovery time
    action_row = Action(
        timestamp=datetime.utcnow(), action=action_name, result=result, reward=reward,
        recovery_time=observed_recovery_time
    )
    db.session.add(action_row)
    db.session.commit()

    return jsonify({'action': action_name, 'result': result, 'reward': reward, 'recovery_time': observed_recovery_time})


@app.route('/train', methods=['POST'])
def train():
    # Train RL model for a short session
    episodes = int((request.get_json(silent=True) or {}).get('episodes', 10))
    rl_agent.train(episodes=episodes)
    rl_agent.save()
    return jsonify({'status': 'trained', 'episodes': episodes})


@app.route('/actions', methods=['GET'])
def actions():
    last_actions = Action.query.order_by(Action.id.desc()).limit(50).all()
    last_actions = list(reversed(last_actions))
    return jsonify([
        {
            'timestamp': a.timestamp.isoformat() + 'Z',
            'action': a.action,
            'result': a.result,
            'reward': a.reward,
            'recovery_time': a.recovery_time,
        } for a in last_actions
    ])


@app.route('/summary', methods=['GET'])
def summary():
    # Compute analytics: average MTTR, failures, success rate
    total_actions = Action.query.count()
    successes = Action.query.filter_by(result='recovered').count()
    failures = Action.query.filter_by(result='failed').count()
    mttr_values = [a.recovery_time for a in Action.query.filter(Action.recovery_time.isnot(None)).all()]
    avg_mttr = (sum(mttr_values) / len(mttr_values)) if mttr_values else None
    success_rate = (successes / total_actions * 100.0) if total_actions > 0 else None
    return jsonify({
        'total_actions': total_actions,
        'successes': successes,
        'failures': failures,
        'success_rate': success_rate,
        'avg_mttr': avg_mttr,
    })


@app.route('/simulate_failure', methods=['POST'])
def simulate_failure():
    metric = simulate_failure_metric()
    db.session.add(metric)
    db.session.commit()
    return jsonify({'status': 'Failed'})


def background_sampler_loop(stop_event):
    with app.app_context():
        while not stop_event.is_set():
            cpu, mem, status = read_current_metrics()
            m = Metric(timestamp=datetime.utcnow(), cpu=cpu, memory=mem, status=status)
            db.session.add(m)
            db.session.commit()
            stop_event.wait(5.0)


if __name__ == '__main__':
    stop_event = threading.Event()
    sampler_thread = threading.Thread(target=background_sampler_loop, args=(stop_event,), daemon=True)
    sampler_thread.start()
    try:
        app.run(debug=True)
    finally:
        stop_event.set()
