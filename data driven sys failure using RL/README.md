# Data-Driven Failure Detection and Automatic Recovery using Reinforcement Learning

An intelligent, self-healing system that uses Reinforcement Learning (RL) to automatically detect system failures and execute optimal recovery actions.

## 🚀 Quick Start

### Prerequisites
- Python 3.11 (recommended) or 3.9-3.12
- pip package manager

### Installation

```bash
# 1. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch first
python -m pip install torch

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Run the application
python project/app.py
```

Open your browser to **http://127.0.0.1:5000** to access the dashboard.

## 📖 Documentation

For complete project documentation, see **[PROJECT_DESIGN_REPORT.md](PROJECT_DESIGN_REPORT.md)**

The report includes:
- Complete system architecture
- Technical implementation details
- API documentation
- Usage guide
- Performance metrics
- Future enhancements

## ✨ Key Features

- **Real-time Monitoring**: Live CPU and Memory usage tracking
- **Adaptive Failure Detection**: Statistical anomaly detection with moving averages
- **RL-Based Recovery**: Automated action selection using PPO algorithm
- **Visual Dashboard**: Modern web interface with charts and analytics
- **Persistent Learning**: Model saves/loads automatically
- **Comprehensive Logging**: Full action history and metrics storage

## 🎮 Usage

1. **Train the AI**: Click "Train AI Model" to initialize the RL agent
2. **Simulate Failure**: Click "Simulate Failure" to inject a system failure
3. **Run Recovery**: Click "Run Recovery" to see the RL agent in action
4. **Monitor**: Watch real-time metrics and action logs update automatically

## 📁 Project Structure

```
project/
├── app.py              # Flask application & API endpoints
├── database.py         # SQLAlchemy setup
├── models.py           # Database models
├── simulator.py        # Metrics collection & failure simulation
├── rl_agent.py         # Reinforcement Learning agent
├── static/             # CSS and JavaScript
└── templates/          # HTML templates
```

## 🔧 Technologies

- **Backend**: Flask, SQLAlchemy, SQLite
- **RL**: Stable-Baselines3 (PPO), PyTorch
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Monitoring**: psutil

## 📊 API Endpoints

- `GET /metrics` - Get system metrics
- `POST /recover` - Execute recovery action
- `POST /train` - Train RL model
- `GET /actions` - Get action history
- `POST /simulate_failure` - Inject failure
- `GET /summary` - System analytics

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

For questions, issues, or contributions, please refer to the codebase and inline documentation.

---

**Built with ❤️ using Python Flask + Reinforcement Learning**

