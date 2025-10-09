## Simulation of Enhanced PERT Techniques with Dynamic and Real-Time Updates in Project Management

## 🎯 Project Aim

This project designs and implements an enhanced Program Evaluation and Review Technique (PERT) simulation system that integrates dynamic and real-time updates into project management.

- Traditional PERT models are static and cannot easily adapt to changes once a project starts.
This enhanced version introduces:

- Real-time recalculation of schedules when project data changes

- Monte Carlo simulation to capture uncertainty

- WebSocket-based live updates to any connected client or dashboard

Together, these features make project planning more adaptive, predictive, and realistic.

## ⚙️ Core Objectives

- Analyze the limitations of traditional PERT in handling real-time changes.

- Design and simulate an adaptive PERT-based algorithm that reacts dynamically to new task data.

- Implement a backend system capable of:

- Real-time project scheduling updates

- Monte Carlo simulation for probabilistic duration prediction

- Live broadcasting via WebSockets

- Compare the enhanced PERT results with classical PERT estimates.

- Provide an integration-ready backend for visualization in a frontend web dashboard.

## 🧩 Tech Stack
Backend Framework	
-FastAPI
-Python 3.10+	


## 🚀 Setup and Installation Guide
1. Clone the repository

```bash
git clone https://github.com/your-username/pert-backend.git
cd pert-backend
```

2. Set up a virtual environment (recommended)

```bash
python -m venv venv
```

**Activate it:**

**Windows**

```bash
.venv\Scripts\Activate.ps1
```

**Mac/Linux**

```bash
source venv/bin/activate
```

**3. Install dependencies**

fastapi
uvicorn[standard]
numpy
networkx
pydantic


Then install:

```bash
pip install -r fastapi uvicorn[standard] numpy networkx pydantic
```

4. Run the server

```bash
uvicorn main:app --reload
```

***Server will start on:***

👉 http://127.0.0.1:8000

You can test that it’s working by visiting:

http://127.0.0.1:8000/


You should see:

{"message": "PERT API is running with calculations!"}

## 📡 API Endpoints

***1. POST /pert***
Performs the enhanced PERT + Monte Carlo simulation.

Request body:

[
  {"id": "A", "optimistic": 2, "most_likely": 4, "pessimistic": 6, "dependencies": []},
  {"id": "B", "optimistic": 3, "most_likely": 5, "pessimistic": 9, "dependencies": ["A"]}
]


Response:

Project duration (expected)

Task timings (ES, EF, LS, LF, slack)

Critical path

Monte Carlo mean, P50, P80, P95 durations

**2. POST /set-tasks**

Stores tasks in memory for dynamic updates.

Body: same as /pert

Response:

{"message": "Tasks stored successfully", "task_count": 2}

**3. PUT /update-task**

Updates a specific task and recalculates the project in real-time.

Body Example:

{"id": "B", "optimistic": 4, "most_likely": 6, "pessimistic": 8, "dependencies": ["A"]}


Response:

{"message": "Task B updated successfully"}


Also triggers a WebSocket broadcast to connected clients.

**4. GET /recalculate**

Re-runs the PERT simulation for all stored tasks.

Use case: when you want to manually refresh calculations.

**5. GET /compare-pert**

Compares classical PERT (deterministic) and enhanced Monte Carlo PERT models.

Shows metrics such as forecast accuracy and schedule variability.

**6. WebSocket /ws/updates**

Clients (like dashboards) can connect to this endpoint to receive real-time updates when any task is updated.





### 🧩 Key Features

✅ Accurate PERT computation

✅ Monte Carlo simulation for probabilistic forecasts

✅ Real-time task updates

✅ WebSocket live broadcasting

✅ Classical vs enhanced PERT comparison

## 🧪 Testing

You can use:

FastAPI Docs UI
 — built-in Swagger interface to test endpoints.

Postman or Thunder Client (VS Code) — for more control.

Console/WebSocket — to monitor real-time updates.


## 📘 How to Run on Another Machine

**Clone this repo:**
git clone https://github.com/your-username/pert-backend.git

Create and activate a virtual environment.



Run pip install -r requirements.txt

**Launch the server:**
uvicorn main:app --reload

**Access Swagger Docs at**
http://127.0.0.1:8000/docs
