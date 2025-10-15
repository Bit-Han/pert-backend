from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import numpy as np
from typing import Optional
from typing import List, Dict
import networkx as nx
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PERT Simulation API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pert-frontend-sdw2.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Define what a task looks like when frontend sends JSON
class TaskIn(BaseModel):
    id: str
    optimistic: float
    most_likely: float
    pessimistic: float
    dependencies: List[str] = []


# Define what the backend will return for each task
class TaskTiming(BaseModel):
    id: str
    duration: float
    ES: float
    EF: float
    LS: float
    LF: float
    slack: float

# Define the result of a Monte Carlo simulation
class MonteCarloResult(BaseModel):
    mean: float
    p50: float
    p80: float
    p95: float
    durations: Optional[List[float]] = None  # raw samples (optional, can be turned off)


# Define the overall result
class PERTResult(BaseModel):
    project_duration: float
    task_timings: List[TaskTiming]
    critical_paths: List[List[str]]
    monte_carlo: Optional[MonteCarloResult] = None

def compute_expected(o: float, m: float, p: float) -> float:
    """Calculate expected task duration using PERT formula."""
    return (o + 4 * m + p) / 6.0



def sample_duration(o: float, m: float, p: float) -> float:
    """Sample from triangular distribution for task duration."""
    return float(np.random.triangular(o, m, p))



def run_monte_carlo(tasks: List[TaskIn], iterations: int = 1000) -> MonteCarloResult:
    project_durations = []

    for _ in range(iterations):
        # Randomize durations each run
        durations = {t.id: sample_duration(t.optimistic, t.most_likely, t.pessimistic) for t in tasks}

        # Build graph
        G = nx.DiGraph()
        for t in tasks:
            G.add_node(t.id)
        for t in tasks:
            for dep in t.dependencies:
                G.add_edge(dep, t.id)

        # Forward pass
        ES, EF = {}, {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            if preds:
                ES[node] = max(EF[p] for p in preds)
            else:
                ES[node] = 0.0
            EF[node] = ES[node] + durations[node]

        project_durations.append(max(EF.values()))

    # Convert to numpy array for stats
    arr = np.array(project_durations)
    mean = float(np.mean(arr))
    p50 = float(np.percentile(arr, 50))
    p80 = float(np.percentile(arr, 80))
    p95 = float(np.percentile(arr, 95))

    return MonteCarloResult(
        mean=mean,
        p50=p50,
        p80=p80,
        p95=p95,
        durations=project_durations  # you can remove if too large
    )




@app.post("/pert", response_model=PERTResult)
def pert_analysis(tasks: List[TaskIn]):
    if not tasks:
        raise HTTPException(status_code=400, detail="No tasks provided")

    # Step 1: Calculate expected durations
    durations: Dict[str, float] = {}
    for t in tasks:
        durations[t.id] = compute_expected(t.optimistic, t.most_likely, t.pessimistic)

    # Step 2: Build dependency graph
    G = nx.DiGraph()
    for t in tasks:
        G.add_node(t.id)
    for t in tasks:
        for dep in t.dependencies:
            if dep not in durations:
                raise HTTPException(status_code=400, detail=f"Dependency {dep} not found for task {t.id}")
            G.add_edge(dep, t.id)

    # Step 3: Check for cycles (invalid dependencies)
    if not nx.is_directed_acyclic_graph(G):
        raise HTTPException(status_code=400, detail="Dependencies contain a cycle!")

    # Step 4: Forward pass (Earliest Start / Finish)
    ES, EF = {}, {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if preds:
            ES[node] = max(EF[p] for p in preds)
        else:
            ES[node] = 0.0
        EF[node] = ES[node] + durations[node]

    project_duration = max(EF.values())

    # Step 5: Backward pass (Latest Finish / Start)
    LS, LF = {}, {}
    for node in reversed(list(nx.topological_sort(G))):
        succs = list(G.successors(node))
        if succs:
            LF[node] = min(LS[s] for s in succs)
        else:
            LF[node] = project_duration
        LS[node] = LF[node] - durations[node]

    # Step 6: Slack + timings
    timings = []
    for node in nx.topological_sort(G):
        slack = LS[node] - ES[node]
        timings.append(TaskTiming(
            id=node,
            duration=durations[node],
            ES=ES[node],
            EF=EF[node],
            LS=LS[node],
            LF=LF[node],
            slack=slack
        ))


    # Step 7: Critical Path (tasks with 0 slack)
    critical_path = [t.id for t in timings if t.slack == 0.0]
    mc_result = run_monte_carlo(tasks, iterations=1000)

    return PERTResult(
        project_duration=project_duration,
        task_timings=timings,
        critical_paths=[critical_path],
        monte_carlo=mc_result
    )
    

  # in memory storage
stored_tasks: List[TaskIn] = []


# Endpoint to store tasks
@app.post("/set-tasks")
def set_tasks(tasks: List[TaskIn]):
    global stored_tasks
    stored_tasks = tasks
    return {"message": "Tasks stored successfully", "task_count": len(stored_tasks)}


# Endpoint to update a task
@app.put("/update-task")
async def update_task(updated: TaskIn):
    global stored_tasks
    for i, task in enumerate(stored_tasks):
        if task.id == updated.id:
            stored_tasks[i] = updated
            # recalc after update
            result = pert_analysis(stored_tasks)
            # broadcast to WebSocket clients
            await manager.broadcast(result.dict())
            return {"message": f"Task {updated.id} updated successfully"}
    raise HTTPException(status_code=404, detail=f"Task {updated.id} not found")


## Compare Pert endpoint
@app.post("/compare-pert")
def compare_pert():
    global stored_tasks
    if not stored_tasks:
        raise HTTPException(status_code=400, detail="No tasks available")

    tasks = stored_tasks  # alias for readability

    # --- Classical PERT expected duration ---
    classical_durations = {
        t.id: (t.optimistic + 4 * t.most_likely + t.pessimistic) / 6
        for t in tasks
    }

    completed = {}
    remaining = set(classical_durations.keys())

    while remaining:
        for tid in list(remaining):
            task = next(t for t in tasks if t.id == tid)
            deps = task.dependencies
            if all(dep in completed for dep in deps):
                max_dep = max([completed[dep] for dep in deps], default=0)
                completed[tid] = max_dep + classical_durations[tid]
                remaining.remove(tid)

    classical_total = max(completed.values())

    # --- Enhanced PERT (Monte Carlo Simulation) ---
    num_simulations = 1000
    total_times = []

    for _ in range(num_simulations):
        sampled = {
            t.id: np.random.triangular(t.optimistic, t.most_likely, t.pessimistic)
            for t in tasks
        }
        completed_sim = {}
        remaining_sim = set(sampled.keys())

        while remaining_sim:
            for tid in list(remaining_sim):
                task = next(t for t in tasks if t.id == tid)
                deps = task.dependencies
                if all(dep in completed_sim for dep in deps):
                    max_dep = max([completed_sim[dep] for dep in deps], default=0)
                    completed_sim[tid] = max_dep + sampled[tid]
                    remaining_sim.remove(tid)

        total_times.append(max(completed_sim.values()))

    # Compute statistics from Monte Carlo
    enhanced_mean = np.mean(total_times)
    enhanced_p10 = np.percentile(total_times, 10)
    enhanced_p90 = np.percentile(total_times, 90)

    return {
        "classical_pert": {
            "expected_duration": round(classical_total, 2)
        },
        "enhanced_pert": {
            "mean_duration": round(enhanced_mean, 2),
            "p10": round(enhanced_p10, 2),
            "p90": round(enhanced_p90, 2)
        },
        "comparison": {
            "difference_in_days": round(enhanced_mean - classical_total, 2),
            "interpretation": (
                "Enhanced PERT accounts for uncertainty with probability ranges, "
                "while classical PERT gives only a fixed expected value."
            )
        }
    }



#  WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

#  WebSocket Endpoint
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keeps connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/")
def home():
    return {"message": "PERT API is running with calculations!"}
