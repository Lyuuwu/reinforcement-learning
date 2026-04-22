import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def load_runs(root: str, agent: str, task: str) -> list[dict]:
    task_safe = task.replace(':', '_')
    base = Path(root) / agent / task_safe
    runs = []
    
    for seed_dir in sorted(base.glob('seed_*')):
        path = seed_dir / 'eval_curve.json'
        if path.exists():
            with open(path) as f:
                runs.append(json.load(f))
    
    return runs

def plot_comparison(root: str, agents: list[str], task: str, ax=None):
    ax = ax or plt.gca()
    
    for agent in agents:
        runs = load_runs(root, agent, task)
        if not runs:
            continue
        
        steps = np.array(runs[0]['steps'])
        scores = np.array([r['scores'] for r in runs])
        mean, std = scores.mean(axis=0), scores.std(axis=0)
        ax.plot(steps, mean, label=agent)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    
    ax.set(title=task, xlabel='Steps', ylabel='eval return')
    ax.legend()