import random
import string
import json
import time
import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

class JSONLLogger:
    def __init__(self, run_dir: str | Path, agent: str, task: str, seed: int):
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        
        task_safe = task.replace(':', '_')
        t = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        uid = ''.join(random.choices(string.ascii_uppercase, k=6))
        
        self._basename = f'[{agent}][{task_safe}][{seed}][{t}]'
        self.run_tag   = f'{self._basename}[Logger][{uid}]'
        
        self._metrics_path = self._run_dir / f'{self._basename}metrics.jsonl'
        self._eval_path    = self._run_dir / f'{self._basename}.json'
        self._config_path  = self._run_dir / f'{self._basename}config.json'
        
        self._metrics_file = open(self._metrics_path, 'a', buffering=1, encoding='utf-8')
        self._start_time = time.time()
        self.steps, self.scores = [], []
        
    @property
    def run_dir(self) -> Path:
        return self._run_dir
        
    def log(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str=''
    ) -> None:
        record: dict[str, Any] = {
            'step': step,
            'wall_time': round(time.time() - self._start_time, 2)
        }
        
        for k, v in metrics.items():
            tag = f'{prefix}/{k}' if prefix else k
            scalar = _to_scalar(v)
            if scalar is not None:
                record[tag] = scalar
        
        self._metrics_file.write(json.dumps(record) + '\n')
            
    def log_print(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str=''
    ) -> None:
        ''' log and print '''
        self.log(metrics, step, prefix)
        parts = [f'{prefix} step={step}'] if prefix else [f'step={step}']
        
        for k, v in metrics.items():
            s = _to_scalar(v)
            if s is not None:
                parts.append(f'{k}={s:.4f}' if isinstance(s, float) else f'{k}={s}')
        print('=' * 60)
        print(' | '.join(parts))

    def eval_log(self, step: int, score: float):
        self.steps.append(step)
        self.scores.append(score)

    def save_config(self, config_dict: dict) -> None:
        config_dict = {'run_tag': self.run_tag, **config_dict}
        with open(self._config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def close(self) -> None:
        with open(self._eval_path, 'w', encoding='utf-8') as f:
            json.dump(
                {'run_tag': self.run_tag, 'steps': self.steps, 'scores': self.scores},
                f
            )
        self._metrics_file.close()

def _to_scalar(v: Any) -> float | int | None:
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return v.item()
    if isinstance(v, np.ndarray) and v.size == 1:
        return float(v.flat[0])
    return None