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
    def __init__(self, log_dir: str | Path, args):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        t = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        file_name_prefix = f'[{args.agent}][{args.task.replace(':', '_')}][{args.seed}][{t}][Logger][{"".join(random.choices(string.ascii_uppercase, k=6))}]'
        
        json_log_name = file_name_prefix + '.json'
        jsonl_log_name = file_name_prefix + 'metrics.jsonl'
        
        self._json_log_path = self._log_dir / json_log_name
        self._jsonl_path = self._log_dir / jsonl_log_name
        
        self._jsonl_file = open(
            self._jsonl_path, 'a', buffering=1, encoding='utf-8'
        )
        
        self._start_time = time.time()
        
        self.steps = []
        self.scores = []
        
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
        
        if len(record) > 2:
            self._jsonl_file.write(json.dumps(record) + '\n')
            
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
        path = self._log_dir / 'config.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def close(self) -> None:
        res = {
            'Steps': self.steps,
            'Score': self.scores
        }
        with open(self._json_log_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(res))
        
        self._jsonl_file.close()

def _to_scalar(v: Any) -> float | int | None:
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return v.item()
    if isinstance(v, np.ndarray) and v.size == 1:
        return float(v.flat[0])
    return None