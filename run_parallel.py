import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description='parallel run multiple seeds')
    p.add_argument('--agent', type=str, required=True)
    p.add_argument('--task', type=str, required=True)
    p.add_argument('--n', type=int, default=3, help='the number of process')
    p.add_argument('--start-seed', type=int, default=1000)
    p.add_argument('--save_dir', default='runs/')
    
    p.add_argument('passthrough', nargs=argparse.REMAINDER)
    
    return p.parse_args()

def main():
    args = parse_args()
    
    task_safe = args.task.replace(':', '_')
    
    # --- log dir ---
    log_dir = Path(args.save_dir) / f'[{args.agent}]' / f'[{args.agent}][{task_safe}]'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 限制 thread 數量 ---
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    
    # --- 去除 '--'
    extra = [a for a in args.passthrough if a != '--']
    
    procs = []
    for i in range(args.n):
        seed = args.start_seed + i
        
        log_path = log_dir / f'[{args.agent}][{task_safe}][{seed}].log'
        log_f = open(log_path, 'w', buffering=1)
        
        cmd = [
            sys.executable, '-u', 'train.py',
            '--agent', args.agent,
            '--task', args.task,
            '--seed', str(seed),
            '--save_dir', args.save_dir,
            *extra
        ]
        
        print(f'[Launch] seed={seed}  log -> {log_path}')
        
        p = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env
        )
        
        procs.append((p, log_f, seed))
        time.sleep(1.0)
    
    try:
        for p, log_f, seed in procs:
            rc = p.wait()
            log_f.close()
            tag = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            print(f'[Done] seed={seed}  {tag}')
    except KeyboardInterrupt:
        print('\n[Inpterrupt] terminate all sub process ...')
        
        for p, log_f, _ in procs:
            if p.poll() is None:
                p.terminate()
        
        for p, log_f, _ in procs:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill()
            log_f.close()
            
if __name__ == '__main__':
    main()