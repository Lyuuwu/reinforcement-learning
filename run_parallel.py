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
    p.add_argument('--max', type=int, default=None, help='max number of concurrent running process')
    p.add_argument('--start-seed', type=int, default=1000)
    p.add_argument('--save_dir', default='runs/')
    
    p.add_argument('passthrough', nargs=argparse.REMAINDER)
    
    return p.parse_args()

def make_cmd(args, seed, extra):
    return [
        sys.executable, '-u', 'train.py',
        '--agent', args.agent,
        '--task', args.task,
        '--seed', str(seed),
        '--save_dir', args.save_dir,
        *extra
    ]

def launch(cmd, log_path, env, seed):
    log_f = open(log_path, 'w', buffering=1)
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    print(f'[Launch] seed={seed}  log -> {log_path}')
    return p, log_f

def main():
    args = parse_args()
    max_concurrent = args.max or args.n
    
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
    
    seeds = [args.start_seed + i for i in range(args.n)]
    seed_queue = list(seeds)

    running = []

    try:
        while seed_queue or running:
            while seed_queue and len(running) < max_concurrent:
                seed = seed_queue.pop(0)
                log_path = log_dir / f'[{args.agent}][{task_safe}][{seed}].log'
                cmd = make_cmd(args, seed, extra)
                p, log_f = launch(cmd, log_path, env, seed)
                running.append((p, log_f, seed))
                time.sleep(1.0)

            still_running = []
            for p, log_f, seed in running:
                rc = p.poll()
                if rc is None:
                    still_running.append((p, log_f, seed))
                else:
                    log_f.close()
                    tag = 'OK' if rc == 0 else f'FAIL(rc={rc})'
                    print(f'[Done]   seed={seed}  {tag}')
            running = still_running

            if seed_queue and len(running) >= max_concurrent:
                time.sleep(2.0)
    except KeyboardInterrupt:
        print('\n[Interrupt] terminate all sub process ...')
        for p, log_f, _ in running:
            if p.poll() is None:
                p.terminate()
        for p, log_f, _ in running:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
            log_f.close()
            
if __name__ == '__main__':
    main()