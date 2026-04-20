# CILAB RL

## usage

> python --agent <agent_name> --task <task_name>

### format

- agent: agents 底下資料夾的名稱，一定要相同
- task: `{domain_name}:{task_name}` 如: `mujoco:Ant` 或是 `dm_conrol:HalfCheetah-run`

### optional flags

- train_type: 預設 off-policy
- config: 實驗檔案的配置，輸入名稱，程式會抓取 experiments/{name}.py 中的 build() 來建構 config
- seed: 種子種子
- num_envs: 預設 1
- device: 預設 auto (可選 cpu, cuda, cuda:0)
- save_dir: 預設 (runs/default) 儲存實驗結果 與 checkpoint 的資料夾
- resume: 載入 checkpoint 的路徑，預設 None，可選 auto 或自訂路徑
- agent-override: 修改 agent config 的參數
- env-override: 修改 env config 的參數
- trainer-override: 修改 trainer config 的參數

## Add Agent

在 agents 資料夾底下新增資料夾，名稱為 agent 的名稱

只要求一定要有 `__init__.py`
剩下愛幹嘛就幹嘛

### `__init__.py` 內容

將 agent config import 進來，並且命名為 `Config`
將 建構 agent 的 function import 進來，並且命名為 `build`

然後加入 `__all__`

範例:

```python
# agents/TQC/__init__.py
from .config import TQCConfig as Config
from .builder import build

__all__ = ['Config', 'build']
```

## Logger 存檔

目前一坨，很亂，之後再改