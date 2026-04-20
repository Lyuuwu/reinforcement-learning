# CILAB RL

## usage

> python --agent <agent_name> --task <{domain_name}:{task_name}>

### format

- agent: agents 底下資料夾的名稱，一定要相同
- task: `{domain_name}:{task_name}` 如: `mujoco:Ant` 或是 `dm_conrol:cheetah-run` (注意大小寫)

### optional flags

- train_type: 預設 off-policy
- config: 實驗檔案的配置，輸入名稱，程式會抓取 experiments/{name}.py 中的 build() 來建構 config
預設沒有 (None)
- seed: 種子種子, 預設 0
- num_envs: 預設 1
- device: 預設 auto (可選 cpu, cuda, cuda:0)
- save_dir: 預設 (runs/default) 儲存實驗結果 與 checkpoint 的資料夾
- resume: 載入 checkpoint 的路徑，預設 None，可選 auto 或自訂路徑
- agent-override: 修改 agent config 的參數
- env-override: 修改 env config 的參數
- trainer-override: 修改 trainer config 的參數

## Add Agent

在 agents 資料夾底下新增資料夾，名稱為 agent 的名稱

檔案只要求一定要有 `__init__.py`

封裝好的 agent 一定要繼承 `base.AgentBase`
建構 agent 的程式要回傳 `base.AgentBase` 的類別的 class
設定 Agent 參數用一個 `dataclass` 封裝，並且一定要繼承 `base.BaseConfig`

### AgentBase

你要 override 以下的 method

1. sample(self, obs): 回傳 action (for train)
2. act(self, obs):    回傳 action (for eval)
3. update(self, batch): 輸入一個 batch 的資料，定義網路如何更新參數

### BaseConfig

- 讓 dataclass 可以轉換成 dict
- 支援使用 `.get` 來呼叫成員，不存在則回傳 default (預設 None) `.get(name, default = None)`
- 支援使用 override，將要 override 的參數用一個 `dict` 封裝，接著傳入參數
範例: `my_config.override(override_dict)`

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

## 未來工作

- 寫一個 .sh 或是 .py 的腳本進行多 process 的執行
- 修改 Logger 的儲存方式，以及寫一個好用的畫圖程式
