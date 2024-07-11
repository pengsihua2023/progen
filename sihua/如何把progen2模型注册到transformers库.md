## 如何把progen2模型注册到transformers库?
```
from transformers import AutoModel, AutoConfig
from transformers import AutoConfig, PretrainedConfig

import sys
sys.path.insert(0, 'D:/progen/progen/progen2')
from models.progen.modeling_progen import ProGenForCausalLM
from models.progen.configuration_progen import ProGenConfig

# 注册模型和配置
AutoModel.register(ProGenForCausalLM, "progen")
#AutoModel.register("progen", ProGenForCausalLM)
#AutoConfig.register(ProGenConfig, "progen")
AutoConfig.register("progen", ProGenConfig)
```
