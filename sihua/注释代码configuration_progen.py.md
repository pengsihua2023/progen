## 代码注释configuration_progen.py

```
# coding=utf-8
# 采用UTF-8编码方式，保证代码支持国际化字符。

# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
# 版权信息，归EleutherAI和HuggingFace团队所有。

# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件遵循Apache License 2.0许可协议。

# you may not use this file except in compliance with the License.
# 只有在遵守许可协议的情况下，才能使用该文件。

# You may obtain a copy of the License at
# 许可协议的链接。

#     http://www.apache.org/licenses/LICENSE-2.0
# Apache License 2.0的完整文本链接。

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 除非法律要求或书面同意，按照“原样”基础分发本软件。

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何形式的明示或暗示的保证。

# See the License for the specific language governing permissions and
# 许可证详细说明了权限和限制。

# limitations under the License.
# 许可证下的限制。

# Modified configuration implementation based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/configuration_gptj.py
# 基于HuggingFace团队的GPT-J配置实现的修改版。

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
# 导入transformers包的相关类和模块。

logger = logging.get_logger(__name__)
# 创建一个用于当前模块的日志记录器。

class ProGenConfig(PretrainedConfig):
    model_type = "progen"
    # 定义模型类型为"progen"。

    def __init__(
        self,
        vocab_size=50400,  # 词汇表大小。
        n_positions=2048,  # 模型可以接受的最大序列长度。
        n_ctx=2048,  # 上下文大小，通常与n_positions相同。
        n_embd=4096,  # 嵌入层大小。
        n_layer=28,  # 模型中的层级数量。
        n_head=16,  # 注意力机制的头数。
        rotary_dim=64,  # 旋转位置编码的维度。
        n_inner=None,  # 内层维度，如果为None，通常是n_embd的四倍。
        activation_function="gelu_new",  # 激活函数。
        resid_pdrop=0.0,  # 残差连接的dropout比率。
        embd_pdrop=0.0,  # 嵌入层的dropout比率。
        attn_pdrop=0.0,  # 注意力层的dropout比率。
        layer_norm_epsilon=1e-5,  # 层归一化的小数值以防除零错误。
        initializer_range=0.02,  # 初始化权重的范围。
        scale_attn_weights=True,  # 是否缩放注意力权重。
        gradient_checkpointing=False,  # 是否使用梯度检查点以节省内存。
        use_cache=True,  # 是否缓存注意力计算结果以加快解码速度。
        bos_token_id=50256,  # 文本开始标记的ID。
        eos_token_id=50256,  # 文本结束标记的ID。
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # 调用父类的构造函数并传入部分参数。

        # 将所有传入的参数设置为类的成员变量。
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def max_position_embeddings(self):
        return self.n_positions
        # 返回最大位置嵌入数量的属性。

    @property
    def hidden_size(self):
        return self.n_embd
        # 返回隐藏层大小的属性。

    @property
    def num_attention_heads(self):
        return self.n_head
        # 返回注意力头数的属性。

    @property
    def num_hidden_layers(self):
        return self.n_layer
        # 返回隐藏层的数量的属性。


```
