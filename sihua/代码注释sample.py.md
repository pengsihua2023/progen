## 代码注释 sample.py
这段代码是一个Python脚本，用于生成使用Transformer模型的文本，特别是基于Salesforce ProGen模型的自回归生成。以下是对脚本各部分的逐行解释：

### 导入模块和文件头部注释

```python
# SPDX-License-Identifier: BSD-3-Clause
# 许可证声明，指定使用BSD-3-Clause许可。

import os
import time
import random
import argparse
import torch
from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
# 导入必要的Python库和模块。
```

### 实用工具类和函数

```python
class print_time:
    def __init__(self, desc):
        self.desc = desc  # 描述当前操作的字符串。

    def __enter__(self):
        print(self.desc)
        self.t = time.time()  # 记录时间开始。

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')  # 打印操作所需时间。
```

```python
def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 设置环境变量以禁用分词器的并行处理。

def set_seed(seed, deterministic=True):
    random.seed(seed)  # 设置随机种子以确保可重复性。
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置环境种子。
    torch.manual_seed(seed)  # 为所有设备设置PyTorch种子。
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为CUDA设备设置种子。
        torch.backends.cudnn.deterministic = deterministic  # 确保CUDA的行为确定性。
        torch.backends.cudnn.benchmark = not deterministic  # 控制是否启用性能优化。
```

### 模型和分词器创建函数

```python
def create_model(ckpt, fp16=True):
    # 根据给定的检查点和精度设置创建模型实例。
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)
```

```python
def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())  # 从文件创建并返回一个Tokenizer实例。
```

### 文本生成和处理函数

```python
def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):
    # 在给定参数下生成文本样本。
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        return tokenizer.decode_batch(as_lists(tokens_batch))

def truncate(sample, terminals):
    # 根据指定的终止符号截断文本。
    pos = [sample.find(terminal, 1) for terminal in terminals if sample.find(terminal, 1) != -1]
    return sample[:(min(pos)+1)] if pos else sample

def cross_entropy(logits, target, reduction='mean'):
    # 计算交叉熵损失。
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)
```

### 主函数和参数解析

```python
def main():
    parser = argparse.ArgumentParser()
    # 添加命令行参数解析。
    # 省略部分参数代码。
    args = parser.parse_args()
    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)
    # 环境设置和种子设置。

    # 模型和分词器加载。
    # 省略部分加载和生成代码。

if __name__ == '__main__':
    main()
    print('done.')
```

**总结**：
此脚本是一个完整的自回归文本生成流程，包括参数解析、环境设置、模型和分词器加载、文本生成和结果处理。脚本使用了PyTorch和自定义的Tokenizer，展示了如何从指定的检查点加载预训练模型，并在指定的设备上执行文本生成任务。脚本也包含了一些调试和验证的代码，确保生成的文本符合预期的性能指标。此外，通过命令行参数提供了高度的可配置性，可以在不同的环境和条件下运行。
