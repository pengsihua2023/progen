## 代码注释 sample.py
### 总结： 
此脚本是一个完整的自回归文本生成流程，包括参数解析、环境设置、模型和分词器加载、文本生成和结果处理。脚本使用了PyTorch和自定义的Tokenizer，展示了如何从指定的检查点加载预训练模型，并在指定的设备上执行文本生成任务。脚本也包含了一些调试和验证的代码，确保生成的文本符合预期的性能指标。此外，通过命令行参数提供了高度的可配置性，可以在不同的环境和条件下运行。  
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
## 单独文件内注释
好的，我将使用中文注释重新组织这段代码，并在每一部分提供详细解释。

```python
# 版权与许可证信息
# SPDX许可证标识符：BSD-3-Clause
# 这个文件是salesforce.com, inc.的一部分，受BSD-3-Clause许可证保护。
# 完整的许可证文本可以在指定的URL找到。

# 导入必要的库
import os
import time
import random
import argparse

import torch  # 用于张量操作和深度学习的PyTorch库
from tokenizers import Tokenizer  # 来自Hugging Face tokenizers库的分词工具
from models.progen.modeling_progen import ProGenForCausalLM  # 导入专门用于因果语言建模的ProGen模型

# 实用工具类和函数
class print_time:
    # 用于计时操作的上下文管理器
    def __init__(self, desc):
        self.desc = desc  # 正在计时的操作描述

    def __enter__(self):
        # 开始计时
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        # 结束计时并打印耗时
        print(f'{self.desc} took {time.time()-self.t:.02f}s')

def set_env():
    # 设置环境变量以关闭分词器并行处理
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def set_seed(seed, deterministic=True):
    # 设置随机种子以确保结果可复现
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic  # 确保CUDA操作确定性
        torch.backends.cudnn.benchmark = not deterministic  # 根据是否需要确定性关闭或开启性能优化

def create_model(ckpt, fp16=True):
    # 创建并返回预训练的ProGen模型，可选使用16位浮点数精度
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)

def create_tokenizer_custom(file):
    # 从文件中加载并返回自定义分词器
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):
    # 生成语言模型的样本输出
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(
            input_ids, do_sample=True, temperature=temp, max_length=max_length,
            top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        return tokenizer.decode_batch(as_lists(tokens_batch))

def truncate(sample, terminals):
    # 根据终止符截断样本
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample

def cross_entropy(logits, target, reduction='mean'):
    # 计算交叉熵损失
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)

# 主函数定义
def main():
    # 定义常量和模型配置
    models_151M = ['progen2-small']
    models_754M = ['progen2-medium', 'progen2-oas', 'progen2-base']
    models_2B = ['progen2-large', 'progen2-BFD90']
    models_6B = ['progen2-xlarge']
    models = models_151M + models_754M + models_2B + models_6B
    
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-large')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    parser.add_argument('--sanity', default=True, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    # 环境设置和种子设定
    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    ckpt = f'./checkpoints/{args.model}'

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # 模型与分词器加载
    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)

    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

    # 校验模式
    if args.sanity:
        with print_time('sanity cross-entropy'):
            def ce(tokens):
                # 计算给定文本的交叉熵损失
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                        logits = model(target, labels=target).logits

                        # 数据偏移处理
                        logits = logits[:-1, ...]
                        target = target[1:]

                        return cross_entropy(logits=logits, target=target).item()

            # 根据不同模型预设的检查点和目标交叉熵
            checkpoint_x_ce = {
                'progen2-small': (x_uniref90bfd30, 2.4),
                'progen2-medium': (x_uniref90bfd30, 1.9),
                'progen2-base': (x_uniref90bfd30, 1.9),
                'progen2-large': (x_uniref90bfd30, 1.8),
                'progen2-xlarge': (x_uniref90bfd30, 1.0),
                'progen2-oas': (x_oas, 0.3),
                'progen2-BFD90': (x_bfd90, 1.3),
            }

            ce_eval = ce(checkpoint_x_ce[args.model][0])
            ce_target = checkpoint_x_ce[args.model][1]

            print(ce_target, ce_eval, abs(ce_eval - ce_target))

            assert abs(ce_eval - ce_target) < 0.1

    # 样本生成与输出
    with print_time('sampling'):
        completions = sample(device=device, model=model, tokenizer=tokenizer, context=args.context, pad_token_id=tokenizer.encode('').ids[0], num_return_sequences=args.num_samples, temp=args.t, top_p=args.p, max_length=args.max_length)
        truncations = [truncate(completion, terminals=['1', '2']) for completion in completions]

        print(args.context)

        for (i, truncation) in enumerate(truncations):
            print()
            print(i)
            print(truncation)
            

if __name__ == '__main__':
    main()
    print('done.')
```

这个代码是一个包含多部分功能的脚本，主要用于处理语言模型的训练和生成。它涵盖了环境设置、模型初始化、参数解析、模型和分词器加载、交叉熵计算检验以及样本生成等功能。此代码使用PyTorch框架和Hugging Face的分词器库，旨在提供灵活的实验环境和性能测试，特别适用于高级自然语言处理任务。  
