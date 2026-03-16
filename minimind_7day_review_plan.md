# MiniMind 一周学习清单（面向考研复试）

> **使用说明**：本清单面向"把 MiniMind 包装成复试简历项目"的同学，每天 2~3 小时，7 天后具备在复试中自然回答老师追问的能力。
> 项目仓库结构参考：`model/`、`trainer/`、`scripts/`、`eval_llm.py`、`dataset/`。

---

## Day 1 — 项目全局认知：MiniMind 是什么、为什么存在

### 1. 今日学习目标

理解 MiniMind 项目的整体定位、核心价值和文件结构，能在 30 秒内向老师介绍这个项目是什么、解决了什么问题。

### 2. 必看源码文件

- `README.md` — 通读全文，重点记住：参数量（26M/512hidden/8层）、训练成本（单卡3090）、全链路覆盖范围
- `model/model_minimind.py` — 只看第 1~80 行，看懂 `MiniMindConfig` 的字段（hidden_size、num_attention_heads、vocab_size、use_moe 等）
- `requirements.txt` — 扫一眼，记住核心依赖：`torch`、`transformers`、`datasets`

### 3. 必须掌握的技术点

| 技术点 | 掌握程度 |
|---|---|
| MiniMind 参数规模（26M）及与 GPT-3（175B）的对比 | 必须能说出数字 |
| Decoder-only 架构的概念（只有解码器，自回归生成） | 理解概念即可 |
| 项目训练全链路：Pretrain → SFT → LoRA/DPO/RLHF | 能说出每步的目的 |
| vocab_size=6400（自训练 tokenizer） | 记住这个数字 |
| 配置文件参数的作用（hidden_size、num_layers、num_heads） | 知道每个参数控制什么 |

### 4. 面试表达模板

> **"请简单介绍一下你的项目"**

*"MiniMind 是一个基于 PyTorch 原生实现的极小参数量语言模型，参数约 26M，架构参考 LLaMA 的 Decoder-only Transformer。项目覆盖了从 Tokenizer 训练、预训练、监督微调（SFT）到 LoRA、DPO、RLHF 的完整训练流程，可在单卡 3090 上完成训练，主要面向教学和理解大模型内部机制，而非工业部署。"*

### 5. 今日最低成果

- 能默写出 MiniMindConfig 中 5 个以上关键参数及其含义
- 能用一段话介绍项目（参考上面模板）
- 能画出项目文件夹结构（model/、trainer/、scripts/、dataset/）

---

## Day 2 — Transformer 核心结构：Attention 与 Decoder-only

### 1. 今日学习目标

理解 MiniMind 模型的核心结构——Multi-Head Attention、Causal Mask、以及为什么选用 Decoder-only，能在老师追问时说清楚原理。

### 2. 必看源码文件

- `model/model_minimind.py` — 重点看以下类（按顺序）：
  - `Attention` 类：理解 Q/K/V 的计算、GQA（分组查询注意力）、softmax(QK^T/sqrt(d))V
  - `apply_rotary_emb` 函数：理解 RoPE 位置编码的应用方式（不必推导公式）
  - Causal Mask 的生成逻辑（`torch.tril` 部分）

> ⚠️ **只需看懂逻辑流程，不必逐行理解每个张量操作**

### 3. 必须掌握的技术点

| 技术点 | 掌握程度 |
|---|---|
| Self-Attention 的计算步骤（Q、K、V、softmax） | 能口头描述流程 |
| Causal Mask 的作用（防止看到未来 token） | 必须能解释 |
| 为什么用 Decoder-only（生成任务不需要 Encoder） | 能说出理由 |
| GQA（分组查询注意力）对比标准 MHA 的区别 | 知道概念：减少 KV 头数以节省显存 |
| RoPE（旋转位置编码）的直觉理解 | 位置信息以旋转矩阵形式编入 Q/K，支持外推 |

### 4. 面试表达模板

> **"为什么用 Decoder-only 架构？"**

*"MiniMind 是语言生成模型，目标是给定前缀、预测下一个 token，这天然是自回归任务。Decoder-only 架构通过 Causal Mask 保证每个 token 只能看到它之前的 token，符合生成任务的需求，同时省去了 Encoder 和 Cross-Attention，结构更简单，参数利用率更高。GPT 系列和 LLaMA 都采用这种设计。"*

> **"RoPE 是什么，为什么用它？"**

*"RoPE 是旋转位置编码，它把位置信息以旋转矩阵的形式编入 Query 和 Key 向量，而不是直接加入 Embedding。好处是：位置信息在计算注意力时自然体现，并且支持比训练长度更长的序列外推，这对需要处理长文本的场景很重要。MiniMind 的 rope_theta 设为 1000000，支持长上下文。"*

### 5. 今日最低成果

- 能用自己的话解释 Causal Mask 的作用
- 能说出 RoPE 相比 Sinusoidal 位置编码的优势
- 找到源码中 Attention 计算的核心一行（`scores = Q @ K.transpose(-2,-1)`）

---

## Day 3 — 归一化与激活函数：RMSNorm 和 SwiGLU

### 1. 今日学习目标

理解 MiniMind 中使用的 RMSNorm 和 SwiGLU 激活函数，能解释为什么这两个组件比传统 LayerNorm + ReLU 更适合现代 LLM。

### 2. 必看源码文件

- `model/model_minimind.py` — 重点看：
  - `RMSNorm` 类（约 10 行）：公式理解，`x / rms(x) * weight`
  - `FeedForward`（或 `MLP`）类：看清楚门控结构 `silu(gate) * up`，以及 down 的线性变换
  - `MiniMindBlock` 类：看清楚"Pre-Norm"结构（先 Norm 再 Attention/FFN）

> ✅ **这部分代码量小，必须看懂源码逻辑**

### 3. 必须掌握的技术点

| 技术点 | 掌握程度 |
|---|---|
| RMSNorm 公式：`x / sqrt(mean(x²) + ε) * γ` | 能写出公式 |
| RMSNorm vs LayerNorm 的区别 | RMSNorm 省去了均值中心化，计算更快 |
| SwiGLU 激活：`SiLU(gate) * up` | 能描述门控机制 |
| Pre-Norm vs Post-Norm | Pre-Norm 在现代 LLM 中更稳定，训练更易收敛 |
| intermediate_size 的计算方式（通常是 hidden_size 的 ~2.67 倍） | 知道这个经验值 |

### 4. 面试表达模板

> **"为什么用 RMSNorm 而不是 LayerNorm？"**

*"LayerNorm 需要同时计算均值和方差，而 RMSNorm 只计算均方根，省去了均值中心化步骤。在大模型中，这能带来约 10~15% 的计算加速，同时实验表明效果基本持平。LLaMA 系列也采用了相同的设计，MiniMind 跟随了这一主流做法。"*

> **"SwiGLU 是什么？"**

*"SwiGLU 是一种门控激活函数，FFN 层有两个线性变换：一个经过 SiLU 激活函数作为'门'，另一个作为'内容'，两者逐元素相乘后再通过下投影。相比 ReLU，SwiGLU 能让模型更好地控制信息流动，在 GPT-4、LLaMA 等模型中被广泛使用。"*

### 5. 今日最低成果

- 能手写出 RMSNorm 的 forward 逻辑（5 行以内）
- 能描述 SwiGLU 的两个分支
- 能说出 Pre-Norm 相比 Post-Norm 的优势

---

## Day 4 — 预训练（Pretrain）：从无到有学语言

### 1. 今日学习目标

理解预训练的目标、数据格式、训练逻辑和损失函数，能解释"预训练在做什么"以及它与 SFT 的本质区别。

### 2. 必看源码文件

- `trainer/train_pretrain.py` — 重点看：
  - 数据加载部分：数据集格式（纯文本 token 序列）
  - 训练循环：`loss = cross_entropy(logits, labels)`
  - `labels` 的构造方式（通常是 `input_ids` 右移一位）
  - 学习率调度（cosine decay 或 warmup）
- `trainer/trainer_utils.py` — 看 loss 计算相关的工具函数

> ⚠️ **只需理解数据流和 loss 计算，优化器细节不必深挖**

### 3. 必须掌握的技术点

| 技术点 | 掌握程度 |
|---|---|
| 预训练目标：Next Token Prediction（自回归语言模型） | 必须能解释 |
| 交叉熵损失的含义（预测概率分布与真实 token 的差距） | 能口头解释 |
| Pretrain 数据是无监督的（无需人工标注，用原始文本） | 理解这个核心特点 |
| Pretrain 使学模型习语言规律，但不会"听指令" | 理解 Pretrain 的局限性 |
| batch_size、learning_rate、gradient_accumulation 的作用 | 知道各自控制什么 |

### 4. 面试表达模板

> **"Pretrain 阶段在做什么？"**

*"预训练阶段的目标是 Next Token Prediction，即给定前面的 token，预测下一个 token，损失函数是交叉熵。训练数据是大量无标注的文本，模型通过这个过程学习语言的统计规律、语法、事实知识等。MiniMind 的预训练数据来自中文语料，vocab_size 为 6400，训练完成后模型具备基本的语言能力，但还不会按照人类指令回答问题。"*

> **"Pretrain 和 SFT 的根本区别是什么？"**

*"Pretrain 是让模型学会'说话'，目标是预测下一个 token，数据是无监督的原始文本。SFT 是让模型学会'听指令'，数据是人工构造的指令-回答对（Instruction-Response），在 Pretrain 好的基础上进行微调。SFT 数据量远小于 Pretrain，但效果见效更快，因为模型已经有了语言基础，只需要学'对话格式'。"*

### 5. 今日最低成果

- 能在代码中找到 labels 的构造方式（右移一位）
- 能说出 Pretrain 的 loss 函数名称和含义
- 能解释为什么 Pretrain 数据不需要人工标注

---

## Day 5 — SFT 与 LoRA：让模型听话、高效微调

### 1. 今日学习目标

理解监督微调（SFT）的数据格式和训练逻辑，以及 LoRA 如何在极少参数下实现高效微调，这是复试中最容易被追问的两个点。

### 2. 必看源码文件

- `trainer/train_full_sft.py` — 重点看：
  - 数据格式：`[system_prompt, user_message, assistant_response]` 的拼接方式
  - `loss_mask`：只对 assistant 回答部分计算 loss（这是 SFT 的关键细节）
- `trainer/train_lora.py` — 重点看：
  - LoRA 模块如何被注入到原始模型（哪些层被替换）
  - 训练时哪些参数被冻结（`requires_grad=False`）
- `model/model_lora.py` — 必看：
  - `LoRALinear` 类：理解 `W = W0 + A @ B * (alpha/r)` 的实现

> ✅ **LoRA 的实现代码量小，必须看懂 `LoRALinear.forward()` 的逻辑**

### 3. 必须掌握的技术点

| 技术点 | 掌握程度 |
|---|---|
| SFT 数据格式：Instruction + Response 对 | 必须能描述 |
| loss_mask 只对 response 计算 loss 的原因 | 必须能解释（不让模型"学"提问部分） |
| LoRA 原理：`ΔW = A @ B`，A 和 B 是低秩矩阵 | 能写出数学表达式 |
| LoRA 为什么节省显存：只有 A、B 有梯度，原始权重冻结 | 必须能解释 |
| LoRA 的两个超参：rank（r）和 alpha（缩放系数） | 知道各自的作用 |
| LoRA 训练后如何合并权重（merge_weights） | 知道可以合并，部署时无额外开销 |

### 4. 面试表达模板

> **"LoRA 为什么能节省显存？"**

*"LoRA 的核心思想是：大模型在微调时权重的变化量 ΔW 是低秩的，因此可以用两个小矩阵 A 和 B 来近似，ΔW = A @ B。训练时原始权重 W0 被冻结，只有 A 和 B 参与梯度更新。以 hidden_size=512 为例，原始权重是 512×512=262144 个参数，而 LoRA rank=8 时只有 512×8 + 8×512=8192 个可训练参数，减少了 97% 的显存占用。"*

> **"SFT 中为什么只对 response 部分计算 loss？"**

*"SFT 的目标是让模型学会生成符合指令的回答，而不是学会生成用户的提问。如果对整个序列（包括 instruction）计算 loss，模型会分散注意力去'预测提问'，而我们只希望它学习'如何回答'。因此通过 loss_mask 将 instruction 部分的 loss 置零，只对 response 部分反向传播。"*

### 5. 今日最低成果

- 能画出 LoRA 的结构图（原始路径 + 低秩旁路）
- 能在 `model/model_lora.py` 中找到 `forward` 函数并解释
- 能说出 loss_mask 在 SFT 代码中的位置和作用

---

## Day 6 — Tokenizer 与推理：端到端理解

### 1. 今日学习目标

理解 MiniMind 的自训练 Tokenizer 和推理过程（包括 KV Cache、采样策略），能解释推理时的完整数据流。

### 2. 必看源码文件

- `trainer/train_tokenizer.py` — 只看训练逻辑大纲：BPE 算法、vocab_size=6400
- `model/tokenizer.json` — 看结构（special tokens、merges 规则），不必深读
- `scripts/chat_openai_api.py` 或 `scripts/web_demo.py` — 看推理调用的接口
- `model/model_minimind.py` — 找到 `generate` 函数或推理时的 forward 逻辑，理解 KV Cache 的使用

> ⚠️ **KV Cache 只需理解概念，不必看懂每行实现细节**

### 3. 必须掌握的技术点

| 技术点 | 掌握程度 |
|---|---|
| BPE（字节对编码）的基本原理 | 能用一句话解释（高频子词合并） |
| 为什么用自训练 Tokenizer（vocab_size 小，适合中文） | 能解释 |
| 推理时的自回归过程（逐 token 生成） | 必须能描述 |
| KV Cache 的作用（避免重复计算历史 token 的 K/V） | 能解释节省了什么计算 |
| 采样策略：temperature、top-p（nucleus sampling） | 知道各自的效果 |
| max_new_tokens 参数的意义 | 控制生成长度上限 |

### 4. 面试表达模板

> **"模型推理时是怎么生成文本的？"**

*"推理时采用自回归方式：首先将输入文本经过 Tokenizer 转化为 token id 序列，送入模型得到 logits，从 logits 中采样（通过 temperature 控制随机性、top-p 过滤低概率词）得到下一个 token，将其拼接到输入序列后再次送入模型，如此循环直到生成 EOS token 或达到最大长度。为了加速，使用 KV Cache 缓存历史 token 的 Key 和 Value，避免重复计算。"*

> **"KV Cache 为什么能加速推理？"**

*"在自回归生成中，每生成一个新 token，都需要计算它和所有历史 token 的 Attention。如果不缓存，每次都要对整个序列重新计算 K 和 V，计算量是 O(n²)。KV Cache 把历史 token 的 K 和 V 保存下来，新 token 只需计算自己的 Q，然后和缓存的 K/V 做 Attention，将逐步生成的时间复杂度从 O(n²) 降为 O(n)。"*

### 5. 今日最低成果

- 能用 3 句话描述 BPE 的训练过程
- 能画出推理时自回归生成的流程图（token→model→logits→sample→next token）
- 能解释 KV Cache 节省了哪部分计算

---

## Day 7 — 项目包装与复试演练：整合所有知识

### 1. 今日学习目标

整合前 6 天的知识，构建完整的"项目介绍叙事"，练习回答老师可能追问的各类问题，并准备"我做了什么"的包装话术。

### 2. 必看源码文件

- `eval_llm.py` — 理解评估逻辑（perplexity 或生成样本质量评估）
- `trainer/train_dpo.py` — 只看开头，理解 DPO（直接偏好优化）的数据格式（chosen vs rejected）
- 回顾 `model/model_minimind.py` 的 `MiniMindConfig`，检查自己能否解释每个参数
- 回顾 `trainer/train_lora.py` 和 `model/model_lora.py`，确认 LoRA 逻辑清晰

### 3. 必须掌握的技术点（整合复习）

| 复试必备问题 | 参考答案要点 |
|---|---|
| MiniMind 和 GPT 有什么关系？ | 架构参考 GPT/LLaMA，Decoder-only，但参数极小（26M vs 175B），面向教学 |
| 项目最大的技术亮点是什么？ | 全链路 PyTorch 原生实现，覆盖 Pretrain/SFT/LoRA/DPO/RLHF |
| 你在项目中具体做了什么？ | （见下方包装话术） |
| 遇到了什么困难？怎么解决的？ | 理解 KV Cache 实现、loss_mask 设计、LoRA 权重合并 |
| 和工业级大模型相比，局限在哪里？ | 参数量小（能力有限）、训练数据小（知识不全）、无 RLHF 大规模对齐 |

### 4. 面试表达模板

> **"你在这个项目里做了什么？"（包装话术）**

*"我系统学习并复现了 MiniMind 项目的核心模块。具体来说：第一，我深入阅读了模型结构代码，理解了 RMSNorm、RoPE、SwiGLU 和 GQA 的设计原理及其相比传统方案的优势；第二，我跑通了预训练流程，理解了 Next Token Prediction 的 loss 计算逻辑；第三，我研究了 SFT 的 loss_mask 设计，理解了为什么只对 response 部分计算梯度；第四，我重点分析了 LoRA 模块的实现（model/model_lora.py），理解了低秩分解如何在保持模型能力的同时大幅减少可训练参数；此外，我对比分析了 Pretrain、SFT、DPO 三个训练阶段在数据格式和优化目标上的差异。"*

> **"这个模型有什么局限性？如何改进？"**

*"MiniMind 因参数量限制（26M），语言能力和知识储备与工业级大模型差距明显。改进方向包括：扩大训练数据（提升知识覆盖）、增加模型参数（如启用 MoE 扩展）、引入更高质量的 RLHF 对齐数据（提升指令跟随能力）、以及使用 Flash Attention 等工程优化提升训练效率。"*

### 5. 今日最低成果

- 能完整背出项目介绍（2~3 分钟版本）
- 能回答"你做了什么"且听起来真实可信
- 能说出至少 3 个"为什么这样设计"的技术问题答案

---

# 一周后必须具备的复试回答能力

> 如果老师围绕 MiniMind 连续追问 10 分钟，你应该具备以下能力：

## 🔲 项目整体层面

- [ ] 能在 30 秒内介绍项目背景、规模、技术栈
- [ ] 能说出 Decoder-only 架构的优势，以及与 Encoder-Decoder（如 T5）的区别
- [ ] 能解释项目的全训练链路：Pretrain → SFT → LoRA/DPO → 推理

## 🔲 模型结构层面

- [ ] 能描述 Self-Attention 的计算过程（Q、K、V、softmax、Causal Mask）
- [ ] 能解释 RMSNorm 相比 LayerNorm 的区别和优势
- [ ] 能描述 SwiGLU 的门控结构
- [ ] 能解释 RoPE 是什么，为什么优于绝对位置编码
- [ ] 能说出 GQA（分组查询注意力）节省显存的原理

## 🔲 训练流程层面

- [ ] 能解释 Pretrain 的目标（NTP）和数据特点（无监督）
- [ ] 能解释 SFT 的 loss_mask 设计（只对 response 计算 loss）
- [ ] 能解释 LoRA 的数学原理（ΔW = A @ B，低秩分解）
- [ ] 能说出 LoRA 节省显存的具体原因（原始权重冻结）
- [ ] 能描述 DPO 的数据格式（chosen/rejected pair）和优化目标

## 🔲 推理与工程层面

- [ ] 能描述自回归生成的完整流程
- [ ] 能解释 KV Cache 的作用和节省的计算量
- [ ] 能说出 temperature 和 top-p 采样的效果
- [ ] 能解释 BPE Tokenizer 的基本原理

## 🔲 项目价值与反思层面

- [ ] 能说出 MiniMind 相比工业级大模型的局限性
- [ ] 能说出你通过这个项目"学到了什么"（不空泛，有具体技术点）
- [ ] 能提出 1~2 个合理的改进方向（扩参数、更多数据、Flash Attention 等）

---

## 📌 附：关键技术点速查表

| 技术名称 | 一句话解释 | 所在文件 |
|---|---|---|
| RMSNorm | 只用均方根归一化，比 LayerNorm 更快 | `model/model_minimind.py` |
| RoPE | 旋转矩阵编码位置信息，支持长序列外推 | `model/model_minimind.py` |
| SwiGLU | 门控激活函数，`SiLU(gate) × up` | `model/model_minimind.py` |
| GQA | 多个 Q 头共享少量 K/V 头，节省显存 | `model/model_minimind.py` |
| Causal Mask | 下三角掩码，防止看到未来 token | `model/model_minimind.py` |
| KV Cache | 缓存历史 K/V，加速自回归推理 | `model/model_minimind.py` |
| LoRA | 低秩矩阵分解，冻结原始权重只训 A/B | `model/model_lora.py` |
| loss_mask | SFT 中只对 response 部分计算 loss | `trainer/train_full_sft.py` |
| BPE | 字节对编码，高频子词合并构建词表 | `trainer/train_tokenizer.py` |
| NTP | Next Token Prediction，预训练目标 | `trainer/train_pretrain.py` |
| DPO | 直接偏好优化，chosen vs rejected 对 | `trainer/train_dpo.py` |
| MoE | 混合专家，用 `use_moe=True` 开启 | `model/model_minimind.py` |

---

*本文档基于 MiniMind 仓库实际文件结构生成，文件路径均已验证。每天学习时间建议 2~3 小时，优先保证理解而非记忆。*
