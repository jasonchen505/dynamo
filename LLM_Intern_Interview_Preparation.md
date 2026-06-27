# LLM 算法实习面试准备指南：从 Dynamo + SGLang + slime 三项目视角

> 本文档基于对 NVIDIA Dynamo（分布式推理编排层）、SGLang（高性能推理引擎）、slime（RL 后训练框架）三个项目的深度分析，面向正在寻找 LLM 算法实习的硕士在读同学。
>
> **定位**：已有 `sglang/LLM_Interview_Preparation.md`（SGLang+slime 综合）、`sglang/Technical_Interview_Deep_Dive.md`（五类面试深挖）、`slime/LLM_Interview_Preparation.md`（slime 专项）等文档，**本文档聚焦于这些文档未覆盖或覆盖不足的交叉视角与 infra-to-algorithm 桥接知识**。

---

## 目录

- [第一部分：全局视角 — 理解 LLM 全生命周期](#第一部分全局视角--理解-llm-全生命周期)
- [第二部分：推理系统深度 — 从 SGLang 到 Dynamo](#第二部分推理系统深度--从-sglang-到-dynamo)
- [第三部分：后训练算法深度 — slime 框架与 RL 核心](#第三部分后训练算法深度--slime-框架与-rl-核心)
- [第四部分：Infra 与 Algorithm 的交叉面试题](#第四部分infra-与-algorithm-的交叉面试题)
- [第五部分：系统设计场景题](#第五部分系统设计场景题)
- [第六部分：Debug 与优化实战](#第六部分debug-与优化实战)
- [第七部分：高级专题](#第七部分高级专题)
- [第八部分：项目介绍模板](#第八部分项目介绍模板)
- [附录：关键代码位置索引](#附录关键代码位置索引)

---

## 第一部分：全局视角 — 理解 LLM 全生命周期

### 1.1 三个项目的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM 全生命周期                                │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐  │
│  │  预训练   │───▶│  后训练   │───▶│      推理服务部署         │  │
│  │(Pretrain) │    │(Post-    │    │  (Inference Serving)     │  │
│  │          │    │ Training)│    │                          │  │
│  └──────────┘    └──────────┘    └──────────────────────────┘  │
│                        │                    │                    │
│                   ┌────┴────┐         ┌────┴────┐              │
│                   │  slime  │         │  Dynamo │              │
│                   │ (RL训练) │         │(编排层)  │              │
│                   └────┬────┘         └────┬────┘              │
│                        │                    │                    │
│                   ┌────┴────────────────────┴────┐              │
│                   │         SGLang               │              │
│                   │   (高性能推理引擎)             │              │
│                   └──────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

**核心理解**：
- **SGLang** 是底层推理引擎，负责单机/单节点内的高效推理（RadixAttention、连续批处理、投机解码）
- **Dynamo** 是 SGLang 之上的编排层，负责多节点/集群级别的推理协调（分离式服务、KV感知路由、多级缓存、自动扩缩容）
- **slime** 使用 SGLang 作为 rollout 引擎进行 RL 后训练，训练完的模型又通过 SGLang/Dynamo 部署上线

### 1.2 为什么算法岗也需要理解 Infra？

面试官视角：一个优秀的 LLM 算法工程师需要理解：
1. **模型能力受推理系统约束**：KV Cache 管理策略直接影响你能服务的上下文长度；投机解码影响用户体验
2. **RL 训练与推理引擎深度耦合**：slime 的 rollout 依赖 SGLang，weight sync 的延迟影响 on-policy 程度
3. **算法选择有系统级代价**：GRPO（无 Critic）比 PPO 省 50% 推理资源；分离式 prefill/decode 改变了资源分配策略
4. **Debug 需要全栈视野**：NaN loss 可能是模型问题、也可能是推理引擎的 KV Cache 溢出导致

### 1.3 面试中的自我定位

> "我研究了三个开源项目：Dynamo（NVIDIA 的分布式推理编排框架）、SGLang（高性能推理引擎）、slime（清华的 RL 后训练框架）。这让我理解了从模型训练到推理部署的完整链路，特别是 RL 后训练如何与推理系统协同优化，以及大规模推理服务中的关键技术决策。"

---

## 第二部分：推理系统深度 — 从 SGLang 到 Dynamo

### 2.1 KV Cache 管理：从 RadixAttention 到多级缓存

#### 2.1.1 RadixAttention 核心原理（SGLang）

**面试必答问题**：RadixAttention 是什么？解决了什么问题？

**回答框架**：

**问题**：在多轮对话、Agent 工作流、批量推理中，多个请求共享相同的前缀（系统提示、历史对话）。传统推理引擎每次都重新计算这些共享前缀的 KV Cache，浪费大量算力。

**解决方案**：SGLang 使用 **Radix Tree（基数树/压缩前缀树）** 管理 KV Cache，实现前缀级别的缓存共享。

**数据结构细节**（可深挖点）：

```python
class TreeNode:
    key: RadixKey          # token 序列段
    value: Tensor          # GPU KV Cache 索引
    lock_ref: int          # 引用计数（防止被驱逐）
    last_access_time: float # 用于 LRU 驱逐
    hit_count: int         # 用于 LFU/SLRU 驱逐
    host_value: Tensor     # CPU 备份（HiCache L2）
    hash_value: List[str]  # SHA256 内容寻址哈希
```

**前缀匹配算法**：指数搜索（Exponential Search）— O(log(前缀长度)) 而非 O(n)

```python
# 伪代码：指数搜索找最长公共前缀
lo = 0; step = 1
while lo < n:
    hi = min(lo + step, n)
    if t0[lo:hi] != t1[lo:hi]:
        # 在窗口内二分搜索
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if t0[lo:mid] == t1[lo:mid]: lo = mid
            else: hi = mid
        return lo
    lo = hi; step *= 2
```

**面试深挖问题**：
1. **Q: 为什么用 Radix Tree 而不是 Hash Map 做前缀缓存？**
   - A: Hash Map 只能做精确匹配，无法高效找最长公共前缀。Radix Tree 天然支持前缀匹配，且支持 LRU 驱逐、层级缓存、内容寻址。
   
2. **Q: 如何处理 LoRA 场景下的缓存隔离？**
   - A: `RadixKey` 有 `extra_key` 字段，可以是 LoRA adapter name，将其纳入 block hash 计算，确保不同 LoRA 的缓存不混淆。

3. **Q: 缓存驱逐策略有哪些？何时用哪种？**
   - A: LRU（默认，通用）、LFU（热点数据场景）、SLRU（分段 LRU，防止单次扫描污染）、FIFO、Priority-aware（自定义优先级）。高频多轮对话用 SLRU，批量推理用 LRU。

#### 2.1.2 从单机到集群：Dynamo 的 KV 多级缓存体系

**面试加分项**：理解 Dynamo 如何将 SGLang 的单机 Radix Cache 扩展到集群级别。

**KV Block Manager (KVBM) 四级存储**：

| 层级 | 存储 | 延迟 | 容量 | 用途 |
|------|------|------|------|------|
| G1 (Device) | GPU VRAM | ~μs | 数十GB | 热数据，正在处理的 KV |
| G2 (Host) | CPU Pinned Memory | ~100μs | 数百GB | 温数据，近期可能复用 |
| G3 (Disk) | 本地 NVMe SSD | ~ms | 数TB | 冷数据，长上下文回溯 |
| G4 (Remote) | S3/Azure Blob | ~10ms | 无限 | 归档数据，跨节点共享 |

**面试深挖问题**：
1. **Q: 为什么需要多级缓存？GPU 内存不够用吗？**
   - A: 对于长上下文（128K+ tokens）和高并发场景，单 GPU 的 KV Cache 容量有限。例如 70B 模型、128K 上下文、FP16 KV 需要 ~40GB，单卡放不下。多级缓存通过分层存储，在不增加 GPU 成本的情况下扩展有效缓存容量。

2. **Q: KV Cache 的 block 状态机是怎样的？**
   - A: `Reset → Partial → Complete → Registered → Reset`。Partial 表示正在填充，Complete 表示可以使用，Registered 表示已注册到 NIXL 可以跨节点传输。RAII 管理生命周期，dropping 触发 Remove 事件。

3. **Q: 如何决定哪些 KV 应该 offload 到 CPU/SSD？**
   - A: Dynamo 使用频率+热度双重策略。`frequency >= 2` 的 block 才会被 offload 到 SSD（防止一次性数据污染 SSD）。频率在 cache hit 时翻倍，在时间衰减时递减。

#### 2.1.3 与 KV Cache 相关的模型架构知识

**面试高频问题**：GQA（Grouped Query Attention）和 MLA（Multi-head Latent Attention，DeepSeek）对 KV Cache 管理有什么影响？

- **MHA**：每个 head 有独立的 K、V，KV Cache 大小 = `2 × num_heads × seq_len × head_dim × dtype`
- **GQA**：多个 query head 共享一组 K、V，KV Cache 大小 = `2 × num_kv_heads × seq_len × head_dim × dtype`，显著减小
- **MLA**（DeepSeek）：将 KV 投影到低秩空间，KV Cache 只存压缩后的 latent，进一步压缩。Dynamo 针对 MLA 提供 NCCL replicated mode：只有 rank 0 从 G2/G3 加载，然后 broadcast 给所有 GPU

**面试可展开**：MLA 的压缩比、GQA 的分组策略、FP8 KV Cache 的精度影响。

### 2.2 分离式服务（Disaggregated Serving）

#### 2.2.1 为什么要分离 Prefill 和 Decode？

**回答框架**：

| 维度 | Prefill（预填充） | Decode（解码） |
|------|------------------|---------------|
| 计算特性 | Compute-bound（大矩阵乘） | Memory-bound（逐 token 读 KV） |
| 批处理 | 可以大批量并行 | 只能逐 token 生成 |
| GPU 利用率 | 高（大 kernel） | 低（小 kernel，频繁启动） |
| 延迟敏感度 | TTFT（首 token 延迟） | ITL（token 间延迟） |

**核心洞察**：Prefill 和 Decode 的硬件需求完全不同。分离后可以：
1. 独立扩缩容（高峰期 decode 不被 prefill 抢占资源）
2. 专用硬件优化（prefill 用高算力卡，decode 用高带宽卡）
3. 避免 prefill 的长计算阻塞 decode 的实时输出

#### 2.2.2 Dynamo 的分离式服务流程

```
Client Request
    │
    ▼
┌──────────┐    ┌───────────────┐    ┌──────────────┐
│ Frontend  │───▶│ PrefillRouter │───▶│ Prefill Worker│
│(OpenAI API)│   │ (KV-aware)    │    │ (计算 KV)    │
└──────────┘    └───────┬───────┘    └──────┬───────┘
                        │                    │
                        │    disaggregated_params
                        │◀───────────────────┘
                        │
                ┌───────▼───────┐    ┌──────────────┐
                │ DecodeRouter  │───▶│ Decode Worker │
                │ (KV-aware)    │    │ (生成 token)  │
                └───────────────┘    └──────┬───────┘
                                           │
                                    NIXL GPU-to-GPU
                                    KV Cache 传输
```

**面试深挖问题**：
1. **Q: Prefill 和 Decode 之间如何传递 KV Cache？**
   - A: 通过 NIXL（NVIDIA Interchange Library）进行 GPU-to-GPU 直接传输。SGLang 使用 RDMA（bootstrap handshake 建立连接），vLLM 使用 block IDs + 远程 worker 信息。传输是非阻塞的，GPU forward pass 在传输过程中继续执行。

2. **Q: 分离式服务的额外开销是什么？如何最小化？**
   - A: 主要开销是 KV Cache 传输延迟。优化方向：(1) RDMA 高带宽网络，(2) 拓扑感知路由（同一个 NVLink domain 内的 prefill/decode 优先配对），(3) FP8 KV 压缩减少传输量。

3. **Q: SGLang 的异步 prefill 与 vLLM 的同步 prefill 有什么区别？**
   - A: SGLang 的 prefill 作为后台任务运行，decode 可以立即开始（不需要等待 prefill 完成），通过 bootstrap handshake 协调。vLLM 需要等 prefill 完成后才选择 decode worker。SGLang 的方式延迟更低但实现更复杂。

### 2.3 KV 感知路由（KV-Aware Routing）

#### 2.3.1 核心算法

**面试必答**：Dynamo 的 KV Router 如何选择 worker？

**代价函数**（来自 `lib/kv-router/src/scheduling/selector.rs`）：

```
adjusted_prefill_blocks = max(
    prefill_blocks
    - overlap_score_credit × device_overlap_blocks   # 本地缓存命中
    - host_cache_hit_weight × host_overlap_blocks     # CPU 缓存命中
    - disk_cache_hit_weight × disk_overlap_blocks     # SSD 缓存命中
    - shared_cache_multiplier × shared_beyond_blocks  # 共享缓存命中
, 0)

cost = prefill_load_scale × adjusted_prefill_blocks + decode_blocks
```

**选择策略**：
- `temperature = 0`：选择 cost 最低的 worker（确定性）
- `temperature > 0`：softmax 采样（随机性，用于负载均衡）

**面试深挖问题**：
1. **Q: 为什么需要 KV 感知路由？普通负载均衡不够吗？**
   - A: 普通负载均衡（round-robin、least-connections）只看 worker 负载，不考虑缓存亲和性。如果请求被路由到没有相关缓存的 worker，需要重新计算 prefill，浪费算力且增加 TTFT。KV 感知路由可以将 TTFT 降低 2x。

2. **Q: 如何知道每个 worker 上有哪些 KV Cache？**
   - A: 通过 Radix Tree 索引器。每个 worker 发布 KV cache 事件（Add/Remove），索引器维护全局的 Radix Tree 视图。路由时查询 tree 找到与请求 token 序列有最大前缀重叠的 worker。

3. **Q: 什么是 Predict-on-Route？为什么需要？**
   - A: 当 `--router-predicted-ttl-secs` 启用时，路由决策会被记录到一个短期 TTL 的辅助索引中。这样，后续到达的兄弟请求（如 best-of-N、parallel sampling）可以看到第一个请求的路由决策并被路由到同一个 worker，避免重复 prefill。

#### 2.3.2 调度队列策略

Dynamo 的 Router 使用 **Deficit Round Robin (DRR)** 进行多队列调度：

| 策略 | 适用场景 | 复杂度 |
|------|---------|--------|
| FCFS | 通用 | O(1) |
| WSPT (Smith's Rule) | 优化平均 TTFT | O(log n) |
| DRR | 多优先级/多策略类 | O(C) per arbitration |

**面试加分理解**：WSPT 的评分公式 `score = (1 + priority_jump) / new_tokens`，其中 `new_tokens = max(ISL - effective_cached_tokens, 1)`。这意味着缓存命中率高的请求优先级更高（因为 new_tokens 更小）。

### 2.4 投机解码（Speculative Decoding）

#### 2.4.1 核心原理

**面试必答**：什么是投机解码？为什么能加速？

**回答框架**：
- **原理**：用快速的 draft 机制生成候选 token，然后用目标模型并行验证。如果候选被接受，节省了多次 forward pass。
- **关键指标**：接受率（acceptance rate）— 越高加速越好
- **上限**：理论上不超过 2-3x（受接受率限制）

#### 2.4.2 SGLang 支持的投机解码算法

| 算法 | Draft 来源 | 特点 |
|------|-----------|------|
| EAGLE/EAGLE3 | 小 draft 模型 + 树验证 | 最高 2.3x 加速，需要额外模型 |
| NGRAM | CPU 侧 n-gram 语料库 | 零 GPU 开销，适合代码补全 |
| DFlash | 目标模型本身 | 下一代，2026年6月发布 |
| FrozenKV-MTP | 模型内部 MTP head | 零额外参数，需要模型支持 |

**面试深挖问题**：
1. **Q: EAGLE 的树验证是如何工作的？**
   - A: Draft 模型生成一棵候选 token 树（topk 控制宽度，num_steps 控制深度）。目标模型一次 forward pass 验证整棵树。使用 `build_tree_kernel_efficient` CUDA kernel 构建树，`reconstruct_indices_from_tree_mask` 处理树掩码。

2. **Q: 什么是自适应投机（Adaptive Speculation）？**
   - A: `AdaptiveController` 根据观察到的接受率动态调整 `num_steps`。接受率高时增加深度（更多投机 token），接受率低时减少深度（减少浪费的计算）。

3. **Q: NGRAM 投机解码为什么适合代码补全？**
   - A: 代码中有大量重复模式（函数名、变量名、语法结构），n-gram 匹配命中率高。且不需要 draft 模型，零 GPU 内存开销。

### 2.5 Planner 自动扩缩容

#### 2.5.1 为什么 LLM 推理需要自定义 autoscaler？

**面试加分理解**：传统 K8s HPA 基于 CPU 利用率扩缩容，但 LLM 推理有特殊性：
1. 延迟取决于请求内容（token 数量），不是请求计数
2. Prefill 和 Decode 有不同的扩缩容特性
3. SLA（TTFT/ITL）不映射到 CPU 利用率
4. 扩缩容决策代价高（分钟级，不是秒级）

#### 2.5.2 Dynamo Planner 的双循环设计

| 循环 | 间隔 | 数据源 | 策略 |
|------|------|--------|------|
| 基于吞吐量 | 180s（默认） | 流量预测（ARIMA/Prophet/Kalman）+ 性能模型 | 设定下界 |
| 基于负载 | 5s | 实时 ForwardPassMetrics | 在下界之上微调 ±1 |

**扩缩容判断逻辑**：
- **Scale Up**：所有引擎的预估 TTFT/ITL > SLA
- **Scale Down**：所有引擎的预估 TTFT/ITL < SLA × sensitivity（且合并后仍在 SLA 内）

**面试深挖问题**：
1. **Q: 为什么 scale down 需要 "合并后预测"（consolidation-aware）？**
   - A: 减少一个 worker 后，剩余 worker 的负载会增加。需要重新预测合并后的 TTFT/ITL，确保仍在 SLA 内。公式：`new_queue = old_queue × N/(N-1)`。

2. **Q: 流量预测有哪些方法？各自适用场景？**
   - A: Constant（简单稳定场景）、ARIMA（有趋势/季节性）、Kalman（实时在线更新）、Prophet（复杂季节性，需要历史数据）。

---

## 第三部分：后训练算法深度 — slime 框架与 RL 核心

### 3.1 RL 后训练全景

#### 3.1.1 为什么需要 RL 后训练？

**面试必答**：SFT（Supervised Fine-Tuning）的局限性是什么？RL 如何补充？

| 维度 | SFT | RL 后训练 |
|------|-----|----------|
| 学习方式 | 模仿学习（teacher forcing） | 探索+反馈 |
| 数据需求 | 高质量标注 | 奖励信号（可自动生成） |
| 能力边界 | 不超过标注数据 | 可超越标注数据 |
| 推理能力 | 有限 | 强（通过 verifiable reward） |
| 对齐效果 | 风格对齐 | 行为对齐 |

#### 3.1.2 RL 后训练的核心循环

```
┌─────────────────────────────────────────────────────────┐
│                    RL 后训练循环                          │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Rollout  │───▶│  Reward  │───▶│ Advantage │          │
│  │ (SGLang生成)│  │ (计算奖励)│   │ (估计优势)│          │
│  └──────────┘    └──────────┘    └──────────┘          │
│       ▲                                │                │
│       │          ┌──────────┐          │                │
│       └──────────│  Policy  │◀─────────┘                │
│        权重同步   │  Update  │                           │
│                  │(Megatron) │                           │
│                  └──────────┘                           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 核心 RL 算法对比

#### 3.2.1 GRPO vs PPO（面试高频）

**GRPO（Group Relative Policy Optimization）**：

```python
# 核心公式
advantages = (rewards - group_mean) / (group_std + 1e-6)
# 不需要 Critic 模型，直接用 group 内的相对奖励

# slime 实现
def get_grpo_returns(rewards, n_samples_per_prompt, kl_coef, kl_per_token):
    rewards = rewards.reshape(-1, n_samples_per_prompt)
    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True)
    normalized_rewards = (rewards - group_mean) / (group_std + 1e-6)
    # 展开并减去 KL 惩罚
    returns = normalized_rewards.reshape(-1) - kl_coef * kl_per_token
    return returns
```

**PPO（Proximal Policy Optimization）**：

```python
# 核心公式：GAE (Generalized Advantage Estimation)
delta = reward + gamma * next_value - value
gae = delta + gamma * lambda * next_gae
advantages = gae
returns = advantages + values

# slime 实现：chunked_gae() 用于高效并行扫描
```

**面试深挖对比**：

| 维度 | GRPO | PPO |
|------|------|-----|
| 是否需要 Critic | ❌ 不需要 | ✅ 需要 |
| 计算开销 | 低（~50% less） | 高（需要 Critic forward） |
| 适用场景 | 可验证奖励（数学、代码） | 复杂/稠密奖励 |
| 稳定性 | 受 group 内方差影响 | 更稳定（GAE 平滑） |
| 样本效率 | 需要多 samples/prompt | 单样本即可 |
| 超参数敏感度 | 对 n_samples_per_prompt 敏感 | 对 gamma/lambda 敏感 |

#### 3.2.2 CISPO（面试可能问到的前沿算法）

**来自 MiniMax-M1 论文**，slime 已实现：

```python
# 核心创新：clipped tokens 仍然贡献梯度
# 标准 PPO：ratio > (1+eps) 的 token 被完全丢弃
# CISPO：这些 token 的 ratio 被 clip 到 (1+eps)，但梯度仍回传

ratio = exp(log_prob - old_log_prob)
clipped_ratio = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
# 关键：stop_gradient on the clipping
loss = -(torch.where(ratio > 1 + eps_high, 
                     clipped_ratio.detach() * advantage,  # clip 后不传梯度
                     ratio * advantage))  # 正常传梯度
```

**面试可展开**：为什么 CISPO 比标准 PPO 更适合长序列生成？因为在长序列中，大部分 token 的 ratio 都会被 clip，标准 PPO 会丢弃大量有效梯度信号。

#### 3.2.3 REINFORCE++ 和 OPSM

**REINFORCE++**：在标准 REINFORCE 基础上加入折扣回报和 KL 惩罚：
```python
# token-level rewards with discount
returns[t] = sum(gamma^(k-t) * reward[k]) - kl_coef * kl_per_token[t]
```

**OPSM（Off-Policy Sequence Masking）**：当序列级 KL 过大且 advantage 为负时，mask 掉该序列的 loss，防止 off-policy 数据中的有害梯度。

### 3.3 训练与推理的协同优化

#### 3.3.1 Weight Synchronization（权重同步）

**面试深挖**：RL 训练中，actor 模型的权重需要同步到推理引擎（SGLang）。延迟越长，off-policy 程度越高。

| 模式 | 原理 | 适用场景 |
|------|------|---------|
| Full Sync + NCCL | 每次同步全部参数 | 默认，简单可靠 |
| Delta Sync + NCCL | 只传输变化的参数（~3% 密度） | 大模型，带宽受限 |
| Full Sync + Disk | 写磁盘再加载 | 跨节点无高速网络 |
| Delta Sync + Disk | 只写变化部分到磁盘 | 最省带宽 |

**Delta Sync 的实现细节**：
1. **Diff**：byte-by-byte 对比当前参数与 pinned-CPU 快照
2. **Encode**：只记录 (position, value) 对
3. **Transfer**：NCCL 或磁盘
4. **Apply**：NaN-masked overwrite

**面试问题**：Delta Sync 的精度风险？— 如果 diff 检测有浮点精度误差，可能遗漏小更新。slime 用 byte-level 对比避免这个问题。

#### 3.3.2 异步训练（Async Training）

**同步训练**：generate → train → sync weights → generate → ...

**异步训练**：generate + train 重叠执行

```python
# 伪代码
rollout_data_next = rollout_manager.generate.remote(start_rollout_id)
for rollout_id in range(start_rollout_id, num_rollout):
    rollout_data_curr = rollout_data_next  # 收集上一轮
    rollout_data_next = rollout_manager.generate.remote(rollout_id + 1)  # 提前开始下一轮
    actor_model.async_train(rollout_id, rollout_data_curr)  # 训练当前轮
    if (rollout_id + 1) % update_weights_interval == 0:
        sync(rollout_data_next)  # 等待进行中的 rollout
        actor_model.update_weights()  # 同步权重
```

**面试问题**：异步训练的 trade-off？— 更高的 GPU 利用率（训练和推理重叠），但更大的 off-policy 程度（生成时的模型版本比训练时旧）。`update_weights_interval` 控制这个 trade-off。

#### 3.3.3 训练与推理的 Colocate vs Separated

| 模式 | 优点 | 缺点 | 适用 |
|------|------|------|------|
| Colocate | 省 GPU（共享） | 需要 CPU offload，切换开销 | 资源紧张 |
| Separated | 简单，各自优化 | 需要更多 GPU | 资源充足 |

### 3.4 Agent 训练的特殊考虑

#### 3.4.1 Loss Masking

**面试必答**：Agent 训练中为什么需要 loss masking？

**回答**：在 Agent 场景中，模型生成的内容包括工具调用和工具返回结果。工具返回的结果不是模型自己生成的，如果也参与 loss 计算，模型会学会"复制"工具输出而非"理解"如何使用工具。

```python
# slime 实现
loss_mask[token] = 0  # 工具返回的 token
loss_mask[token] = 1  # 模型生成的 token

# loss 计算
loss = -sum(log_prob * loss_mask) / sum(loss_mask)
```

#### 3.4.2 Session Routing（会话路由）

在多轮 Agent 交互中，同一会话的请求应该路由到同一个 worker，以复用前缀缓存。Dynamo 通过 `X-Dynamo-Session-ID` header 实现会话亲和性，使用一致性哈希确保路由稳定性。

#### 3.4.3 Fan-out Training

当一个 rollout 产生多个可训练片段时（如多轮工具调用），slime 支持 fan-out training：每个片段作为独立的 `Sample` 返回，但共享同一个 `rollout_id`，确保梯度贡献正确聚合。

---

## 第四部分：Infra 与 Algorithm 的交叉面试题

### 4.1 KV Cache 管理如何影响 RL 训练？

**Q: 在 RL 后训练中，KV Cache 管理策略如何影响训练效率？**

**回答**：
1. **Rollout 阶段**：SGLang 的 RadixAttention 让同一 prompt 的多个 responses（GRPO 的 n_samples_per_prompt）共享 prefill 的 KV Cache，节省 50%+ 的 prefill 计算
2. **长上下文训练**：128K+ 上下文的 Agent 训练，KV Cache 可能超出 GPU 内存。需要 chunked prefill + retraction 机制
3. **Weight Sync 后的缓存失效**：每次权重同步后，旧的 KV Cache 失效。Delta Sync（只更新变化部分）可以减少缓存失效范围

### 4.2 分离式服务如何影响推理质量？

**Q: Prefill 和 Decode 分离后，对模型输出质量有影响吗？**

**回答**：
- **理论上无影响**：KV Cache 是精确传输的，decode 阶段看到的 KV 与不分裂时完全一致
- **实践中可能有影响**：(1) KV 传输延迟导致 decode 开始时的上下文可能稍旧；(2) FP8 KV 压缩有精度损失；(3) 跨节点浮点误差累积
- **验证方法**：对比分离和非分离模式下同一 prompt 的输出，检查 token-level 一致性

### 4.3 路由策略如何影响 RL 数据质量？

**Q: 在 RL 训练中，推理服务的路由策略如何影响 rollout 数据质量？**

**回答**：
1. **缓存亲和性**：KV-aware 路由让同一 prompt 的多个 responses 倾向于路由到同一 worker，共享缓存。这提高了 GRPO 的 rollout 效率
2. **负载均衡**：如果路由不均衡，部分 worker 过载可能导致生成超时或截断，影响 rollout 数据完整性
3. **温度参数**：`router_temperature > 0` 引入随机性，可以增加 rollout 数据的多样性，但可能降低缓存命中率

### 4.4 投机解码与 RL 训练的交互

**Q: 在 RL rollout 阶段使用投机解码，会影响训练效果吗？**

**回答**：
- **接受的 token**：与正常生成完全等价，不影响训练
- **被拒绝的 draft token**：不会出现在最终输出中，不参与 loss 计算
- **训练效果**：理论上无影响（投机解码是无损的），但需要确保 log prob 计算正确（只计算接受的 token）
- **实践考虑**：投机解码改变了 token 生成的时序，可能影响 temperature sampling 的随机性

### 4.5 自动扩缩容与 RL 训练的资源管理

**Q: 如何为 RL 训练配置自动扩缩容？**

**回答**：
- **Rollout 阶段**：需要大量推理资源，应该根据 pending prompt 数量动态扩缩 SGLang 实例
- **Training 阶段**：需要稳定的 GPU 资源，不应该被抢占
- **Dynamo Planner**：可以为 prefill/decode pool 分别设置 SLA，在 rollout 高峰期自动增加 decode worker
- **slime 的 Ray 调度**：使用 Placement Group 隔离训练和推理资源

---

## 第五部分：系统设计场景题

### 5.1 设计一个支持 100K 并发的 Agent 推理服务

**考察点**：系统架构、KV Cache 管理、路由策略、扩缩容

**回答框架**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    架构设计                                      │
│                                                                 │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────────┐     │
│  │ API GW   │───▶│ Dynamo Router │───▶│ Prefill Workers  │     │
│  │(K8s GW)  │    │ (KV-aware)    │    │ (N 个实例)       │     │
│  └──────────┘    └───────┬───────┘    └──────────────────┘     │
│                          │                                      │
│                  ┌───────▼───────┐    ┌──────────────────┐     │
│                  │ Decode Workers│    │ KVBM (G2+G3)     │     │
│                  │ (M 个实例)    │◀──▶│ 多级缓存          │     │
│                  └───────────────┘    └──────────────────┘     │
│                                                                 │
│  关键设计决策：                                                  │
│  1. Session Affinity：同一 agent session 路由到同一 decode       │
│  2. KV-aware 路由：复用前缀缓存，降低 TTFT                       │
│  3. 分离式 Prefill/Decode：独立扩缩容                            │
│  4. 多级 KV 缓存：长上下文 agent 的 KV offload 到 CPU/SSD        │
│  5. Planner SLA 驱动：TTFT < 500ms, ITL < 50ms                  │
└─────────────────────────────────────────────────────────────────┘
```

**可深入讨论的点**：
- 如何处理 Agent 的长尾延迟（某些 tool call 很慢）？
- 如何实现请求迁移（worker 故障时 in-flight 请求不丢失）？
- 如何监控和优化 prefix cache 命中率？

### 5.2 设计一个数学推理的 RL 训练系统

**考察点**：RL 算法选择、奖励设计、训练效率、推理协同

**回答框架**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    系统设计                                      │
│                                                                 │
│  算法选择：GRPO（可验证奖励，无需 Critic）                        │
│  奖励设计：rule-based（\boxed{} 答案匹配 + sympy 验证）          │
│  训练框架：slime (Megatron + SGLang)                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 数据流                                                   │  │
│  │                                                          │  │
│  │ 1. Prompt 采样 → 2. SGLang Rollout (n=8 samples/prompt) │  │
│  │ 3. 答案验证 → 4. GRPO Advantage 计算                     │  │
│  │ 5. Megatron 训练 → 6. Delta Weight Sync → 回到 2        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  关键优化：                                                     │
│  - RadixAttention：8 个 samples 共享 prompt 的 KV Cache         │
│  - Delta Sync：只传输 ~3% 变化参数，减少同步开销                 │
│  - Dynamic Sampling：过滤 reward std=0 的 group（全对/全错）    │
│  - 异步训练：overlap rollout 和 training                        │
│  - Loss Masking：只对 response token 计算 loss                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 设计一个支持多模态的推理服务

**考察点**：多模态处理、缓存策略、资源管理

**回答框架**：
- **Encoder 分离**：图像编码（CLIP/SigLIP）单独运行，与 LLM prefill/decode 分离
- **Embedding Cache**：相同图像的 embedding 可以缓存，避免重复编码
- **Dynamo 支持**：Multimodal E/P/D（Encode/Prefill/Decode）三级分离
- **路由策略**：基于 multimodal hash 的路由，相同图像路由到同一 worker

---

## 第六部分：Debug 与优化实战

### 6.1 RL 训练常见问题诊断

#### 6.1.1 NaN Loss

**诊断路径**：
1. 检查梯度范数：`--clip-grad 1.0` 是否生效
2. 检查 KL divergence：是否爆炸（kl_coef 过大）
3. 检查 reward normalization：group_std 是否为 0
4. 检查 log prob：是否有极端值（underflow/overflow）
5. 检查数据：是否有异常长/短的序列

#### 6.1.2 推理输出乱码

**诊断路径**：
1. 检查 weight sync：是否正确（对比同步前后的参数）
2. 检查 precision：bf16 训练 vs fp16 推理的精度差异
3. 检查 prefix cache：权重更新后缓存是否失效
4. 检查 tokenizer：chat template 是否一致

#### 6.1.3 训练速度下降

**诊断路径**：
1. GPU 利用率：`nvidia-smi` 查看是否空闲
2. 队列状态：SGLang 的 `num_running_reqs` 是否过低
3. Weight sync 延迟：是否占用了大量时间
4. 数据加载：是否成为瓶颈

### 6.2 推理服务优化 Checklist

| 优化方向 | 检查项 | 预期收益 |
|---------|--------|---------|
| Prefix Cache | cache_hit_rate > 0.5? | TTFT 降低 2-5x |
| 批处理 | max_batch_size 是否最大? | 吞吐量提升 2-3x |
| KV Cache | mem_fraction_static 是否合理? | 避免 OOM 和 retraction |
| CUDA Graph | cuda_graph_max_bs 覆盖常用 batch? | 延迟降低 10-20% |
| 投机解码 | accept_rate > 0.7? | 吞吐量提升 1.5-2x |
| Chunked Prefill | 长 prompt 是否分块? | 避免 decode 饥饿 |

### 6.3 监控指标速查

**SGLang Prometheus 指标**：
```
sglang:num_running_reqs          # 正在运行的请求数
sglang:num_waiting_reqs          # 等待队列长度
sglang:token_usage               # KV Cache 使用率
sglang:cache_hit_rate            # Prefix Cache 命中率
sglang:gen_throughput            # 生成吞吐量 (tokens/s)
sglang:time_to_first_token_seconds  # TTFT
sglang:e2e_request_latency_seconds  # 端到端延迟
```

**告警阈值**：
- `num_waiting_reqs > 2000`：系统过载
- `token_usage > 0.95`：KV Cache 即将溢出
- `cache_hit_rate < 0.3`：Prefix Cache 效果差
- `TTFT P99 > 1s`：首 token 延迟过高

---

## 第七部分：高级专题

### 7.1 Reward Model 设计

**面试可能问到**：如何设计一个好的奖励模型？

**回答框架**：
1. **Rule-based**（可验证奖励）：数学用 `\boxed{}` 匹配 + sympy，代码用单元测试通过率。优点：无噪声，可扩展。缺点：只适用于可验证任务。
2. **Model-based**（学习奖励）：训练一个 reward model 打分。优点：适用任意任务。缺点：reward hacking，需要高质量标注。
3. **Environment-based**（环境反馈）：Agent 执行后从环境获取反馈。优点：最真实。缺点：反馈延迟大，信用分配困难。

**slime 支持**：内置 math/dapo/deepscaler/f1/gpqa/ifbench 奖励模型，支持 `--custom-rm-path` 自定义，支持 `--remote-rm-url` 远程 RM 服务。

### 7.2 MoE 模型的 RL 训练挑战

**面试深挖**：MoE（Mixture of Experts）模型的 RL 训练有什么特殊挑战？

1. **路由稳定性**：RL 训练中策略不断变化，可能导致 expert routing 分布剧烈波动。slime 通过 **Routing Replay**（记录并重放 expert routing 决策）来稳定训练。
2. **负载均衡**：部分 expert 可能被过度激活或冷落。需要 auxiliary loss 保持均衡。
3. **通信开销**：Expert Parallelism 需要 All-to-All 通信，在 RL 训练中（频繁 weight sync）通信压力更大。

### 7.3 长上下文 RL 训练

**面试问题**：128K+ 上下文的 Agent 训练有哪些挑战？

1. **KV Cache 溢出**：单 GPU 放不下 128K 的 KV Cache。需要 chunked prefill + KV offload（HiCache）
2. **训练效率**：长序列的 forward/backward 计算量巨大。需要 Context Parallelism（Zigzag Ring Attention）
3. **奖励稀疏**：长上下文 Agent 的奖励可能只在最后一步，credit assignment 困难
4. **内存管理**：需要 activation recomputation + CPU offload

### 7.4 量化与推理效率

**面试问题**：FP8 量化对模型质量的影响？

- **KV Cache FP8**：几乎无损（SGLang 支持 `--kv-cache-dtype fp8_e4m3`），节省 50% 内存
- **Weight FP8**：需要校准，对小模型影响更大，大模型（70B+）影响可忽略
- **FP4**：更激进，需要 AWQ/GPTQ 等量化方法，质量损失更明显

### 7.5 Dynamo 的容错机制

**面试问题**：大规模推理服务如何处理故障？

| 层级 | 机制 | Dynamo 实现 |
|------|------|------------|
| 请求级 | 迁移（保留已生成 token） | `Migration` 重试管理器 |
| Worker 级 | 健康检查 + 优雅关闭 | `/health` + SIGTERM 处理 |
| 引擎级 | Shadow Engine Failover | GPU Memory Service 常驻权重 |
| 系统级 | 负载脱落 | 可配置的 busy 阈值 |
| 基础设施级 | Discovery lease 过期 | etcd 10s TTL + keep-alive |

**请求迁移细节**：
- 可迁移错误：`CannotConnect`, `Disconnected`, `ConnectionTimeout`, `EngineShutdown`
- 不可迁移错误：`Cancelled`, `ResourceExhausted`
- 不支持：guided-decoding（FSM 状态不可转移）、n>1 请求
- Token replay：已生成的 token ID 被追加到请求中，`max_tokens` 递减

---

## 第八部分：项目介绍模板

### 8.1 简短版（1-2 分钟）

> "我深入研究了三个开源项目来理解 LLM 的全生命周期。SGLang 是高性能推理引擎，核心创新是 RadixAttention（用 Radix Tree 管理 KV Cache 实现前缀共享）和零开销调度器。Dynamo 是 NVIDIA 的分布式推理编排层，在 SGLang 之上实现了分离式服务（Prefill/Decode 独立扩缩容）、KV 感知路由（基于缓存重叠的智能调度）、多级 KV 缓存（GPU→CPU→SSD→远程存储）。slime 是清华的 RL 后训练框架，用 SGLang 做 rollout 生成，支持 GRPO/PPO/CISPO 等算法。
>
> 这个项目组合让我理解了：(1) 推理系统如何通过缓存管理和调度优化影响模型服务能力；(2) RL 后训练如何与推理引擎协同工作；(3) 大规模部署中的关键工程决策。"

### 8.2 详细版（3-5 分钟，STAR 方法）

**Situation**：LLM 推理面临效率和规模的双重挑战——单机推理引擎无法充分利用多节点资源，RL 后训练需要高效的 rollout 生成。

**Task**：理解从推理引擎到集群编排到后训练的完整技术栈。

**Action**：
1. 深入分析 SGLang 源码，理解 RadixAttention 的前缀匹配算法（指数搜索 O(log n)）和多级缓存（HiCache GPU→CPU→Storage）
2. 研究 Dynamo 的 KV 感知路由算法——代价函数综合考虑 prefill 负载、缓存重叠、decode 负载
3. 分析 slime 的 RL 训练流程——GRPO 的 group-relative advantage 计算、Delta Weight Sync 的 ~3% 参数密度

**Result**：形成了对 LLM 全生命周期的系统理解，能够在算法设计时考虑系统约束（如 GRPO 省 Critic 推理资源），在系统优化时理解算法需求（如 Agent 训练的 loss masking）。

### 8.3 面试中被追问的应对策略

**如果面试官问"你实际做了什么？"**：
> "我通过阅读源码和文档来学习这些框架。我的贡献是：(1) 整理了技术文档帮助其他学习者；(2) 在本地复现了关键功能；(3) 分析了框架的设计决策和 trade-off。"

**如果面试官问"你遇到了什么问题？如何解决的？"**：
> 准备 2-3 个具体的 debug 故事，参考第六部分的诊断路径。

---

## 附录：关键代码位置索引

### Dynamo 关键代码

| 模块 | 路径 | 说明 |
|------|------|------|
| KV Router 代价函数 | `lib/kv-router/src/scheduling/selector.rs` | `worker_logit()` 方法 |
| RadixTree 实现 | `lib/kv-router/src/indexer/radix_tree.rs` | 前缀匹配 + 缓存索引 |
| KVBM Block 管理 | `lib/kvbm-engine/` | 四级存储状态机 |
| Planner 负载扩缩 | `components/src/dynamo/planner/load_scaling.py` | SLA 驱动的扩缩容 |
| Planner 吞吐扩缩 | `components/src/dynamo/planner/throughput_scaling.py` | 流量预测 + 性能模型 |
| 请求迁移 | `lib/llm/src/migration.rs` | RetryManager + token replay |
| Agent 会话头 | `lib/llm/src/protocols/agents.rs` | Session ID 提取 |
| Frontend 请求处理 | `components/src/dynamo/frontend/main.py` | OpenAI API 兼容层 |
| 分离式服务设计 | `docs/design-docs/disagg-serving.md` | 架构文档 |
| KVBM 设计 | `docs/components/kvbm/` | 多级缓存设计文档 |
| 整体架构 | `docs/design-docs/overall-architecture.md` | 三平面架构 |

### SGLang 关键代码

| 模块 | 路径 | 说明 |
|------|------|------|
| RadixCache | `python/sglang/srt/mem_cache/radix_cache.py` | Radix Tree 缓存管理 |
| Scheduler | `python/sglang/srt/managers/scheduler.py` | 核心调度器（4324 行） |
| SchedulePolicy | `python/sglang/srt/managers/schedule_policy.py` | LPM/FCFS/DFS-Weight |
| Memory Pool | `python/sglang/srt/mem_cache/memory_pool.py` | 两级内存池 |
| HiRadixCache | `python/sglang/srt/mem_cache/hiradix_cache.py` | 多级缓存 |
| EAGLE Worker | `python/sglang/srt/speculative/eagle_worker_v2.py` | 投机解码 |
| PD Disaggregation | `python/sglang/srt/disaggregation/` | 分离式服务 |
| ForwardBatch | `python/sglang/srt/model_executor/forward_batch_info.py` | GPU 批处理数据 |
| Attention Backends | `python/sglang/srt/layers/attention/` | 40+ 注意力后端 |
| sgl-kernel | `sgl-kernel/csrc/` | 自定义 CUDA kernel |

### slime 关键代码

| 模块 | 路径 | 说明 |
|------|------|------|
| 训练入口 | `train.py`, `train_async.py` | 同步/异步训练 |
| Loss 函数 | `slime/backends/megatron_utils/loss.py` | PPO/GRPO/CISPO loss |
| PPO 工具 | `slime/utils/ppo_utils.py` | GAE、advantage 计算 |
| SGLang Rollout | `slime/rollout/sglang_rollout.py` | 推理生成 |
| 奖励模型 | `slime/rollout/rm_hub/` | 内置奖励模型 |
| 权重同步 | `slime/backends/megatron_utils/update_weight/` | Delta sync |
| Ray 编排 | `slime/ray/` | Placement Group + Actor |
| 动态采样 | `slime/rollout/filter_hub/dynamic_sampling_filters.py` | DAPO 过滤 |
| 数据源 | `slime/rollout/data_source.py` | Prompt 采样 + buffer |
| 参数定义 | `slime/utils/arguments.py` | 所有超参数 |

### 已有面试准备文档索引

| 文档 | 路径 | 侧重 |
|------|------|------|
| SGLang+slime 综合 | `sglang/LLM_Interview_Preparation.md` | 架构 + 10 Q&A |
| 五类面试深挖 | `sglang/Technical_Interview_Deep_Dive.md` | 原理/验证/Debug/工程/业务 |
| 学习笔记 | `sglang/Learning_Increment_Notes.md` | 代码级细节 |
| slime 专项 | `slime/LLM_Interview_Preparation.md` | RL 算法 + Agent 训练 |
| 五类指南 | `slime/Interview_Five_Categories_Guide.md` | 回答框架 + 成本估算 |
| 复现计划 | `sglang/Reproduction_Plan_8x4090.md` | 8x4090 实操指南 |

---

> **使用建议**：
> 1. 先读本文档建立全局视角
> 2. 根据面试方向深入对应部分（算法岗重点看第三、四部分，infra 岗重点看第二部分）
> 3. 结合已有文档（`sglang/Technical_Interview_Deep_Dive.md` 的五类框架）准备具体回答
> 4. 准备 2-3 个具体的 debug 故事和系统设计案例
> 5. 在本地复现关键功能（参考 `sglang/Reproduction_Plan_8x4090.md`）
