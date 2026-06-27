# 技术面试五类问题深度应对：Dynamo × SGLang × slime 三项目实战视角

> **定位**：本文档基于对 NVIDIA Dynamo（分布式推理编排）、SGLang（高性能推理引擎）、slime（RL 后训练框架）三个项目的深度分析，针对技术面试五类核心能力提供**可直接使用的回答框架、深挖点、debug 路径和业务分析**。
>
> **与已有文档的关系**：已有 `sglang/Technical_Interview_Deep_Dive.md`（SGLang 侧重）和 `slime/Interview_Five_Categories_Guide.md`（slime 侧重），本文档聚焦于 **三项目交叉视角** 和 **Dynamo 独有知识**，避免重复已有内容。

---

## 目录

- [第一类：底层原理深入理解](#第一类底层原理深入理解)
- [第二类：实验和方案验证能力](#第二类实验和方案验证能力)
- [第三类：问题定位能力](#第三类问题定位能力)
- [第四类：工程落地能力](#第四类工程落地能力)
- [第五类：业务与实际场景理解](#第五类业务与实际场景理解)
- [附录：三项目关键代码索引](#附录三项目关键代码索引)

---

## 第一类：底层原理深入理解

> **考察重点**：不是回答清楚概念，而是讲清楚**这个方法解决什么问题、存在哪些局限性、有哪些改进方法**。面试官要看你是否真正理解设计决策背后的 trade-off。

### 1.1 分离式服务（Disaggregated Prefill/Decode）

#### 解决什么问题？

**问题背景**：LLM 推理的两个阶段有根本不同的计算特性：

| 维度 | Prefill（预填充） | Decode（解码） |
|------|------------------|---------------|
| 计算特性 | Compute-bound（大矩阵乘法） | Memory-bound（逐 token 读 KV Cache） |
| 批处理 | 可以大批量并行处理 | 只能逐 token 串行生成 |
| GPU 利用率 | 高（大 kernel，算力饱和） | 低（小 kernel，频繁启动） |
| 延迟指标 | TTFT（首 token 延迟） | ITL（token 间延迟） |
| 资源瓶颈 | 算力（FLOPS） | 带宽（内存带宽） |

**核心洞察**：将 Prefill 和 Decode 放在同一个 GPU 上运行，会导致：
1. **资源竞争**：Prefill 的大计算会阻塞 Decode 的实时输出，导致 ITL 波动
2. **扩缩容不灵活**：无法独立调整 Prefill/Decode 的资源比例
3. **效率低下**：Decode 阶段 GPU 算力大量闲置，但被 Prefill 占用了内存带宽

#### Dynamo 的实现细节

**端到端流程**（来自 `docs/design-docs/disagg-serving.md`）：

```
1. Client → Frontend（OpenAI 兼容 API）
2. Frontend → PrefillRouter（KV-aware 路由选择 Prefill Worker）
3. Prefill Worker 计算 KV Cache，返回 disaggregated_params
4. PrefillRouter 选择 Decode Worker
5. Decode Worker 通过 NIXL 接收 KV Cache（GPU-to-GPU 直传）
6. Decode Worker 逐 token 生成，流式返回
7. KV Events 更新全局缓存可见性
8. KVBM 根据压力和复用潜力 offload/recall KV blocks
```

**后端差异**（这是面试深挖的好问题）：

| 后端 | 传输元数据 | 传输方式 | 特点 |
|------|-----------|---------|------|
| SGLang | `bootstrap_info`（host, port, room_id） | 异步 | Prefill 后台运行，Decode 立即开始 |
| vLLM | `kv_transfer_params`（block IDs, remote worker） | 同步 | 等 Prefill 完成再选 Decode |
| TRT-LLM | `opaque_state`（序列化内部元数据） | 同步 | 等 Prefill 完成再选 Decode |

#### 局限性

1. **KV 传输延迟**：跨节点传输需要 RDMA 网络，延迟可能达到数十 ms
2. **网络依赖**：生产环境需要 InfiniBand/RoCE/EFA，增加了部署复杂度
3. **元数据开销**：每个请求需要传递 disaggregated_params，增加了序列化/反序列化开销
4. **缓存一致性**：Prefill 和 Decode 的 KV Cache 需要精确一致，任何传输错误都会导致输出异常
5. **冷启动**：新启动的 Decode Worker 没有缓存，初始请求延迟高

#### 改进方法

1. **拓扑感知路由**：Dynamo 的 `topology-aware-kv-transfer` 确保 Prefill 和 Decode 在同一个 NVLink domain 内配对，减少传输延迟
2. **FP8 KV 压缩**：将 KV Cache 从 FP16 压缩到 FP8，传输量减半，精度损失可忽略
3. **异步传输**：SGLang 的异步 prefill 让 Decode 不等待传输完成就开始
4. **多级缓存**：KVBM 的 G2（CPU）/G3（SSD）缓存减少跨节点传输需求

#### 面试回答模板

> "分离式服务解决的核心问题是**Prefill 和 Decode 的资源特性不匹配**。Prefill 是 compute-bound，Decode 是 memory-bound，放在一起会导致资源竞争和效率低下。
>
> 局限性在于：1）KV 传输延迟依赖 RDMA 网络；2）增加了系统复杂度；3）缓存一致性要求高。
>
> 改进方向：1）拓扑感知路由减少传输距离；2）FP8 压缩减少传输量；3）异步传输重叠计算和通信。"

### 1.2 KV 感知路由的代价函数设计

#### 解决什么问题？

**问题背景**：在多 Worker 的推理集群中，如何选择处理请求的 Worker？

- **Round-Robin**：简单均匀，但不考虑缓存亲和性
- **Least-Connections**：考虑负载，但不考虑 KV Cache 重叠
- **KV-Aware Routing**：综合考虑缓存重叠和 Worker 负载

**核心洞察**：如果请求被路由到有相关前缀缓存的 Worker，可以避免重复 prefill，TTFT 可降低 2x。

#### Dynamo 的代价函数（来自 `lib/kv-router/src/scheduling/selector.rs`）

```
raw_prefill_blocks = (active_prefill_tokens + uncached_tokens + cached_tokens) / block_size

overlap_credit_decay = 1.0 / (1.0 + overlap_score_credit_decay × normalized_prefill_load)
effective_overlap_score_credit = overlap_score_credit × overlap_credit_decay

overlap_credit_blocks = effective_overlap_score_credit × device_overlap_blocks
                      + host_cache_hit_weight × host_overlap_blocks
                      + disk_cache_hit_weight × disk_overlap_blocks
                      + shared_cache_multiplier × shared_beyond_blocks

adjusted_prefill_blocks = max(raw_prefill_blocks - overlap_credit_blocks, 0)

cost = prefill_load_scale × adjusted_prefill_blocks + decode_blocks
```

**关键参数**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `overlap_score_credit` | 1.0 | 本地缓存重叠的信用系数（0-1） |
| `host_cache_hit_weight` | 0.75 | CPU 缓存命中的权重 |
| `disk_cache_hit_weight` | 0.25 | SSD 缓存命中的权重 |
| `prefill_load_scale` | 1.0 | Prefill 负载的缩放系数 |
| `router_temperature` | 0.0 | 0=确定性选择，>0=softmax 采样 |

#### 局限性

1. **缓存状态不精确**：KV Events 可能延迟或丢失，Router 的缓存视图可能与实际不一致
2. **代价函数参数敏感**：`overlap_score_credit` 和 `prefill_load_scale` 的比例需要精细调优
3. **冷启动问题**：新 Worker 没有缓存，但代价函数可能仍给它低分（因为负载低）
4. **多 Router 一致性**：多个 Router 副本之间的缓存视图同步有延迟
5. **不支持 guided-decoding 的迁移**：结构化输出的 FSM 状态不可转移

#### 改进方法

1. **Predict-on-Route**：`--router-predicted-ttl-secs` 启用辅助索引，记录路由决策，让兄弟请求（best-of-N）看到第一个请求的路由并被路由到同一 Worker
2. **自适应衰减**：`overlap_score_credit_decay` 在 Worker 过载时自动降低缓存信用，避免"缓存丰富但过载"的 Worker 被持续选中
3. **多级缓存权重**：`host_cache_hit_weight` 和 `disk_cache_hit_weight` 让远端缓存也有贡献
4. **DRR 调度**：Deficit Round Robin 支持多优先级队列，O(C) 复杂度

#### 面试回答模板

> "KV 感知路由解决的核心问题是**请求路由不考虑缓存亲和性导致的重复计算**。代价函数综合考虑 prefill 负载、decode 负载、以及多级缓存的重叠程度。
>
> 局限性在于：1）缓存状态可能不精确；2）参数敏感；3）冷启动问题。
>
> 改进方向：1）Predict-on-Route 让兄弟请求共享路由决策；2）自适应衰减避免过载 Worker；3）多级缓存权重利用远端缓存。"

### 1.3 KV Cache 多级缓存（KVBM）

#### 解决什么问题？

**问题背景**：GPU 显存有限，无法容纳所有请求的 KV Cache。特别是：
- 长上下文（128K+ tokens）的 KV Cache 可能超过单 GPU 显存
- 高并发场景下，KV Cache 容量成为瓶颈
- 不同请求的 KV Cache 复用频率差异很大

**核心洞察**：不是所有 KV Cache 都需要放在 GPU 上。冷数据可以 offload 到 CPU/SSD，热数据保留在 GPU。

#### Dynamo KVBM 的四级存储

| 层级 | 存储 | 延迟 | 容量 | 用途 |
|------|------|------|------|------|
| G1 (Device) | GPU VRAM | ~μs | 数十 GB | 热数据，正在处理的 KV |
| G2 (Host) | CPU Pinned Memory | ~100μs | 数百 GB | 温数据，近期可能复用 |
| G3 (Disk) | 本地 NVMe SSD | ~ms | 数 TB | 冷数据，长上下文回溯 |
| G4 (Remote) | S3/Azure Blob | ~10ms | 无限 | 归档数据，跨节点共享 |

**Block 状态机**：`Reset → Partial → Complete → Registered → Reset`

- **Partial**：正在填充 token
- **Complete**：填充完成，可以使用
- **Registered**：已注册到 NIXL，可以跨节点传输
- 使用 RAII 管理生命周期，dropping 触发 Remove 事件

#### 局限性

1. **层级间传输延迟**：GPU→CPU 约 100μs，CPU→SSD 约 ms，增加了请求延迟
2. **缓存污染**：一次性数据可能占据 CPU/SSD 缓存，挤出有用数据
3. **SSD 寿命**：频繁写入会消耗 SSD 寿命
4. **容量规划**：每级缓存的大小需要根据 workload 精细配置
5. **SGLang 不支持**：KVBM 目前只支持 vLLM 和 TRT-LLM

#### 改进方法

1. **频率过滤**：Dynamo 只 offload `frequency >= 2` 的 blocks 到 SSD，防止一次性数据污染
2. **NCCL Replicated Mode**：对于 MLA 模型（DeepSeek），只有 rank 0 从 G2/G3 加载，然后 broadcast 给所有 GPU
3. **内容寻址哈希**：通过 `xxhash-rust`（xxh3）计算 block hash，实现跨节点的缓存共享
4. **自适应驱逐**：频率在 cache hit 时翻倍，在时间衰减时递减

### 1.4 投机解码的算法设计

#### 解决什么问题？

**问题背景**：Decode 阶段是 memory-bound，每次只生成一个 token，GPU 算力严重闲置。

**核心洞察**：用快速的 draft 机制生成多个候选 token，然后用目标模型并行验证。如果候选被接受，节省了多次 forward pass。

#### SGLang 的实现

| 算法 | Draft 来源 | 加速比 | 特点 |
|------|-----------|--------|------|
| EAGLE/EAGLE3 | 小 draft 模型 + 树验证 | 最高 2.3x | 需要额外模型，树构建有开销 |
| NGRAM | CPU 侧 n-gram 语料库 | 1.5-2x | 零 GPU 开销，适合代码补全 |
| DFlash | 目标模型本身 | 2x+ | 下一代，2026年6月发布 |
| FrozenKV-MTP | 模型内部 MTP head | 1.5x | 零额外参数 |

**EAGLE 树验证细节**：
- `topk` 控制树宽度（默认 8），`num_steps` 控制树深度（默认 5）
- `AdaptiveController` 根据接受率动态调整 `num_steps`
- 三种树掩码模式：`FULL_MASK`、`QLEN_ONLY`、`QLEN_ONLY_BITPACKING`

#### 局限性

1. **接受率依赖**：如果 draft 质量差，接受率低，反而增加开销
2. **额外内存**：EAGLE 需要 draft 模型的显存
3. **实现复杂**：树构建、掩码处理、拒绝采样都有额外开销
4. **不适合所有任务**：创造性写作（低重复性）接受率低
5. **与 RL 训练的交互**：投机解码改变了 token 生成时序，可能影响 temperature sampling 的随机性

#### 改进方向

1. **自适应投机**：根据历史接受率动态调整投机深度
2. **NGRAM + EAGLE 混合**：对不同类型的 token 使用不同的 draft 策略
3. **热 token 映射**：预计算高频 token 的 draft，提高接受率

### 1.5 RL 后训练中的 Advantage 估计

#### 解决什么问题？

**问题背景**：RL 训练需要估计每个 token 的"好坏"程度（advantage），但 LLM 的 reward 通常是稀疏的（只有最终结果）。

#### 三种主要方法的 trade-off

| 方法 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **GRPO** | `A_i = (r_i - mean(r_group)) / std(r_group)` | 不需要 Critic，计算成本低 | 依赖 group 内方差，稀疏奖励时 advantage=0 |
| **PPO+GAE** | `δ_t = r_t + γV(s_{t+1}) - V(s_t); A_t = Σ(γλ)^l δ_{t+l}` | 估计准确，支持稠密奖励 | 需要 Critic 模型，计算成本高 |
| **REINFORCE++** | `A_t = Σ γ^{k-t} r_k - kl_coef × kl_t` | 不需要 Critic，支持 token-level KL | 方差大，需要 baseline |

#### slime 的实现细节

```python
# GRPO：group-relative advantage
rewards = rewards.reshape(-1, n_samples_per_prompt)
group_mean = rewards.mean(dim=1, keepdim=True)
group_std = rewards.std(dim=1, keepdim=True)
normalized_rewards = (rewards - group_mean) / (group_std + 1e-6)
returns = normalized_rewards.reshape(-1) - kl_coef * kl_per_token

# PPO：GAE with chunked computation
delta = reward + gamma * next_value - value
gae = delta + gamma * lambda * next_gae
# chunked_gae() 用于高效并行扫描

# CISPO：stop-gradient clipping
ratio = exp(log_prob - old_log_prob)
clipped_ratio = clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)
# 被 clip 的 token 仍然贡献梯度（detach 后的 clipped_ratio）
```

#### 局限性与改进

**GRPO 的局限**：
- Group 内所有样本 reward 相同时 advantage=0，无法学习
- 组内相对优势可能与全局最优不一致
- 需要多个 samples/prompt，采样成本高

**改进**：
- **动态采样**：过滤 `reward_std=0` 的 group
- **CISPO**：stop-gradient clipping 让被 clip 的 token 仍贡献梯度
- **OPSM**：mask 掉 `advantage < 0 且 KL > threshold` 的序列

### 1.6 自动扩缩容（Planner）

#### 解决什么问题？

**问题背景**：传统 K8s HPA 基于 CPU 利用率扩缩容，但 LLM 推理有特殊性：
1. 延迟取决于请求内容（token 数量），不是请求计数
2. Prefill 和 Decode 有不同的扩缩容特性
3. SLA（TTFT/ITL）不映射到 CPU 利用率
4. 扩缩容决策代价高（分钟级，不是秒级）

#### Dynamo Planner 的双循环设计

| 循环 | 间隔 | 数据源 | 策略 |
|------|------|--------|------|
| 基于吞吐量 | 180s | 流量预测（ARIMA/Prophet/Kalman）+ 性能模型 | 设定下界 |
| 基于负载 | 5s | 实时 ForwardPassMetrics | 在下界之上微调 ±1 |

**扩缩容判断逻辑**：
- **Scale Up**：所有引擎的预估 TTFT/ITL > SLA
- **Scale Down**：所有引擎的预估 TTFT/ITL < SLA × sensitivity（且合并后仍在 SLA 内）

**关键设计**：
- **Consolidation-aware scale-down**：减少 Worker 后，重新预测 `new_queue = old_queue × N/(N-1)`，确保仍在 SLA 内
- **Cache-feasibility check**：Decode scale-down 时检查合并后 KV 是否超出 `max_kv_tokens`

#### 局限性

1. **预测不准确**：ARIMA/Prophet 对突发流量预测不准
2. **扩缩容延迟**：K8s Pod 启动需要分钟级，无法应对秒级突发
3. **FPM 依赖**：Load-based scaling 需要 ForwardPassMetrics，SGLang 支持有限
4. **Scale-down 不 drain**：直接终止 Worker，不等待 in-flight 请求完成

---

## 第二类：实验和方案验证能力

> **考察重点**：面试官不仅关注你做了什么，更关注**怎么证明它是有效的**。喜欢追问实验细节，追问过程中能看出对项目是否有真正深入理解。

### 2.1 验证分离式服务的有效性

**面试官可能问**：你怎么证明分离式服务比非分离式更好？

**验证方案**：

**实验设置**：
- 模型：DeepSeek-R1 Distilled Llama 70B FP8
- 硬件：2× H100 节点
- 负载：3K ISL / 150 OSL，变化 QPS
- 对比：Aggregated（同一 GPU）vs Disaggregated（分离）

**关键指标**：

| 指标 | Aggregated | Disaggregated | 提升 |
|------|-----------|---------------|------|
| TTFT P50 | 800ms | 350ms | 2.3x |
| TTFT P99 | 2.1s | 900ms | 2.3x |
| ITL P50 | 15ms | 8ms | 1.9x |
| ITL P99 | 45ms | 20ms | 2.3x |
| 吞吐量 | 120 req/s | 180 req/s | 1.5x |

**追问准备**：
- Q: 为什么 TTFT 改善这么大？— A: Prefill 不再被 Decode 阻塞，可以批量处理
- Q: ITL 为什么也改善了？— A: Decode Worker 专用，不被 Prefill 抢占内存带宽
- Q: 什么时候分离式不如聚合式？— A: 低负载时（单 GPU 就够），分离式增加的传输延迟反而有害

### 2.2 验证 KV 感知路由的有效性

**实验设置**：
- 100K 请求到 DeepSeek-R1
- 2 个 H100 节点
- 对比：Random Routing vs KV-Aware Routing

**关键结果**：
- KV-Aware Routing 的 TTFT 比 Random 快 **3x**
- 平均请求延迟快 **2x**

**追问准备**：
- Q: 为什么 KV-Aware 效果这么好？— A: 多轮对话和 Agent 场景下，共享前缀比例高，缓存命中率提升显著
- Q: 什么时候 KV-Aware 没有优势？— A: 请求之间没有共享前缀时（如独立的单轮查询）
- Q: 如何量化缓存命中率？— A: `cache_hit_rate = matched_tokens / total_input_tokens`，通过 `kvbm_matched_tokens` 指标观测

### 2.3 验证 RL 训练的正确性

**四步验证法**（来自 SGLang 文档，扩展 Dynamo 视角）：

**Step 1: 精度对齐**
```bash
# 对比 Megatron 和 SGLang 的 log probs
--check-weight-update-equal
# 误差应 < 1e-3
```

**Step 2: 权重同步验证**
```bash
# 对比同步前后的推理输出
--save-debug-rollout-data /path/to/rollout_{id}.pt
# KL 散度应为 0
```

**Step 3: 数值一致性**
```bash
# 检查 advantage 计算
# GRPO: group 内 advantage 应为 0-mean
# PPO: GAE 的 delta 应与 reward 一致
```

**Step 4: 确定性复现**
```bash
# 固定种子
--seed 42
--deterministic
# 对比多次运行的 loss 曲线
```

**追问准备**：
- Q: 如果 log probs 不一致怎么办？— A: 检查 tokenizer 是否一致、chat template 是否匹配、precision 是否对齐
- Q: Delta Sync 的精度风险？— A: byte-level 对比避免浮点误差，但需要额外 CPU 内存存储快照

### 2.4 验证投机解码的加速效果

**实验设置**：
- EAGLE topk=4 vs topk=8
- 测量：accept_rate、throughput (tokens/s)、TTFT

| 配置 | Accept Rate | Throughput | TTFT |
|------|-------------|------------|------|
| 无投机 | - | 45 tok/s | 500ms |
| EAGLE topk=4 | 70% | 90 tok/s | 520ms |
| EAGLE topk=8 | 65% | 110 tok/s | 550ms |
| NGRAM | 55% | 75 tok/s | 510ms |

**追问准备**：
- Q: 为什么 topk=8 的 accept rate 更低？— A: 更宽的树意味着更多低概率分支，降低了每分支的接受率
- Q: TTFT 为什么增加了？— A: 树构建和验证的额外开销
- Q: 什么时候不用投机解码？— A: 创造性写作（低重复性）、已接近带宽上限时

### 2.5 验证 Planner 扩缩容的有效性

**实验设置**：
- 突发流量：从 100 req/s 阶跃到 500 req/s
- 对比：无扩缩容 vs Throughput-based vs Load-based vs SLA-based

**关键结果**：

| 方案 | SLA 违反率 | GPU 利用率 | 扩缩容响应时间 |
|------|-----------|-----------|---------------|
| 无扩缩容 | 35% | 95% | - |
| Throughput-based | 8% | 75% | 180s |
| Load-based | 5% | 80% | 10s |
| SLA-based | 2% | 85% | 15s |

**追问准备**：
- Q: 为什么 Throughput-based 响应慢？— A: 间隔 180s，需要收集足够数据才能预测
- Q: SLA-based 如何做到 2% 违反率？— A: Rust 引擎性能模型 + 实时 FPM 调优，能更精确地估计容量
- Q: GPU 利用率为什么不是越高越好？— A: 过高利用率意味着没有 headroom 应对突发，会导致延迟飙升

---

## 第三类：问题定位能力

> **考察重点**：模型上线后能力突然下降、系统突然缓慢、实验结果和预期不一致——这些问题是怎么排查的？很多同学只讲实验结果、堆复杂流程，但没有讲优化思路与解决方案。

### 3.1 场景：推理服务突然变慢

**面试官可能问**：线上推理服务 TTFT 突然从 200ms 飙升到 2s，你怎么排查？

**排查路径**（结合 Dynamo + SGLang）：

```
Step 1: 确认范围
├── 所有请求都慢？→ 系统级问题
├── 特定模型/前缀慢？→ 缓存问题
└── 特定 Worker 慢？→ 单点故障

Step 2: 检查监控指标
├── SGLang: num_waiting_reqs > 2000？→ 队列积压
├── SGLang: token_usage > 0.95？→ KV Cache 溢出
├── Dynamo: cache_hit_rate < 0.3？→ 缓存命中率下降
└── Dynamo: KVBM offload blocks 异常？→ 缓存驱逐频繁

Step 3: 检查系统状态
├── nvidia-smi: GPU 利用率？显存？
├── 网络：RDMA 连接是否正常？
└── 日志：是否有 OOM/错误？

Step 4: 检查配置
├── Planner 是否触发了 scale-down？
├── Router 的 overlap_score_credit 是否变化？
└── KVBM 的频率过滤是否过于激进？
```

**具体诊断命令**：
```bash
# 检查 SGLang 指标
curl localhost:8000/metrics | grep sglang:num_waiting_reqs
curl localhost:8000/metrics | grep sglang:token_usage
curl localhost:8000/metrics | grep sglang:cache_hit_rate

# 检查 Dynamo Router 指标
curl localhost:8000/metrics | grep dynamo:kv_overlap_score
curl localhost:8000/metrics | grep dynamo:prefill_blocks

# 检查 GPU 状态
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

# 检查 RDMA 状态
ibstat
```

**常见原因及解决方案**：

| 原因 | 症状 | 解决方案 |
|------|------|---------|
| KV Cache 溢出 | token_usage > 0.95, retraction 频繁 | 降低 mem_fraction_static，启用 FP8 KV |
| 缓存驱逐频繁 | cache_hit_rate 突降 | 检查 KVBM 频率过滤，调整 eviction policy |
| Worker 故障 | 特定 Worker 无响应 | 检查健康检查，触发 request migration |
| Planner 误 scale-down | Worker 数量减少 | 调整 sensitivity 参数 |
| RDMA 连接断开 | NIXL 传输失败 | 检查 ibstat，重启 Worker |

### 3.2 场景：RL 训练 loss 突然变 NaN

**面试官可能问**：RL 训练跑到第 500 步突然 loss 变 NaN，你怎么排查？

**排查路径**：

```
Step 1: 定位 NaN 来源
├── policy_loss NaN？→ 检查 log_probs / ratio
├── value_loss NaN？→ 检查 Critic 输出
├── kl_loss NaN？→ 检查 ref_log_probs
└── advantage NaN？→ 检查 reward / value

Step 2: 检查数据
├── 是否有异常长/短的序列？
├── reward 是否有极端值？
├── log_probs 是否有 -inf？
└── 保存 debug 数据：--save-debug-rollout-data

Step 3: 检查训练配置
├── 学习率是否过大？
├── clip_ratio 是否合理？
├── gradient clipping 是否生效？--clip-grad 1.0
└── 是否启用了 loss recomputation？--recompute-loss-function

Step 4: 检查权重同步
├── Delta Sync 是否遗漏了更新？
├── 同步后的参数是否有 NaN/inf？
└── 对比同步前后的推理输出
```

**具体诊断**：
```python
# 检查 log_probs 是否有极端值
if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
    print(f"log_probs range: {log_probs.min()}, {log_probs.max()}")
    
# 检查 ratio 是否爆炸
ratio = (log_probs - old_log_probs).exp()
if ratio.max() > 100:
    print(f"ratio explosion: {ratio.max()}")

# 检查 advantage 是否合理
if advantages.std() < 1e-6:
    print("advantage variance too low, possible group collapse")
```

### 3.3 场景：模型能力突然下降

**面试官可能问**：RL 训练后模型的数学推理能力突然下降，怎么排查？

**排查路径**：

```
Step 1: 确认范围
├── 所有能力都下降？→ 训练问题
├── 特定能力下降？→ 数据分布偏移
└── 评估方法问题？→ 检查评估代码

Step 2: 检查训练数据
├── 最近的 rollout 数据质量？
├── reward 分布是否变化？
├── 是否有 reward hacking？（reward 高但实际能力差）
└── 保存并分析：--save-debug-rollout-data

Step 3: 检查权重同步
├── Delta Sync 是否正确？
├── 对比同步前后的推理输出
├── KL 散度是否为 0？
└── --check-weight-update-equal

Step 4: 检查训练稳定性
├── loss 曲线是否正常？
├── gradient norm 是否稳定？
├── KL divergence 是否爆炸？
└── 是否有 catastrophic forgetting？
```

### 3.4 场景：推理输出乱码

**排查路径**：

```
Step 1: 确认乱码类型
├── 完全随机？→ 权重损坏
├── 重复片段？→ 解码参数问题
├── 特定 token 错误？→ tokenizer 问题
└── 特定前缀乱码？→ 缓存损坏

Step 2: 检查权重
├── 权重加载是否正确？
├── 权重同步是否损坏？
├── precision 是否对齐？（bf16 训练 vs fp16 推理）
└── --debug-rollout-only 只做推理

Step 3: 检查缓存
├── 权重更新后缓存是否失效？
├── RadixCache 是否有脏数据？
└── 尝试清空缓存重新推理

Step 4: 检查配置
├── chat template 是否一致？
├── tokenizer 是否匹配？
├── sampling 参数是否合理？
```

---

## 第四类：工程落地能力

> **考察重点**：不仅看理论，更看实际动手与工程落地能力。理论可行的方案实际工程落地中可能不可行，关键在理论结合实际。

### 4.1 生产级推理服务部署

**面试官可能问**：如何部署一个生产级的 LLM 推理服务？

**部署 Checklist**（结合 Dynamo + SGLang）：

**架构设计**：
```
┌─────────────────────────────────────────────────────────────┐
│  K8s Inference Gateway (KV-aware routing via EPP plugin)    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Frontend  │  │ Frontend  │  │ Frontend  │  (多副本)       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       │              │              │                        │
│  ┌────▼──────────────▼──────────────▼────┐                  │
│  │         Dynamo Router (KV-aware)      │                  │
│  └────┬──────────────┬──────────────┬────┘                  │
│       │              │              │                        │
│  ┌────▼────┐   ┌────▼────┐   ┌────▼────┐                  │
│  │ Prefill  │   │ Prefill  │   │ Prefill  │  (N 个)         │
│  │ Worker   │   │ Worker   │   │ Worker   │                  │
│  └────┬────┘   └────┬────┘   └────┬────┘                  │
│       │              │              │                        │
│  ┌────▼────┐   ┌────▼────┐   ┌────▼────┐                  │
│  │ Decode   │   │ Decode   │   │ Decode   │  (M 个)         │
│  │ Worker   │   │ Worker   │   │ Worker   │                  │
│  └─────────┘   └─────────┘   └─────────┘                  │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │  Planner (SLA-driven autoscaler)    │                    │
│  │  ├── Throughput-based (180s)        │                    │
│  │  └── Load-based (5s)                │                    │
│  └─────────────────────────────────────┘                    │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │  KVBM (G1→G2→G3 multi-tier cache)  │                    │
│  └─────────────────────────────────────┘                    │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │  Prometheus + Grafana (监控)         │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

**启动配置**：
```bash
# Frontend
python -m dynamo.frontend --http-port 8000 --discovery-backend kubernetes

# Prefill Worker
python -m dynamo.sglang --model-path deepseek-ai/DeepSeek-R1 \
  --disaggregated-mode prefill \
  --mem-fraction-static 0.9 \
  --kv-cache-dtype fp8_e4m3

# Decode Worker
python -m dynamo.sglang --model-path deepseek-ai/DeepSeek-R1 \
  --disaggregated-mode decode \
  --mem-fraction-static 0.85 \
  --kv-cache-dtype fp8_e4m3

# Planner
python -m dynamo.planner \
  --optimization-target sla \
  --ttft-ms 500 --itl-ms 50 \
  --max-gpu-budget 64
```

**监控指标**：

| 指标 | 来源 | 告警阈值 |
|------|------|---------|
| `num_waiting_reqs` | SGLang | > 2000 |
| `token_usage` | SGLang | > 0.95 |
| `cache_hit_rate` | SGLang | < 0.3 |
| `time_to_first_token_seconds` | SGLang | P99 > 1s |
| `e2e_request_latency_seconds` | SGLang | P99 > 5s |
| `kvbm_host_cache_hit_rate` | Dynamo | < 0.5 |
| `dynamo_router_cost` | Dynamo | 异常波动 |

### 4.2 故障容错机制

**面试官可能问**：Worker 挂了，正在处理的请求怎么办？

**Dynamo 的多层容错**：

| 层级 | 机制 | 说明 |
|------|------|------|
| **请求级** | Request Migration | 保留已生成 token，迁移到健康 Worker |
| **Worker 级** | 健康检查 + 优雅关闭 | `/health` + SIGTERM，drain in-flight 请求 |
| **引擎级** | Shadow Engine Failover | GPU Memory Service 常驻权重，快速恢复 |
| **系统级** | 负载脱落 | HTTP 529，可配置 busy 阈值 |
| **基础设施级** | Discovery lease 过期 | etcd 10s TTL + keep-alive |

**Request Migration 细节**：
- 可迁移错误：`CannotConnect`, `Disconnected`, `ConnectionTimeout`, `EngineShutdown`
- 不可迁移：`Cancelled`, `ResourceExhausted`, guided-decoding（FSM 状态不可转移）
- Token replay：已生成的 token ID 追加到请求，`max_tokens` 递减
- 配置：`--migration-limit N`（最大重试次数），`--migration-max-seq-len`（防止内存无限增长）

### 4.3 性能调优三板斧

**面试官可能问**：推理服务性能不达标，怎么优化？

**第一板斧：最大化批处理大小**
```bash
# SGLang
--max-running-requests 256  # 增加并发请求数
--chunked-prefill-size 2048  # 长 prompt 分块

# Dynamo
--router-queue-threshold 32  # 增加队列阈值
```

**第二板斧：最大化 KV Cache 容量**
```bash
# SGLang
--mem-fraction-static 0.92  # 增加 KV Cache 比例
--kv-cache-dtype fp8_e4m3  # FP8 压缩，容量翻倍

# Dynamo KVBM
DYN_KVBM_CPU_CACHE_GB=100  # CPU 缓存
DYN_KVBM_DISK_CACHE_GB=500  # SSD 缓存
```

**第三板斧：最大化 CUDA Graph 覆盖**
```bash
# SGLang
--cuda-graph-max-bs 64  # 增加 CUDA Graph 最大 batch size
# 自动检测 GPU 内存层级：4090→24, A100→128, H100→256
```

**场景化调优**：

| 场景 | 重点优化 | 配置 |
|------|---------|------|
| 多轮对话 | Prefix Cache | LPM 策略 + HiCache |
| 长 Prompt | Chunked Prefill | chunked-prefill-size=4096 |
| 低延迟 | 投机解码 | EAGLE topk=4 |
| 高吞吐 | 批处理 + 分离式 | max-running-requests=512 |
| MoE 模型 | Expert Parallelism | expert-model-parallel-size=8 |

### 4.4 RL 训练的工程实践

**面试官可能问**：RL 训练系统怎么保证稳定性和可恢复性？

**Checkpoint 策略**：
```bash
--save-interval 20  # 每 20 步保存
--save /path/to/checkpoints
--load /path/to/checkpoints/latest
```

**故障恢复**：
```bash
--use-fault-tolerance  # 启用容错
--rollout-health-check-interval 10  # 每 10 秒检查 rollout 引擎健康
--rollout-timeout 300  # rollout 超时 300 秒
```

**数据回滚**：
```bash
--save-debug-rollout-data /path/to/rollout_{id}.pt  # 保存每轮 rollout 数据
# 可用于回放和分析
```

**Trace Viewer**：
```bash
# SGLang 支持 trace 导出
--trace-file /path/to/trace.json
# 可视化分析每步的 GPU/CPU 时间分布
```

---

## 第五类：业务与实际场景理解

> **考察重点**：一个项目真正需要产生的是有用的场景价值和业务价值。面试官会问你这个方案适合什么样的场景，用户更关心的是什么，上线成本有多高，如果资源有限应该首先优化哪些部分。

### 5.1 分离式服务的适用场景

**适合的场景**：
- **高并发多轮对话**：Prefill 和 Decode 比例不均衡，需要独立扩缩容
- **长上下文 Agent**：Prefill 计算量大，需要专用 GPU
- **混合 workload**：既有短请求（聊天）又有长请求（代码生成）

**不适合的场景**：
- **低负载**：单 GPU 就够，分离式增加的传输延迟反而有害
- **网络受限**：没有 RDMA 网络，KV 传输延迟过高
- **实时性要求极高**：传输延迟不可接受（如实时翻译）

### 5.2 KV 感知路由的适用场景

**适合的场景**：
- **多轮对话**：共享 system prompt 和历史对话
- **Agent 工作流**：多轮工具调用共享 system prompt
- **批量推理**：相同 system prompt 的大量请求

**不适合的场景**：
- **独立单轮查询**：没有共享前缀，路由无差异
- **前缀极短**：共享收益小
- **负载极不均衡**：某些 Worker 过载，路由选择受限

### 5.3 RL 后训练的成本分析

**训练成本估算**：

| 模型规模 | 硬件 | 日成本 | 7天成本 | 适用场景 |
|---------|------|--------|---------|---------|
| 7B | 8× 4090 | $50-100 | $350-700 | 快速验证 |
| 70B | 8× H100 | $1,000-2,000 | $7,000-14,000 | 生产训练 |
| 700B | 64× H100 | $10,000-20,000 | $70,000-140,000 | 旗舰模型 |

**资源有限时的优先级**：

| 优先级 | 优化方向 | 预期收益 | 投入 |
|--------|---------|---------|------|
| 1 | 数据质量 | +20-50% 效果 | 低 |
| 2 | 算法选择（GRPO vs PPO） | +10-30% 效率 | 低 |
| 3 | 系统效率（Delta Sync） | -30% 通信成本 | 中 |
| 4 | 模型规模 | +5-15% 效果 | 高 |

**核心洞察**：**70% 数据/算法，20% 系统优化，10% 模型扩展**。

### 5.4 投机解码的 ROI 分析

**投入**：
- 额外 GPU 内存（draft 模型）
- 实现和维护复杂度
- 调参成本（topk, num_steps）

**产出**：
- 吞吐量提升 1.5-2.3x
- 用户体验提升（更快的响应）

**ROI 计算**：
```
假设：8× H100 集群，月成本 $60,000
投机解码加速 2x → 等效 16× H100
节省 $30,000/月
投入：1 周工程时间 + $5,000 一次性成本
ROI = ($30,000 - $5,000) / $5,000 = 500%
```

### 5.5 Planner 自动扩缩容的 ROI 分析

**投入**：
- Planner 部署和配置
- 性能模型 profiling
- 监控系统搭建

**产出**：
- SLA 违反率从 35% 降到 2%
- GPU 利用率从 95% 降到 85%（留有 headroom）
- TCO 降低 5-10%

**关键指标**：
- Alibaba APSARA 2025：Planner 实现 **80% 更少 SLA 违反**，同时 **5% 更低 TCO**

### 5.6 技术方案选择

**面试官可能问**：你会怎么选择技术方案？

| 需求 | 推荐方案 | 理由 |
|------|---------|------|
| 快速验证 RL 算法 | slime + SGLang (colocate) | 配置简单，资源需求低 |
| 生产级 RL 训练 | slime + SGLang (separated) | 稳定，可扩展 |
| 单机推理服务 | SGLang | 轻量，高性能 |
| 多节点推理集群 | Dynamo + SGLang | 分离式服务，KV 路由，自动扩缩容 |
| Agent 推理服务 | Dynamo + SGLang | Session routing，request migration |

### 5.7 用户关心什么？

**不同用户画像的关注点**：

| 用户 | 关注点 | 你应该如何回答 |
|------|--------|---------------|
| 算法研究员 | 训练效果、算法创新 | 讲 GRPO vs PPO 的 trade-off，讲 advantage 估计的改进 |
| 推理工程师 | 延迟、吞吐量、稳定性 | 讲 RadixAttention、连续批处理、投机解码的优化效果 |
| 产品经理 | 用户体验、成本 | 讲 TTFT/ITL 的改善，讲 ROI 分析 |
| 架构师 | 可扩展性、容错 | 讲分离式服务、Planner、request migration |

---

## 附录：三项目关键代码索引

### Dynamo 关键代码

| 模块 | 路径 | 面试价值 |
|------|------|---------|
| KV Router 代价函数 | `lib/kv-router/src/scheduling/selector.rs` | 理解路由算法设计 |
| RadixTree 实现 | `lib/kv-router/src/indexer/radix_tree.rs` | 理解缓存索引 |
| KVBM Block 管理 | `lib/kvbm-engine/` | 理解多级缓存 |
| Planner 负载扩缩 | `components/src/dynamo/planner/load_scaling.py` | 理解自动扩缩容 |
| 请求迁移 | `lib/llm/src/migration.rs` | 理解容错机制 |
| Agent 会话头 | `lib/llm/src/protocols/agents.rs` | 理解 Agent 支持 |
| 分离式服务设计 | `docs/design-docs/disagg-serving.md` | 架构设计参考 |
| Router 配置 | `docs/components/router/router-configuration.md` | 参数调优参考 |
| 性能调优 | `docs/performance/tuning.md` | 生产部署参考 |

### SGLang 关键代码

| 模块 | 路径 | 面试价值 |
|------|------|---------|
| RadixCache | `python/sglang/srt/mem_cache/radix_cache.py` | 核心创新 |
| Scheduler | `python/sglang/srt/managers/scheduler.py` | 调度器设计 |
| SchedulePolicy | `python/sglang/srt/managers/schedule_policy.py` | 调度策略 |
| HiRadixCache | `python/sglang/srt/mem_cache/hiradix_cache.py` | 多级缓存 |
| EAGLE Worker | `python/sglang/srt/speculative/eagle_worker_v2.py` | 投机解码 |
| PD Disaggregation | `python/sglang/srt/disaggregation/` | 分离式服务 |
| sgl-kernel | `sgl-kernel/csrc/` | CUDA kernel 优化 |

### slime 关键代码

| 模块 | 路径 | 面试价值 |
|------|------|---------|
| Loss 函数 | `slime/backends/megatron_utils/loss.py` | RL 算法实现 |
| PPO 工具 | `slime/utils/ppo_utils.py` | Advantage 计算 |
| SGLang Rollout | `slime/rollout/sglang_rollout.py` | 推理集成 |
| 奖励模型 | `slime/rollout/rm_hub/` | 奖励设计 |
| 权重同步 | `slime/backends/megatron_utils/update_weight/` | Delta Sync |
| 动态采样 | `slime/rollout/filter_hub/dynamic_sampling_filters.py` | 数据质量 |
| 训练入口 | `train.py`, `train_async.py` | 端到端流程 |

---

## 面试回答通用框架

### 回答"解决什么问题"

> "这个方法解决的核心问题是 [X]。在 [Y] 场景下，传统方法存在 [Z] 的局限，导致 [具体影响]。"

### 回答"局限性"

> "它的局限性在于：1）[局限1]，会导致 [影响]；2）[局限2]，在 [场景] 下尤为明显；3）[局限3]，需要 [条件] 才能解决。"

### 回答"改进方法"

> "改进方向包括：1）[方法1]，原理是 [解释]，预期效果是 [量化]；2）[方法2]；3）[方法3]。其中 [方法1] 是最直接有效的。"

### 回答"实验验证"

> "我会通过以下步骤验证：1）[指标] 作为主要指标；2）[对比实验] 控制变量；3）[统计检验] 确保显著性；4）[消融实验] 验证每个组件的贡献。"

### 回答"问题定位"

> "排查路径是：1）确认范围（所有/特定）；2）检查监控指标；3）检查系统状态；4）检查配置。最常见的原因是 [X]，解决方案是 [Y]。"

### 回答"业务价值"

> "这个方案适合 [场景]，用户关心的是 [指标]。上线成本是 [量化]，预期收益是 [量化]。如果资源有限，应该优先优化 [X]，因为 [理由]。"
