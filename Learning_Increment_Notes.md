# Dynamo 学习增量笔记

> **使用方法**：每个阶段复现完成后，记录"前两轮认知"（已有理解）和"新增理解"（通过动手复现新学到的点）。重点记录**从"知道概念"到"理解为什么"的转变**。
>
> **参考格式**：沿用 `sglang/Learning_Increment_Notes.md` 的"前两轮认知 → 新增理解"结构。

---

## 目录

- [阶段一：环境搭建与基础验证](#阶段一环境搭建与基础验证)
- [阶段二：Dynamo 核心功能复现](#阶段二dynamo-核心功能复现)
- [阶段三：分离式服务与 KV 管理](#阶段三分离式服务与-kv-管理)
- [阶段四：Agent 推理与 Tool Calling](#阶段四agent-推理与-tool-calling)
- [阶段五：Dynamo + slime 联动](#阶段五dynamo--slime-联动)
- [阶段六：性能调优与总结](#阶段六性能调优与总结)
- [跨阶段综合洞察](#跨阶段综合洞察)

---

## 阶段一：环境搭建与基础验证

### 1.1 Dynamo 三平面架构

**前两轮认知**：
- Dynamo 有 Request Plane、Control Plane、Storage Plane
- Request Plane 处理请求，Control Plane 管理扩缩容，Storage Plane 管理 KV Cache

**新增理解**：
> （动手后填写）
>
> 例如：
> - Request Plane 的默认传输是 TCP（不是 NATS），NATS 只是可选的消息总线
> - Discovery backend 可以用 `file` 模式，完全不需要 etcd
> - Frontend 是 Rust 实现的，通过 PyO3 暴露给 Python

### 1.2 后端集成方式

**前两轮认知**：
- Dynamo 支持 vLLM、SGLang、TensorRT-LLM 三个后端
- 每个后端是一个 Python 模块

**新增理解**：
> （动手后填写）
>
> 例如：
> - SGLang 后端有 10 种 worker 类型（decode、prefill、embedding、multimodal、diffusion 等）
> - vLLM 的 KV events 通过 ZMQ 发布，不需要 NATS
> - `--discovery-backend file` 让单机开发完全不需要外部服务

### 1.3 Mocker 的设计

**前两轮认知**：
- Mocker 是模拟引擎，用于测试

**新增理解**：
> （动手后填写）
>
> 例如：
> - Mocker 可以模拟多 Worker 的行为，用于测试 Router 而不需要真实 GPU
> - Mocker 使用 `aiconfigurator` 进行性能建模
> - 可以用 Mocker 学习 Dynamo 的架构而不需要等待模型下载

---

## 阶段二：Dynamo 核心功能复现

### 2.1 KV-Aware Routing 代价函数

**前两轮认知**：
- 代价函数考虑 prefill 负载、decode 负载、缓存重叠
- 选择 cost 最低的 Worker

**新增理解**：
> （动手后填写）
>
> 例如：
> - `overlap_score_credit_decay` 参数的作用：当 Worker 过载时，自动降低缓存重叠的信用，避免"缓存丰富但过载"的 Worker 被持续选中
> - `router_temperature > 0` 时使用 softmax 采样，引入随机性用于负载均衡
> - Predict-on-Route 机制让 best-of-N 的多个请求被路由到同一 Worker

### 2.2 SGLang vs vLLM 在 Dynamo 中的差异

**前两轮认知**：
- SGLang 和 vLLM 是两个不同的推理引擎
- Dynamo 对它们有统一的接口

**新增理解**：
> （动手后填写）
>
> 例如：
> - SGLang 的 prefill 是异步的（后台运行），vLLM 的 prefill 是同步的
> - vLLM 需要 `--kv-events-config` 来启用 KV 事件，SGLang 不需要
> - SGLang 的 `--mem-fraction-static` 和 vLLM 的 `--gpu-memory-utilization` 是同一概念的不同命名

### 2.3 Prefix Cache 在 Dynamo 中的体现

**前两轮认知**：
- SGLang 的 RadixAttention 管理单机的 prefix cache
- Dynamo 的 KV Router 基于缓存重叠做路由

**新增理解**：
> （动手后填写）
>
> 例如：
> - Dynamo Router 维护一个全局的 Radix Tree 索引，跟踪每个 Worker 上的缓存状态
> - KV events 从 Worker 发布到 Router，Router 据此更新索引
> - 当 `--no-router-kv-events` 启用时，Router 用预测模式（基于路由决策推断缓存状态）

### 2.4 投机解码在 Dynamo 中的集成

**前两轮认知**：
- 投机解码用 draft model 生成候选，target model 验证
- EAGLE 用树验证

**新增理解**：
> （动手后填写）
>
> 例如：
> - Dynamo 的 Planner 在计算 ITL 时会考虑 accept_length：`est_ms = est * 1000 / accept_length`
> - 投机解码不影响 KV 路由，因为路由只看 input tokens，不看 draft tokens
> - 在 4090 上，EAGLE 的额外显存开销（draft model）可能不适合 24GB 限制

---

## 阶段三：分离式服务与 KV 管理

### 3.1 分离式服务的传输机制

**前两轮认知**：
- Prefill 计算 KV Cache，传输给 Decode
- 通过 NIXL 进行 GPU-to-GPU 传输

**新增理解**：
> （动手后填写）
>
> 例如：
> - 在同节点 PCIe 上，UCX 使用 `cuda_ipc` 进行 GPU 间直接传输，带宽 ~32 GB/s
> - `UCX_TLS=cuda_copy,cuda_ipc` 是同节点分离的关键配置
> - SGLang 的 bootstrap handshake 在分离式服务中建立 RDMA 连接
> - 同 GPU 分离（disagg_same_gpu）通过 memory fraction 切分 VRAM，适合开发测试

### 3.2 KVBM 多级缓存的实际效果

**前两轮认知**：
- KVBM 有 G1（GPU）、G2（CPU）、G3（SSD）、G4（Remote）四级存储
- 冷数据 offload 到 CPU/SSD

**新增理解**：
> （动手后填写）
>
> 例如：
> - G2 使用 CUDA D2H copy（`cudaMemcpy`），延迟 ~100μs
> - 频率过滤：只有 `frequency >= 2` 的 blocks 才会被 offload 到 SSD
> - SSD 寿命保护：避免一次性数据污染 SSD
> - Block 状态机：`Reset → Partial → Complete → Registered → Reset`
> - MLA 模型（DeepSeek）需要 NCCL Replicated Mode：rank 0 加载，broadcast 给所有 GPU

### 3.3 4090 上分离式服务的实际表现

**前两轮认知**：
- PCIe 带宽 ~32 GB/s，比 NVLink 慢
- 分离式服务在 4090 上可能不如聚合式

**新增理解**：
> （动手后填写）
>
> 例如：
> - 对于小模型（0.6B-8B），KV Cache 很小，PCIe 传输延迟可忽略
> - 对于 32B FP8 模型，KV Cache 传输量约为 `num_layers × seq_len × hidden_dim × 2 × FP8`
> - 当 KV Cache 传输时间 > prefill 计算时间时，分离式不如聚合式
> - 调优建议：短上下文用聚合式，长上下文（>4K tokens）考虑分离式

---

## 阶段四：Agent 推理与 Tool Calling

### 4.1 Tool Call Parser 的设计

**前两轮认知**：
- Dynamo 支持 20+ 种 tool call parser
- 不同模型有不同的工具调用格式

**新增理解**：
> （动手后填写）
>
> 例如：
> - Parser 分为两类：JSON 格式（hermes、llama3_json）和 XML 格式（某些模型用 XML 标签）
> - Reasoning parsing 在 tool call parsing 之前执行
> - `--dyn-tool-call-parser` 配置在 backend worker 上，不是 frontend
> - Structural tags 使用 xgrammar 实现 guided decoding

### 4.2 Session Routing 的实现

**前两轮认知**：
- 同一会话的请求应该路由到同一 Worker
- 通过 `X-Dynamo-Session-ID` header 实现

**新增理解**：
> （动手后填写）
>
> 例如：
> - Session affinity 是"尽力而为"的（passive metadata），不保证 100% 亲和
> - `--router-session-affinity-ttl-secs` 控制亲和性的 TTL
> - 当目标 Worker 过载时，Router 可能选择其他 Worker
> - Claude Code、Codex、OpenCode 各有自己的 session header 格式

### 4.3 Agent Hints

**前两轮认知**：
- Dynamo 支持 per-request 的 agent hints
- 可以设置 priority、expected output length

**新增理解**：
> （动手后填写）
>
> 例如：
> - `priority` 影响 Router 的 DRR 调度和 Backend 的引擎调度
> - `strict_priority` 创建独立的队列层级
> - `osl`（expected output length）帮助 Router 预估 decode 负载
> - `speculative_prefill` 允许 Router 预取下一 turn 的 prefix

---

## 阶段五：Dynamo + slime 联动

### 5.1 slime 如何使用推理引擎

**前两轮认知**：
- slime 用 SGLang 做 rollout
- 权重从 Megatron 同步到 SGLang

**新增理解**：
> （动手后填写）
>
> 例如：
> - slime 可以通过 HTTP API 使用 Dynamo Frontend 作为推理入口
> - Dynamo 的 KV-aware routing 让同一 prompt 的多个 GRPO samples 被路由到同一 Worker
> - Mocker 可以用于 RL 训练流程的端到端测试，不需要真实 GPU

### 5.2 RL 训练中的资源管理

**前两轮认知**：
- Colocate 模式共享 GPU，Separated 模式分离 GPU
- 需要权衡资源利用率和通信开销

**新增理解**：
> （动手后填写）
>
> 例如：
> - 在 4090 上，Colocate 模式需要 `--sglang-mem-fraction-static 0.6` 留出空间给训练
> - Separated 模式在 8 卡上可以 4+4 分配（4 rollout + 4 training）
> - Delta Sync 的 ~3% 参数密度在 PCIe 上传输很快

---

## 阶段六：性能调优与总结

### 6.1 4090 特有的调优点

**前两轮认知**：
- 4090 没有 NVLink，TP 通信走 PCIe
- 需要更保守的显存分配

**新增理解**：
> （动手后填写）
>
> 例如：
> - `--mem-fraction-static 0.92` 在 4090 上可能太激进，因为 PCIe 通信也需要显存
> - `--cuda-graph-max-bs 24` 是 4090 的自适应值（SGLang 自动检测）
> - `--chunked-prefill-size 2048` 是 4090 的默认值
> - TP=2 在 PCIe 上的通信开销约为 NVLink 的 3-5x

### 6.2 监控指标的实际含义

**前两轮认知**：
- 有 TTFT、ITL、cache_hit_rate 等指标
- 通过 Prometheus 暴露

**新增理解**：
> （动手后填写）
>
> 例如：
> - `cache_hit_rate` 的计算方式：`matched_tokens / total_input_tokens`
> - `token_usage` 接近 1.0 时会触发 retraction（请求回退）
> - `num_waiting_reqs > 2000` 时系统开始拒绝请求（HTTP 529）
> - Grafana dashboard 可以直观看到 prefill/decode 的负载分布

### 6.3 故障容错的实际效果

**前两轮认知**：
- Request Migration 可以迁移 in-flight 请求
- 健康检查检测 Worker 故障

**新增理解**：
> （动手后填写）
>
> 例如：
> - Migration 不支持 guided-decoding（FSM 状态不可转移）
> - Migration 不支持 n>1（多采样请求）
> - Token replay 将已生成的 token 追加到新请求，`max_tokens` 递减
> - 优雅关闭的 grace period 默认 5s，可以通过 `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS` 调整

---

## 跨阶段综合洞察

### 洞察 1：Dynamo 的设计哲学

**前两轮认知**：
- Dynamo 是推理编排层，不替代推理引擎
- 用 Rust 写性能敏感部分，Python 写集成部分

**新增理解**：
> （动手后填写）
>
> 例如：
> - Dynamo 的核心价值是**集群级别的资源管理**，而不是单机优化
> - 单机场景下，直接用 SGLang/vLLM 更高效；Dynamo 的价值在多 Worker、多节点
> - Rust + Python 的 split 让热路径（routing、indexing）在 Rust 中运行，冷路径（配置、集成）在 Python 中

### 洞察 2：从 SGLang 到 Dynamo 的层级关系

**前两轮认知**：
- SGLang 是推理引擎，Dynamo 是编排层
- slime 用 SGLang 做 rollout

**新增理解**：
> （动手后填写）
>
> 例如：
> - SGLang 管理单机内的 KV Cache（RadixAttention），Dynamo 管理集群级别的 KV Cache（KVBM + Router）
> - SGLang 的 prefix cache 是"本地视图"，Dynamo 的 KV Router 是"全局视图"
> - slime 的 rollout 通过 SGLang 的 HTTP API 工作，Dynamo 可以作为这个 API 的前置代理

### 洞察 3：4090 上的最佳实践

**前两轮认知**：
- 4090 适合中小模型（0.6B-8B 单卡，32B 多卡）
- 没有 NVLink 限制了 TP 和分离式服务

**新增理解**：
> （动手后填写）
>
> 例如：
> - 最佳实践：用多个独立 Worker + KV 路由，而不是大 TP
> - 8 卡 4090 最佳配置：4 个 TP=2 的 Qwen3-32B-FP8 Worker + KV 路由
> - 分离式服务在 4090 上的价值有限，除非是长上下文场景
> - KVBM 的 CPU offload 在 4090 上特别有价值（弥补 24GB VRAM 的限制）

---

## 学习检查清单

### 阶段一完成后
- [ ] 能解释 Dynamo 的三平面架构
- [ ] 能区分 SGLang/vLLM/TRT-LLM 后端的集成方式
- [ ] 能用 Mocker 进行无 GPU 测试

### 阶段二完成后
- [ ] 能解释 KV Router 的代价函数
- [ ] 能配置 KV-Aware Routing
- [ ] 能对比 SGLang 和 vLLM 在 Dynamo 中的差异
- [ ] 能运行投机解码

### 阶段三完成后
- [ ] 能解释分离式服务的传输机制（NIXL/UCX）
- [ ] 能配置同节点分离式服务
- [ ] 能配置 KVBM 的 CPU/SSD 缓存
- [ ] 能分析分离式 vs 聚合式的适用场景

### 阶段四完成后
- [ ] 能配置 Tool Calling 和 Reasoning Parser
- [ ] 能解释 Session Routing 的实现
- [ ] 能理解 Agent Hints 的作用

### 阶段五完成后
- [ ] 能解释 slime 如何使用 Dynamo Frontend
- [ ] 能配置 RL 训练中的推理资源分配
- [ ] 理解 Colocate vs Separated 的 trade-off

### 阶段六完成后
- [ ] 能进行 4090 特有的性能调优
- [ ] 能搭建 Prometheus + Grafana 监控
- [ ] 能解释关键监控指标的含义
- [ ] 能分析故障容错的实际效果
