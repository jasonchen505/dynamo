# Dynamo 8x RTX 4090 全流程复现计划

> **目标**：用 8 卡 RTX 4090（24GB VRAM/卡，PCIe，无 NVLink）完整复现 Dynamo 的核心功能，覆盖 LLM 推理、Agent 推理、RL 后训练三个方向。
>
> **前置条件**：已完成 `sglang/Reproduction_Plan_8x4090.md` 和 `slime/Reproduction_Plan_8x4090.md` 中的基础环境搭建。

---

## 目录

- [一、硬件资源评估与限制分析](#一硬件资源评估与限制分析)
- [二、复现总览与阶段规划](#二复现总览与阶段规划)
- [三、阶段一：环境搭建与基础验证（2-3天）](#三阶段一环境搭建与基础验证2-3天)
- [四、阶段二：Dynamo 核心功能复现（4-5天）](#四阶段二dynamo-核心功能复现4-5天)
- [五、阶段三：分离式服务与 KV 管理（3-4天）](#五阶段三分离式服务与-kv-管理3-4天)
- [六、阶段四：Agent 推理与 Tool Calling（3-4天）](#六阶段四agent-推理与-tool-calling3-4天)
- [七、阶段五：Dynamo + slime 联动（3-4天）](#七阶段五dynamo--slime-联动3-4天)
- [八、阶段六：性能调优与总结（2-3天）](#八阶段六性能调优与总结2-3天)
- [附录 A：配置模板速查](#附录-a配置模板速查)
- [附录 B：关键命令速查](#附录-b关键命令速查)
- [附录 C：学习资源索引](#附录-c学习资源索引)

---

## 一、硬件资源评估与限制分析

### 1.1 硬件规格

| 项目 | 规格 |
|------|------|
| GPU | 8× RTX 4090 |
| VRAM | 24 GB GDDR6X / 卡，共 192 GB |
| 架构 | Ada Lovelace (SM89) |
| 互联 | PCIe Gen4 x16（~32 GB/s 双向） |
| NVLink | ❌ 不支持 |
| RDMA/InfiniBand | ❌ 不支持 |
| FP8 支持 | ✅ Ada 架原生支持 FP8 |
| CUDA | 12.x |

### 1.2 功能可用性矩阵

| 功能 | 可用？ | 说明 |
|------|--------|------|
| Aggregated Serving（单机聚合推理） | ✅ | 核心场景，所有后端均支持 |
| KV-Aware Routing（KV感知路由） | ✅ | ZMQ 传输，无需外部基础设施 |
| KV Cache Offload 到 CPU/SSD（KVBM） | ✅ | G2/G3 层级，PCIe D2H/H2D |
| 同 GPU 分离式服务（开发测试） | ✅ | `disagg_same_gpu.sh` 脚本 |
| 同节点分离式服务（PCIe） | ✅（有限） | UCX cuda_ipc，~32 GB/s / GPU 对 |
| 跨节点分离式服务 | ❌（不可行） | 无 RDMA，TCP 回退延迟过高 |
| WideEP（宽专家并行） | ❌ | 需要 NVLink + RDMA |
| 多节点 TP | ❌（不可行） | PCIe 跨节点太慢 |
| 投机解码 | ✅ | 单 GPU，16GB+ 即可 |
| LoRA 适配器 | ✅ | vLLM 后端 |
| 多模态推理（小 VLM） | ✅ | 受 VRAM 限制 |
| 扩散模型 | ✅ | SGLang 后端 |
| Planner 自动扩缩容（advisory） | ✅ | 纯软件 |
| Planner 自动扩缩容（K8s） | ⚠️ 部分 | 需要 K8s 集群 |
| 可观测性（Prometheus/Grafana） | ✅ | 纯软件 |
| 故障容错（取消/优雅关闭） | ✅ | 纯软件 |
| Mocker（模拟引擎） | ✅ | 无 GPU 需求，用于测试/基准 |

### 1.3 模型选择策略

**单 GPU（1× 4090, 24GB）**：

| 模型 | 量化 | 权重大小 | KV Cache 剩余 | 适用场景 |
|------|------|---------|--------------|---------|
| Qwen3-0.6B | FP16 | ~1.2 GB | ~20 GB | 快速验证、功能测试 |
| Qwen3-1.7B | FP16 | ~3.4 GB | ~18 GB | 日常开发 |
| Qwen3-4B | FP16 | ~8 GB | ~14 GB | Agent 测试 |
| Qwen3-8B | FP8 | ~8 GB | ~14 GB | 中等任务 |
| Llama-3.1-8B | FP8 | ~8 GB | ~14 GB | 通用任务 |

**多 GPU（2× 4090, 48GB, TP=2）**：

| 模型 | 量化 | 说明 |
|------|------|------|
| Qwen3-32B-FP8 | FP8 | 推荐，舒适运行 |
| Qwen3-30B-A3B (MoE) | FP16 | 仅 3B 活跃参数，轻松运行 |

**多 GPU（4× 4090, 96GB, TP=4）**：

| 模型 | 量化 | 说明 |
|------|------|------|
| Llama-3-70B | FP8 | 可行但 KV Cache 紧张 |
| Qwen3-32B-FP8 | FP8 | 可运行 2 个副本（2×TP=2） |

**多 GPU（8× 4090, 192GB, TP=8）**：

| 模型 | 量化 | 说明 |
|------|------|------|
| Llama-3-70B | FP8 | 舒适运行，有 KV Cache 空间 |

---

## 二、复现总览与阶段规划

### 2.1 时间线（共 6 周）

```
Week 1: 环境搭建 + 基础验证
Week 2: Dynamo 核心功能（聚合推理 + KV 路由 + 多后端）
Week 3: 分离式服务 + KV Cache 多级管理
Week 4: Agent 推理 + Tool Calling
Week 5: Dynamo + slime 联动（RL 后训练集成）
Week 6: 性能调优 + 总结复盘
```

### 2.2 每阶段目标与产出

| 阶段 | 目标 | 产出 |
|------|------|------|
| 1 | 环境跑通，Hello World | 可运行的 Dynamo 环境 |
| 2 | 聚合推理 + KV 路由 + 多后端对比 | 性能对比数据 |
| 3 | 分离式服务 + KVBM | P/D 分离效果数据 |
| 4 | Agent 推理 + Tool Calling | 多轮工具调用 demo |
| 5 | slime + Dynamo 联动 | RL 训练曲线 |
| 6 | 性能调优 + 文档总结 | 调优报告 |

---

## 三、阶段一：环境搭建与基础验证（2-3天）

### 3.1 安装 Dynamo

**方式 A：从 PyPI 安装（推荐，最快）**

```bash
# 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv dynamo-env && source dynamo-env/bin/activate

# 安装 Dynamo（SGLang 后端）
uv pip install --prerelease=allow "ai-dynamo[sglang]"

# 或者安装 vLLM 后端
# uv pip install --prerelease=allow "ai-dynamo[vllm]"
```

**方式 B：从源码构建（需要深入代码时）**

```bash
# 安装系统依赖（Ubuntu 24.04）
sudo apt install -y build-essential libhwloc-dev libudev-dev \
  pkg-config libclang-dev protobuf-compiler python3-dev cmake

# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 创建虚拟环境
uv venv dynamo-env && source dynamo-env/bin/activate
uv pip install pip 'maturin[patchelf]'

# 构建 Python 绑定
cd lib/bindings/python && maturin develop --uv && cd /data/home/yizhou/dynamo

# 安装 GPU 内存服务
uv pip install -e lib/gpu_memory_service

# 安装 Dynamo 主包
uv pip install -e .
```

**方式 C：Docker 容器（最简单）**

```bash
# 拉取预构建容器
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1

# 运行
docker run --gpus all --network host --rm -it \
  nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1
```

### 3.2 启动基础设施

```bash
# 方式 1：Docker Compose（推荐，启动 NATS + etcd）
cd /data/home/yizhou/dynamo/dev
docker compose up -d

# 方式 2：File-based discovery（零基础设施，推荐单机开发）
# 不需要启动任何服务，启动时加 --discovery-backend file
```

### 3.3 验证安装：Hello World

```bash
# 终端 1：启动 Frontend
python3 -m dynamo.frontend --http-port 8000 --discovery-backend file

# 终端 2：启动 SGLang Worker
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file

# 终端 3：发送请求
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello! What is Dynamo?"}],
    "max_tokens": 100
  }' | python3 -m json.tool
```

### 3.4 验证 Mocker（无 GPU 测试）

```bash
# Mocker 不需要真实 GPU，用于学习架构和测试路由
python3 -m dynamo.mocker --num-workers 4 --discovery-backend file
```

### 3.5 阶段一验收标准

- [ ] Dynamo 环境安装成功
- [ ] 能通过 Frontend 发送 OpenAI 兼容请求
- [ ] SGLang Worker 能正常响应
- [ ] Mocker 能模拟多 Worker 场景
- [ ] 理解 Dynamo 的三平面架构（Request/Control/Storage）

---

## 四、阶段二：Dynamo 核心功能复现（4-5天）

### 4.1 聚合推理（Aggregated Serving）

**单 GPU 基线**：

```bash
# SGLang 后端
python3 -m dynamo.frontend --discovery-backend file &
python3 -m dynamo.sglang --model-path Qwen/Qwen3-8B \
  --discovery-backend file --mem-fraction-static 0.9

# vLLM 后端
python3 -m dynamo.frontend --discovery-backend file &
python3 -m dynamo.vllm --model Qwen/Qwen3-8B \
  --discovery-backend file --gpu-memory-utilization 0.90 \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

**多 GPU TP（TP=2，2 卡 32B 模型）**：

```bash
python3 -m dynamo.frontend --discovery-backend file &
CUDA_VISIBLE_DEVICES=0,1 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-32B-FP8 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

### 4.2 KV-Aware Routing

**多 Worker + KV 路由**：

```bash
# 启动 Frontend + KV Router
python3 -m dynamo.frontend --router-mode kv --http-port 8000 &

# Worker 1（GPU 0）
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-8B --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# Worker 2（GPU 1）
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
  python3 -m dynamo.vllm \
  --model Qwen/Qwen3-8B --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &
```

**验证 KV 路由效果**：

```bash
# 发送有共享前缀的请求，观察 cache_hit_rate
curl -s localhost:8000/v1/chat/completions -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 1+1?"}
  ]
}'

# 再发一个共享 system prompt 的请求
curl -s localhost:8000/v1/chat/completions -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ]
}'
```

### 4.3 多后端对比

在相同硬件上对比 SGLang、vLLM、TRT-LLM 三个后端：

| 对比项 | SGLang | vLLM | TRT-LLM |
|--------|--------|------|---------|
| 启动速度 | | | |
| TTFT (P50/P99) | | | |
| ITL (P50/P99) | | | |
| 吞吐量 (tokens/s) | | | |
| 显存占用 | | | |
| Prefix Cache 命中率 | | | |

### 4.4 投机解码

```bash
# vLLM + Eagle3 投机解码（单 GPU 即可）
python3 -m dynamo.frontend --discovery-backend file &
python3 -m dynamo.vllm --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"model":"yuhuili/EAGLE-LLaMA3.1-Instruct-8B","num_speculative_tokens":5}' \
  --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

### 4.5 阶段二验收标准

- [ ] 能在单 GPU 上运行聚合推理（SGLang + vLLM）
- [ ] 能在 2 GPU 上运行 TP=2 的大模型
- [ ] 能配置 KV-Aware Routing 并观察缓存命中
- [ ] 能对比 SGLang 和 vLLM 的性能差异
- [ ] 能运行投机解码并测量加速比
- [ ] 理解 Dynamo Router 的代价函数原理

---

## 五、阶段三：分离式服务与 KV 管理（3-4天）

### 5.1 同 GPU 分离式服务（开发测试模式）

```bash
# SGLang 同 GPU 分离
cd /data/home/yizhou/dynamo/examples/backends/sglang
bash launch/disagg_same_gpu.sh --model Qwen/Qwen3-0.6B

# vLLM 同 GPU 分离
cd /data/home/yizhou/dynamo/examples/backends/vllm
bash launch/disagg_same_gpu.sh --model Qwen/Qwen3-0.6B
```

### 5.2 同节点分离式服务（PCIe，4P+4D）

```bash
# 设置 UCX 使用 PCIe 传输
export UCX_TLS=cuda_copy,cuda_ipc

# Frontend
python3 -m dynamo.frontend --router-mode kv --http-port 8000 &

# Prefill Workers（GPU 0-3）
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-32B-FP8 --tensor-parallel-size 4 \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# Decode Workers（GPU 4-7）
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
  python3 -m dynamo.vllm \
  --model Qwen/Qwen3-32B-FP8 --tensor-parallel-size 4 \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
```

### 5.3 KVBM 多级缓存

```bash
# 配置 KVBM CPU/SSD 缓存
export DYN_KVBM_CPU_CACHE_GB=50    # 50GB CPU 缓存
export DYN_KVBM_DISK_CACHE_GB=200   # 200GB SSD 缓存

# 启动带 KVBM 的 vLLM Worker
python3 -m dynamo.vllm --model Qwen/Qwen3-8B \
  --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}' \
  --discovery-backend file
```

### 5.4 Planner（Advisory 模式）

```bash
# Advisory 模式：只计算建议，不执行扩缩容
python3 -m dynamo.planner \
  --mode agg \
  --backend vllm \
  --optimization-target throughput \
  --advisory true \
  --min-endpoint 1 \
  --max-gpu-budget 8
```

### 5.5 阶段三验收标准

- [ ] 能运行同 GPU 分离式服务（disagg_same_gpu）
- [ ] 能运行同节点分离式服务（4P+4D over PCIe）
- [ ] 理解 NIXL/UCX 在 PCIe 上的传输机制
- [ ] 能配置 KVBM 的 CPU/SSD 缓存
- [ ] 理解分离式服务的适用场景和限制
- [ ] 能用 Planner 的 advisory 模式分析资源使用

---

## 六、阶段四：Agent 推理与 Tool Calling（3-4天）

### 6.1 Tool Calling 验证

```bash
# 启动带 Tool Calling 支持的服务
python3 -m dynamo.frontend \
  --chat-processor vllm \
  --dyn-tool-call-parser hermes \
  --discovery-backend file &

python3 -m dynamo.vllm --model Qwen/Qwen3-8B \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

**测试工具调用**：

```bash
curl -s localhost:8000/v1/chat/completions -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
      }
    }
  }]
}'
```

### 6.2 Reasoning Parser 验证

```bash
# 启动带推理解析的服务
python3 -m dynamo.frontend \
  --chat-processor vllm \
  --dyn-reasoning-parser deepseek_r1 \
  --discovery-backend file &

python3 -m dynamo.vllm --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

### 6.3 Session Routing（会话路由）

```bash
# 配置会话亲和性
python3 -m dynamo.frontend --router-mode kv \
  --router-session-affinity-ttl-secs 3600 \
  --discovery-backend file &

# 带 Session ID 的请求
curl -s localhost:8000/v1/chat/completions \
  -H "X-Dynamo-Session-ID: agent-session-001" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Help me write a Python function"}]
  }'
```

### 6.4 多模态推理

```bash
# SGLang 多模态推理
python3 -m dynamo.frontend --discovery-backend file &
python3 -m dynamo.sglang --model-path Qwen/Qwen3-VL-8B \
  --discovery-backend file --mem-fraction-static 0.8
```

### 6.5 阶段四验收标准

- [ ] 能运行 Tool Calling 并解析工具调用结果
- [ ] 能运行 Reasoning Parser 并提取推理过程
- [ ] 理解 Session Routing 对 Agent 场景的价值
- [ ] 能运行多模态推理（图文输入）
- [ ] 理解 20+ 种 tool call parser 的设计差异

---

## 七、阶段五：Dynamo + slime 联动（3-4天）

### 7.1 思路：用 SGLang 做 rollout，Dynamo 做推理服务

slime 的 rollout 引擎是 SGLang。Dynamo 可以作为 SGLang 之上的编排层，提供：
- KV-aware routing 提升 rollout 效率
- 多 Worker 负载均衡
- 会话路由保持 prefix cache

### 7.2 配置 slime 使用 Dynamo Frontend

```bash
# 启动 Dynamo Frontend + SGLang Workers
python3 -m dynamo.frontend --router-mode kv --http-port 8000 &
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B --discovery-backend file &

# slime 配置使用 Dynamo 的 endpoint
# 在 slime 的训练脚本中：
# --sglang-url http://localhost:8000
```

### 7.3 用 Dynamo Mocker 做 RL 训练模拟

```bash
# Mocker 不需要真实 GPU，可以模拟推理延迟
# 用于学习 RL 训练流程而不需要真实推理
python3 -m dynamo.mocker --num-workers 4 --discovery-backend file
```

### 7.4 RL 训练中的推理资源管理

**问题**：RL 训练时，rollout 和 training 共享 GPU 资源。

**方案 A：Colocate（共享 GPU）**
```bash
# slime colocate 模式 + Dynamo Frontend
--colocate
--sglang-mem-fraction-static 0.6
```

**方案 B：Separated（分离 GPU）**
```bash
# 4 GPU 用于 rollout（Dynamo 管理），4 GPU 用于 training（slime 管理）
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B --discovery-backend file &
# slime training 使用 GPU 4-7
```

### 7.5 阶段五验收标准

- [ ] 理解 slime 如何使用 SGLang 做 rollout
- [ ] 能配置 Dynamo Frontend 作为 rollout 的入口
- [ ] 理解 RL 训练中的资源分配策略
- [ ] 能用 Mocker 模拟推理负载进行 RL 训练测试

---

## 八、阶段六：性能调优与总结（2-3天）

### 8.1 性能调优 Checklist

| 优化项 | 配置 | 预期效果 |
|--------|------|---------|
| Prefix Cache | `--router-mode kv` + LPM 调度 | TTFT 降低 2-5x |
| FP8 KV Cache | `--kv-cache-dtype fp8` | KV 容量翻倍 |
| CUDA Graph | `--cuda-graph-max-bs 24`（4090 自适应） | 延迟降低 10-20% |
| Chunked Prefill | `--chunked-prefill-size 2048` | 避免 decode 饥饿 |
| jemalloc | `LD_PRELOAD=libjemalloc.so.2` | Frontend 内存优化 |
| GPU 显存最大化 | `--mem-fraction-static 0.92` | 更多 KV Cache 空间 |

### 8.2 监控搭建

```bash
# 启动 Prometheus + Grafana
cd /data/home/yizhou/dynamo/dev
docker compose -f docker-observability.yml up -d

# 访问 Grafana
# http://localhost:3000 (admin/admin)
```

### 8.3 基准测试

```bash
# 使用 AIPerf 进行基准测试
# 安装
pip install aiperf

# 运行测试
aiperf serve \
  --model Qwen/Qwen3-8B \
  --url http://localhost:8000/v1/chat/completions \
  --concurrency 1,4,8,16,32 \
  --duration 60
```

### 8.4 阶段六验收标准

- [ ] 完成性能调优并记录对比数据
- [ ] 搭建 Prometheus + Grafana 监控
- [ ] 完成基准测试并分析结果
- [ ] 撰写复现总结文档

---

## 附录 A：配置模板速查

### A.1 最小配置（单 GPU，Qwen3-0.6B）

```bash
python3 -m dynamo.frontend --discovery-backend file &
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file
```

### A.2 标准配置（2 GPU，KV 路由）

```bash
python3 -m dynamo.frontend --router-mode kv --http-port 8000 &
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model Qwen/Qwen3-8B \
  --enforce-eager --gpu-memory-utilization 0.90 \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
  python3 -m dynamo.vllm --model Qwen/Qwen3-8B \
  --enforce-eager --gpu-memory-utilization 0.90 \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &
```

### A.3 分离式配置（4P+4D，同节点 PCIe）

```bash
export UCX_TLS=cuda_copy,cuda_ipc
python3 -m dynamo.frontend --router-mode kv --http-port 8000 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-32B-FP8 --tensor-parallel-size 4 \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
  python3 -m dynamo.vllm \
  --model Qwen/Qwen3-32B-FP8 --tensor-parallel-size 4 \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
```

### A.4 TP=8 大模型配置（Llama-3-70B FP8）

```bash
python3 -m dynamo.frontend --discovery-backend file &
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m dynamo.vllm \
  --model meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

---

## 附录 B：关键命令速查

### B.1 服务管理

```bash
# 启动基础设施
cd /data/home/yizhou/dynamo/dev && docker compose up -d

# 启动 Frontend
python3 -m dynamo.frontend --http-port 8000 --discovery-backend file

# 启动 Worker（SGLang）
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file

# 启动 Worker（vLLM）
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'

# 启动 Mocker
python3 -m dynamo.mocker --num-workers 4 --discovery-backend file

# 启动 Planner（advisory）
python3 -m dynamo.planner --mode agg --advisory true
```

### B.2 测试与调试

```bash
# 发送测试请求
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# 查看 Prometheus 指标
curl localhost:8000/metrics

# 查看 GPU 状态
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1

# 查看日志
export DYN_LOG=debug
```

### B.3 运行测试

```bash
# 单元测试（不需要 GPU）
pytest tests/basic/ -m "gpu_0"

# Frontend 测试
pytest tests/frontend/ -m "gpu_0"

# Router 测试（用 Mocker）
pytest tests/router/test_router_e2e_with_mockers.py

# SGLang 集成测试（1 GPU）
pytest tests/serve/test_sglang.py -m "gpu_1"

# vLLM 集成测试（1 GPU）
pytest tests/serve/test_vllm.py -m "gpu_1"
```

---

## 附录 C：学习资源索引

### C.1 Dynamo 文档

| 文档 | 路径 | 内容 |
|------|------|------|
| 整体架构 | `docs/design-docs/overall-architecture.md` | 三平面架构 |
| 分离式服务 | `docs/design-docs/disagg-serving.md` | P/D 分离设计 |
| Router 概念 | `docs/components/router/router-concepts.md` | KV 路由算法 |
| Router 配置 | `docs/components/router/router-configuration.md` | 参数调优 |
| KVBM 设计 | `docs/components/kvbm/` | 多级缓存 |
| Planner | `docs/components/planner/planner-guide.md` | 自动扩缩容 |
| 故障容错 | `docs/fault-tolerance/` | 迁移/取消/优雅关闭 |
| 性能调优 | `docs/performance/tuning.md` | 生产调优指南 |
| Tool Calling | `docs/tool-calling/README.md` | 工具调用支持 |
| Agent 支持 | `docs/agents/` | 会话/提示/追踪 |

### C.2 代码阅读顺序

**入门级**：
1. `components/src/dynamo/frontend/main.py` — Frontend 入口
2. `components/src/dynamo/sglang/` — SGLang 后端集成
3. `components/src/dynamo/vllm/` — vLLM 后端集成

**进阶级**：
4. `lib/kv-router/src/scheduling/selector.rs` — KV 路由代价函数
5. `lib/kv-router/src/indexer/radix_tree.rs` — RadixTree 实现
6. `lib/llm/src/migration.rs` — 请求迁移

**高级**：
7. `lib/kvbm-engine/` — KV Block Manager
8. `lib/runtime/` — 分布式运行时
9. `components/src/dynamo/planner/` — 自动扩缩容

### C.3 示例代码

| 示例 | 路径 | 用途 |
|------|------|------|
| vLLM 聚合 | `examples/backends/vllm/launch/agg.sh` | 最简启动 |
| vLLM + Router | `examples/backends/vllm/launch/agg_router.sh` | KV 路由 |
| vLLM 分离 | `examples/backends/vllm/launch/disagg.sh` | P/D 分离 |
| vLLM 同 GPU 分离 | `examples/backends/vllm/launch/disagg_same_gpu.sh` | 开发测试 |
| vLLM KVBM | `examples/backends/vllm/launch/agg_kvbm.sh` | KV 缓存卸载 |
| SGLang 聚合 | `examples/backends/sglang/launch/agg.sh` | SGLang 后端 |
| SGLang + Router | `examples/backends/sglang/launch/agg_router.sh` | SGLang + KV 路由 |
| SGLang 同 GPU 分离 | `examples/backends/sglang/launch/disagg_same_gpu.sh` | SGLang P/D 测试 |
| Hello World | `examples/custom_backend/hello_world/` | 无 GPU，学概念 |
| Mocker | `components/src/dynamo/mocker/` | 模拟引擎 |
