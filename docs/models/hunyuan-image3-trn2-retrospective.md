# HunyuanImage-3.0 在 AWS Trainium2 上的推理适配与优化：完整复盘

> 写给新入职同事。读完你应该知道三件事：这个模型怎么在 Trainium2 上跑、现在的默认方案为什么长这样、以及我们一路上踩过哪些坑。
> 第一次出现的专业词会就地解释，第 2 节还有一张集中的术语表。
>
> 时间跨度：2026-10-09 到 2026-10-13（Lite/NKI 路线），前面还有 9 月底的 NxDI 路线作为前史。

---

## 0. 当前默认方案

### 0.1 一句话

一台 **trn2.48xlarge**，用 33 个逻辑核跑完整的 80B MoE 模型 HunyuanImage-3.0-Instruct：
- **DiT 主干**用 32 个核：两组 TP16，组内张量/专家并行。开 CFG 时两组各跑一个 CFG 分支（CFG 并行），不开 CFG 时两组各分一半图像 token（上下文并行）。
- **attention** 走 NKI flash kernel。
- **UNet 头尾**在 32 个核上按 GroupNorm 组切开。
- **跳步**用 cache-dit（阈值 0.12）。
- **VAE** 用 fp16，单独占第 33 个核。
- **think（先写一段思考文字）**复用同一套权重和核。

### 0.2 核的分配

```
trn2.48xlarge：16 颗 Trainium2 芯片，按 LNC=2 共 64 个逻辑核，每核 24 GB HBM

逻辑核  0 ─ 15  │ TP 块 A（芯片 0–3，片间环网的一行）     ← CFG 时跑 cond 分支 / 无 CFG 时算前一半 token
逻辑核 16 ─ 31  │ TP 块 B（芯片 4–7，环网的下一行）       ← CFG 时跑 uncond 分支 / 无 CFG 时算后一半 token
                │  rank r 和 r+16 组成一个 "CP 对"，权重完全相同
逻辑核 32       │ VAE 解码（fp16，独立子进程）
逻辑核 33 ─ 63  │ 空闲
```

### 0.3 每个部件怎么跑

| 部件 | 默认实现 | 关键配置（stage yaml 的 `model_config`） |
| --- | --- | --- |
| Transformer 主干（32 层） | TP16 × CP2：每组 16 个 rank，每 rank 有 2 个 query 头、1 个 KV 头、64 个专家中的 4 个，权重约 10 GB | `context_parallel: 2` |
| CFG | 两个 TP16 块各跑一个分支，batch 1 | `cfg_parallel: true` |
| denoise attention | nkilib `attention_cte` flash kernel，prompt K/V 作为 prefix | `denoise_attention: nki` |
| MoE | nkilib `moe_cte`，`shard_on_i_hybrid` 动态循环，block 512 | `moe_implementation`、`moe_block_size: 512` |
| UNet patch embed / final layer | 按 32 个 GroupNorm 组切到 32 个 rank 上 | `unet_parallel: true` |
| 跳步缓存 | cache-dit DBCache，阈值 0.12（50 步里约跳 25 步） | `step_cache: {residual_diff_threshold: 0.12}` |
| VAE 解码 | 在设备上分 7 段解码，fp16、fp32 归约，独占核 32 | `vae_device: neuron`、`vae_device_dtype: float16`、`vae_device_core: 32` |
| AR（think / recaption） | 同一套权重；CP 对里的两个 rank 各算一半再全局求和 | `bot_task: think`（只在 AR yaml 里） |

### 0.4 性能（1024×1024，50 步，seed 42，热请求）

| 场景 | 发版时（v0.24.0.0.2.0，TP32） | **现在** |
| --- | ---: | ---: |
| 默认：只出图（DiT），CFG 2.5，cache-dit 0.12 | 17.15 s | **12.6 s** |
| 只出图，CFG 2.5，不跳步 | 29.89 s（0.558 s/步） | **19.8 s（0.357 s/步）** |
| 和 GPU 同口径：think，不开 CFG，不跳步 | 27.48 s | **21.1 s** |
| 　其中 AR（think 文字） | 8.5 s（15.5 ms/token） | 8.7 s（15.7 ms/token） |
| 　其中 DiT（denoise + VAE） | 18.8 s | **12.3 s** |

同口径的 GPU 参考（vLLM-Omni，DiT 各用 4 张卡，不开 CFG，不跳步）：
- g7e（RTX PRO 6000）：DiT 13.68 s，整次请求约 42 s。
- H100：DiT 6.03 s，整次请求约 42 s。

trn2 的 DiT 现在比 g7e 快，比 H100 慢 2 倍；整次请求比两者都快，主要靠 AR。
这个对比有前提，见第 6 节。

**精度**：用 teacher forcing（见术语表）逐步对比，CFG 合成预测相对 CPU fp32 参考的误差是 5.27e-2，在 bf16 的正常范围内（5.2e-2 到 5.6e-2）。

### 0.5 代码在哪

- 仓库：GitHub `AutoJunjie/vllm-omni-neuron`，分支 `feat/hunyuan-image3-neuron-lite`。
- 已发布：tag `v0.24.0.0.2.0`（TP32 + UNet 切分 + cache-dit）。
- 发版之后的提交在分支 `feat/hunyuan-image3-neuron-lite` 上，**还没合进 main、没发版**：
  - `931e5e8`：通信基准
  - `a729bd4`：TP16×CP2
  - `8efdeef`：CFG 并行
  - `e64ef74`：flash attention
- 使用说明和配置参考见同目录的 [`hunyuan-image3-instruct.md`](hunyuan-image3-instruct.md)。
- 模型代码在 `vllm_omni_neuron/diffusion/models/hunyuan_image3/`：

| 文件 | 内容 |
| --- | --- |
| `hunyuan_image3_transformer.py` | 主干：attention、MoE、分组、kernel 封装 |
| `pipeline_hunyuan_image3.py` | 请求流程、denoise 循环、跳步缓存、CFG 并行 |
| `hunyuan_image3_ar.py` | think / recaption 的文字生成 |
| `unet_parallel.py` | UNet 头尾切分 |
| `vae_device.py`、`vae_nki_conv.py`、`vae_worker.py` | 设备上的 VAE |

---

## 1. 先认识这个模型

### 1.1 HunyuanImage-3.0 是什么

腾讯的文生图模型（输入文字，输出图片），特别之处在于它是**统一模型**：一个 80B 参数的 MoE 语言模型主干，同时承担两件事。
- **写文字**（AR，自回归）：比如 think 模式下，先就你的 prompt 写一段几百字的"构图思考"。
- **画图**（DiT，扩散 Transformer）：把一张随机噪声图逐步"去噪"成真实图片。

常见的文生图模型（比如 Stable Diffusion）是"文本编码器 + 独立的去噪网络"两部分；HunyuanImage-3.0 的文字和图像 token 走的是同一套 32 层 Transformer。

主干的关键尺寸：

| 项 | 值 |
| --- | --- |
| 层数 | 32 |
| hidden size | 4096 |
| attention 头 | 32 个 query 头，8 个 KV 头（GQA），head_dim 128 |
| MoE | 每层 64 个专家，每个 token 选 8 个（top-8），外加 1 个共享专家 |
| 词表 | 约 13.3 万 |
| 权重 | bf16 约 157 GB |

### 1.2 一次请求经过哪些步骤

```
用户 prompt
   │
   ├─(可选) AR think：先在主干上逐 token 生成约 540 个 token 的思考文字（贪心解码）
   │
   ▼
prompt prefill：把 prompt（+思考文字）过一遍 32 层，存下每层的 K/V（prompt 的"记忆"）
   │
   ▼
50 个 denoise step，每一步：
   latent（64×64×32 的噪声图）
     → patch embed（UNetDown）：变成 4096 个图像 token，再加 1 个 timestep token = 4097 个 token
     → 32 层 Transformer（每层：attention 看 prompt 的 K/V 和图像自身，再过 MoE）
     → final layer（UNetUp）：token 变回 latent 形状的"噪声预测"
     → CFG 合成（有条件/无条件两个预测做加权差）
     → scheduler 按预测把 latent 往干净图推一步
   │
   ▼
VAE decode：64×64 latent → 1024×1024 像素
   │
   ▼
PNG
```

时间主要花在 50 个 denoise step 上，所以绝大部分优化都围绕"每步多少秒"展开。

### 1.3 我们的软件栈

```
vLLM-Omni 0.24（上游的多模态推理框架，负责服务、调度、OpenAI 兼容接口）
   └─ vllm-omni-neuron（我们的插件：把 HunyuanImage3 的 pipeline 换成 Neuron 版）
        └─ vllm-neuron（Neuron 的 vLLM 平台层）
             └─ PyTorch + torch.compile
                  └─ libtorch-neuronx-lite（"Lite"，Neuron 的原生 PyTorch 后端）
                       └─ neuronx-cc（编译器，把计算图编成 NEFF）+ NKI kernel（手写算子）
                            └─ Neuron Runtime → Trainium2 芯片
```

---

## 2. 术语表

### 2.1 硬件

| 术语 | 解释 |
| --- | --- |
| **Trainium2（trn2）** | AWS 自研的机器学习芯片。trn2.48xlarge 是一台装了 16 颗 Trainium2 的实例 |
| **NeuronCore** | 芯片里的计算核。每颗 Trainium2 有 8 个物理 NeuronCore |
| **LNC（Logical NeuronCore）** | 把几个物理核合并成一个"逻辑核"来用。我们用 LNC=2：2 个物理核算 1 个逻辑核，所以每颗芯片 4 个逻辑核，整机 64 个。下文说"核"都指逻辑核 |
| **HBM** | 芯片上的高带宽显存。每颗芯片 96 GB，分到每个逻辑核 24 GB |
| **SBUF / PSUM** | 核内的片上小缓存（SRAM）。计算要先把数据从 HBM 搬进 SBUF，矩阵乘的累加结果在 PSUM。HBM 和 SBUF 之间的搬运就是 DMA |
| **DMA** | 直接内存搬运。数据搬得太碎（大量小块 DMA）会非常慢，这是我们遇到过的瓶颈 |
| **Tensor / Vector / Scalar / GpSimd 引擎** | 核里分工不同的计算单元：Tensor 做矩阵乘，Vector 和 Scalar 做逐元素运算，GpSimd 做通用计算 |
| **片间环网（torus）** | 16 颗芯片用 4×4 的二维环形网络互连。集合通信的分组要贴合这个拓扑，否则走不通或很慢 |
| **MFU** | 实际算力 ÷ 峰值算力。我们 profile 时只有 6–7%，说明大部分时间不在做矩阵乘 |
| **g7e / p5（H100）** | 用来对比的 GPU 实例。g7e 是 8 张 RTX PRO 6000（卡间走 PCIe），p5 是 8 张 H100（卡间走 NVLink） |

### 2.2 编译与运行时

| 术语 | 解释 |
| --- | --- |
| **torch.compile / Dynamo** | PyTorch 把 Python 代码"追踪"成计算图再交给后端编译。我们用 `fullgraph=True, dynamic=False`：整段函数必须编成一张图，而且形状固定 |
| **静态形状** | 编出来的图只认一种输入形状。形状变了（比如换分辨率、换 batch）就要重新编一张图 |
| **neuronx-cc** | Neuron 的编译器，把计算图编成 NEFF。一张大图冷编译要几分钟到几十分钟 |
| **NEFF** | Neuron 可执行文件，相当于编好的程序。每个 rank 的 NEFF 都不同 |
| **NEFF 缓存 / 冷编译** | 编过的 NEFF 存在磁盘上，下次命中就直接加载；没命中就是"冷编译"。**坑：缓存 key 里包含被追踪代码的源文件路径和行号**，在被追踪函数的上方加一行代码都会导致重编 |
| **Lite（libtorch-neuronx-lite）** | Neuron 的原生 PyTorch 后端，我们用的 `neuron_native` 路线。它和旧的 torch-neuronx、NxDI 不是一套东西 |
| **NKI（Neuron Kernel Interface）** | 用 Python 写 Neuron 底层 kernel 的接口，相当于 GPU 上的 CUDA/Triton |
| **nkilib** | 官方提供的 NKI kernel 库。我们用了其中的 `attention_cte`（flash attention）、`moe_cte`（MoE）、`output_projection_cte`（attention 输出投影）、`conv3d`（VAE 卷积） |
| **ICE** | Internal Compiler Error，编译器内部错误，比如 `NCC_IMPR902`、`NCC_IBTN020` |
| **spill** | 片上缓存放不下的中间结果被写回 HBM，再读回来，很耗时 |
| **profile / NeuronProfiler / neuron-explorer** | 抓一段执行的时间线：每个引擎在做什么、每次通信花多久。是找瓶颈的根本手段 |

### 2.3 并行与通信

| 术语 | 解释 |
| --- | --- |
| **rank** | 并行里的一个进程，绑定一个核。我们的 DiT 有 32 个 rank |
| **TP（张量并行）** | 把每层的权重矩阵切成几份，每个 rank 算一部分，最后求和。TP32 就是切 32 份 |
| **EP（专家并行）** | MoE 的 64 个专家分给不同 rank 持有。TP32 时每 rank 持有 2 个专家 |
| **CP（上下文并行）/ SP（序列并行）** | 把 token 序列切给不同 rank，每个 rank 只算自己那段 token。attention 需要看所有 token 的 K/V，所以要交换 K/V |
| **CFG 并行** | CFG 要算"有条件"和"无条件"两个分支。CFG 并行就是让两组 rank 各算一个分支 |
| **集合通信（collective）** | 多个 rank 一起参与的通信，比如 all-reduce、all-gather |
| **all-reduce** | 每个 rank 都有一份同样形状的张量，大家求和后人人拿到结果。TP 每层要做 2 次（attention 后、MoE 后） |
| **all-gather** | 每个 rank 贡献一块，拼起来后人人拿到完整的一份 |
| **replica groups** | 编译器需要知道"这次集合通信的分组是哪些 rank"。新建的进程组要在 Lite 的 mesh 注册表里登记分组，否则编译时报 "replica id not seen" |
| **algbw / busbw** | 通信带宽的两种算法：algbw = 数据量 ÷ 时间；busbw 按 all-reduce 的实际传输量折算，可以和链路带宽直接比 |

### 2.4 模型结构

| 术语 | 解释 |
| --- | --- |
| **Transformer 层** | RMSNorm → attention → 残差相加 → RMSNorm → MLP/MoE → 残差相加 |
| **attention / Q / K / V** | 每个 token 生成 query（我要找什么）、key（我是什么）、value（我的内容）。用 Q 和所有 K 的相似度做 softmax，再加权 V |
| **GQA（分组查询注意力）** | 多个 query 头共享一个 KV 头（这里是 32 个 query 头共享 8 个 KV 头），省显存。TP16 时每 rank 2 个 query 头共享 1 个 KV 头 |
| **RoPE** | 旋转位置编码，把 token 的位置信息"旋转"进 Q 和 K。这里是图像用的 2D 版本 |
| **RMSNorm / QK norm** | 一种归一化。QK norm 是对每个头的 Q 和 K 再各做一次 RMSNorm |
| **MoE（混合专家）** | MLP 换成 64 个"专家"小 MLP，由 **router**（路由器，一个小线性层 + softmax）给每个 token 选 top-8 个专家，按权重相加。参数多，但每个 token 只算一小部分 |
| **共享专家** | 每个 token 都会经过的那个专家 |
| **flash attention** | 分块计算 attention，不把完整的 `[query 数 × key 数]` 分数矩阵写进显存，省带宽和显存 |
| **prefix caching** | 把一段"前缀"的 K/V 当作已知前文，attention 时直接拼在前面用。我们用它来放 prompt 的 K/V |
| **prefill / decode** | prefill 是把整段 prompt 一次性过一遍、存下 K/V；decode 是之后每次只喂 1 个新 token |
| **KV cache** | 存下来的历史 K/V，decode 时不用重复计算 |
| **AR（自回归）** | 一个 token 一个 token 地生成文字。**贪心解码**：每步取概率最大的 token |
| **LM head** | 把最后一层的 hidden 映射到词表上的那个大矩阵 |

### 2.5 扩散模型

| 术语 | 解释 |
| --- | --- |
| **扩散 / denoise** | 从纯噪声出发，每一步让模型预测"噪声长什么样"，去掉一点，50 步后得到干净的图 |
| **DiT** | Diffusion Transformer，用 Transformer 做去噪网络 |
| **latent** | VAE 压缩后的"潜空间图"。这里 1024×1024 的图对应 32 通道 × 64×64 的 latent |
| **VAE** | 变分自编码器。decode 把 latent 还原成像素图，只在最后跑一次 |
| **timestep / scheduler** | timestep 表示当前在去噪的哪个阶段；scheduler（这里用 FlowMatchEuler）根据模型预测把 latent 往前推一步 |
| **CFG（Classifier-Free Guidance）** | 同时算"看 prompt"（cond）和"不看 prompt"（uncond）两个预测，用 `uncond + 2.5 × (cond − uncond)` 放大 prompt 的影响。代价是每步算两遍。`guidance_scale ≤ 1` 就是不开 CFG |
| **patch embed / final layer（UNetDown / UNetUp）** | DiT 的"入口"和"出口"：把 latent 变成 token、把 token 变回 latent。HunyuanImage3 用的是小型 UNet 结构（ResBlock + 卷积） |
| **ResBlock / GroupNorm** | 卷积残差块；GroupNorm 把通道分成 32 组，每组单独做归一化 |
| **step cache（跳步缓存）** | 相邻步的计算结果很像，有些步可以复用上一步的结果，不全算 |
| **cache-dit DBCache** | 一种跳步方法：每步先算第 1 层，比较它的残差和上次完整计算时的差异，差异小就跳过后 31 层，复用缓存的残差 |
| **TeaCache** | 另一种跳步方法：只看 timestep embedding 的变化来决定整步跳不跳，不看图像内容。vLLM-Omni 给这个模型默认配的是 TeaCache 0.2 |

### 2.6 精度与评估

| 术语 | 解释 |
| --- | --- |
| **bf16 / fp16 / fp32** | 三种浮点格式。bf16 范围大但精度低；fp16 精度更高但容易溢出；fp32 最准但慢、占内存 |
| **归约（reduction）** | 求和、求均值这类"把很多数合成一个"的运算。低精度下做大规模求和误差很大，所以要用 fp32 累加 |
| **相对 L2 误差** | ‖实际 − 参考‖ ÷ ‖参考‖ |
| **cosine** | 两个向量方向的相似度，1 表示完全同向 |
| **PSNR** | 两张图的像素相似度，单位 dB，越高越像。这个项目里，同一 seed 换一种实现，50 步后通常差约 35 dB，画面看不出区别；低于约 28 dB 能看出变化 |
| **teacher forcing（逐步强制对齐）** | 比较两个实现时，每一步都喂给它们完全相同的输入 latent，只比这一步的输出。否则 50 步误差会滚雪球，没法判断是谁的问题 |
| **bf16 噪声底** | 不同但都正确的 bf16 实现，相对 fp32 参考的误差都在 5.2e-2 到 5.6e-2。新实现落在这个范围就算合格 |
| **gold / 参考实现** | 在 CPU 上用 fp32、不切分、不用 kernel 跑出来的"标准答案"，一步约 20 分钟、要 320 GB 内存 |

### 2.7 运维

| 术语 | 解释 |
| --- | --- |
| **HyperPod** | AWS SageMaker 的集群服务，我们的 trn2 节点在里面 |
| **NodeRecovery** | HyperPod 的自动换节点开关。调试时必须设成 `None`，否则一次设备卡死就会把整台节点换掉，NVMe 上的东西全没 |
| **SSM** | AWS Systems Manager，我们通过它在节点上执行命令（没有 SSH） |
| **presigned URL（预签名 URL）** | 带临时签名的 S3 链接。节点本身没有 S3 写权限，所以本地生成 PUT 链接给节点用 |
| **systemd-run / nerdctl exec -d** | 在节点上启动后台长任务的方法。用 `setsid nohup` 起的任务会随 SSM 会话结束被杀掉 |

---

## 3. 时间线复盘

每个阶段都按"问题 → 做法 → 结果 → 教训"来写。耗时没特别说明的，都是 1024×1024、50 步、默认 CFG 2.5、只出图（DiT）的热请求。

### 阶段 0（9 月）：NxDI 路线的前史

最早用的是另一套栈（NxDI + torch-neuronx），服务能跑通，但 AR 的精度一直没能对齐到 0.03 的门限。
拿 H100 跑出 gold logits 对比后发现，连 H100 自己不同的 attention 实现之间都有 8% 的差异，单个指标没法判断对错。
10 月初决定换到原生 PyTorch + torch.compile + Lite + NKI 的新路线，从头做。

### 阶段 1（10-09）：在 Lite + NKI 上跑通，72.8 s

- **做法**：
  - 按 TP32 切权重：每 rank 1 个 query 头、2 个专家。
  - prompt 的 prefill 和每个 denoise step 各编成一张静态图。
  - prefill 的 attention 和输出投影用 nkilib kernel，MoE 先用普通矩阵乘。
- **结果**：
  - OpenAI 接口能出图，0.895 s/步，热请求 72.8 s，其中 VAE 在 CPU 上跑要 27.8 s。
  - 同一 seed 的图在离线和 API、冷编译和热缓存、两台不同机器之间**逐字节一致**。
  - 上游的 benchmark 4/4 通过。
- **坑**：
  1. 镜像预装的 Lite 版本缺子包，要升级。
  2. `NEURON_RT_VISIBLE_CORES` 在多进程下要由框架设置，不能手动设。
  3. Lite 的 host↔设备拷贝要求 dtype 一致，`.to(device=..., dtype=...)` 会报错，要先转类型再拷贝。
  4. 一次设备异常配上 `NodeRecovery=Automatic`，节点被整台替换了好几次，NVMe 上的东西全丢。之后的规矩：调试前一律设成 `None`。

### 阶段 2（10-09 至 10-10）：MoE 换 NKI kernel，先是编译器崩，再是不够快

- **问题**：`moe_cte` 单独测试没问题，放进 32 层的整图就报编译器内部错误 `NCC_IMPR902`。
- **定位**：一次排除一个假设：kernel 实例数、token 数、router、通信组、和 attention kernel 的冲突，全部排除。
  最后发现是**在图里用 `zeros + 切片赋值` 构造 mask**，改成纯比较 `arange(n) < live` 就编过了。
- **结果**：一开始比普通矩阵乘还慢（1.115 对 0.895 s/步）。调到 block 1024 时持平。
- **教训**：
  - mask 要用比较表达式生成，不要在图里往新张量里写值。
  - 稀疏化（只算选中的专家）省了 8 倍 FLOP，却被 block 的记账开销全部吃掉了。

### 阶段 3（10-10 至 10-11）：VAE 搬上设备，29.6 → 2.5 s，请求 47.0 s

- **问题**：CPU 上的 VAE 解码要 29 s，占请求的 36%。
- **四个必要条件**（每一个都是推翻前一个答案得来的）：
  1. **按深度切成 7 段分别编译，不按面积切**。整图指令数超过 1000 万的上限，切小图块反而指令更多。
  2. **3×3×3 卷积走 NKI 的 conv3d**。那个超宽的 1024→8192 卷积要按输出通道再切 4 份。
  3. **GroupNorm 的统计量必须用 fp32 累加**。纯 bf16 出来的图是垃圾，误差 100%。原因不是尾数位数不够，而是每组约 1680 万个元素的低精度求和，加上 `E[x²]−E[x]²` 的相减抵消。
  4. **fp32 放不进 24 GB**，所以用 16 位。
- **再改进**：
  - **fp16 比 bf16 准 7 倍、速度一样**，相对 CPU 的 PSNR 66 对 59 dB。fp16 要注意平方溢出，所以先按 2 的幂缩放再算。
  - 和主干放在同一个核上，**第二个请求会 OOM**。最后把 VAE 放到独立子进程，单独占核 32。
- **结果**：VAE 1.84 s，请求 47.0 s。
- **意外发现**：**NEFF 缓存 key 里有源码路径和行号**。把函数挪个文件、换个安装目录、在上面加一行，都会触发冷编译。VAE 冷编译一次要约 2 小时，所以编译前要先冻结代码。

### 阶段 4（10-11）：think 模式复用同一套主干

- **问题**：上游 vLLM-Omni 是两套模型、两组 GPU，一组写文字、一组画图，中间用连接器传结果。我们没有第二组卡。
- **做法**：在 DiT 自己的 32 个核和权重上加一张"单 token decode"图。
  - 生成的 KV cache 用 one-hot 乘加更新，不在图里切片写入（切片写入会触发阶段 2 那种编译器 bug）。
  - LM head 按词表切 32 份，一次 all-reduce 后每个 rank 拿到完整 logits，各自取同一个贪心 token。
- **结果**：**15.5 ms/token**，而 GPU 上的 vLLM-Omni 是 56–65 ms/token。think 请求 60.6 s（当时还开着 CFG）。

### 阶段 5（10-12）：MoE 换动态循环，0.90 → 0.76 s/步，请求 39.9 s

- **方法**：写了 `bench_denoise_layer.py`，在单核上按真实形状把一层拆开逐块计时：MoE 占 10.5 ms/层，是大头。
- **发现**：慢的不是记账，而是 kernel 的循环方式。`shard_on_block` 和 `shard_on_i` 会遍历每个 block，哪怕里面没有 token。
  `shard_on_i_hybrid` 配合 `num_static_block=0` 用硬件动态循环，只跑有 token 的 block：**5.2 ms/层**（block 512）。
- **教训**：同一个 kernel 的不同实现方式，性能可以差 3 倍。先单层拆开测，再改整图。

### 阶段 6（10-12）：建立精度验证方法

- **问题**：换 kernel 以后，怎么判断"出的图不一样"是 bug 还是正常的数值漂移？
- **做法**：
  - pipeline 里加了 `_StepTrace`，可以录下每步的输入和输出，也可以强制每步使用录好的输入（teacher forcing）。
  - 在 CPU 上用 fp32、TP1、纯 torch 跑同一个 pipeline 类，得到 gold（step 0、25、49）。
- **结论**：三种正确的 MoE 实现相对 fp32 都在 5.5e-2 左右，彼此之间也差 3.5e-2。这就是 **bf16 噪声底**。之后所有改动都拿这个标准衡量。
- **顺手发现的 bug**：`run.py --moe-kernel torch` 之前根本没生效。验证某个变体之前，先在日志里找能证明它确实生效的那一行。

### 阶段 7（10-12）：通信实测，推翻了"通信占 0.4 s"的猜测

- **当时的推断**：每步 0.76 s，各部分计算加起来约 0.35 s，所以剩下的 0.4 s"应该是"all-reduce，倒推出每次约 6 ms。
- **实测**：写了 `bench_allreduce.py`，在和模型完全相同的环境里，把 all-reduce 放进编译图里测。
  67 MB 一次只要 **1.07 ms**，每步 64 次合计约 68 ms。
- **另一个实测结论**：同一张图里的计算和通信**不会重叠**，放在一起反而更慢。所以原计划的"CFG 两分支交错，用计算掩盖通信"直接放弃了。
- **教训**：倒推出来的数不可信，要直接测量。

### 阶段 8（10-12）：和 GPU 同口径对比

- GPU 的基准是：不开 CFG、不跳步、50 步全算。我们核实了 GPU 的配置和日志。
- trn2 用相同输入（cabin prompt、think、seed 42）、也不开 CFG，DiT（denoise+VAE）是 **23.2 s**：H100 的 3.85 倍，g7e 的 1.70 倍。
- 用 trace 确认了 CFG 确实关掉了：每步预测的形状是 batch 1。

### 阶段 9（10-12）：profile 找到真正的瓶颈，UNet 切分，0.7585 → 0.5578 s/步

- **做法**：用 vLLM-Neuron 的 NeuronProfiler 抓了一步的设备时间线。
  - 坑：模型 NEFF 要按 UUID 从缓存里找出来，放到 profile 目录里才能解析。
- **发现**：
  - 一步 760 ms 里，**开头 66 ms 是 patch embed，末尾 179 ms 是 final layer**，32 个 rank 每个都把它们完整算了一遍。
  - final layer 的 167 ms 是 profiler 标出来的"excessive small DMA"，来源是 67 MB 的 token↔图像布局转置。
  - 通信 109 ms，其中约 40 ms 是在等别的 rank。之前猜的"MoE 负载不均"只能解释这 40 ms。
- **做法**：`unet_parallel.py` 按 GroupNorm 的 32 个组切通道，每个 rank 一组，归一化的统计量在本地就能算完。卷积按输出通道切，输入先 all-gather。大转置只转本 rank 自己的那一片。
- **结果**：每步 −26.5%，请求 39.9 → 29.9 s。精度 5.25e-2，在噪声底内。
- **顺带评估**：GPU 上用的 GroupNorm+SiLU 融合 kernel，切分后只能省不到 0.5 ms/步，**没做**。

### 阶段 10（10-12）：跳步缓存，cache-dit 胜过 TeaCache

把两种方法都移植过来做了对比（Mars prompt，CFG 2.5，PSNR 以不跳步的图为参照）：

| 方法 | 实际算的步数 | 请求耗时 | PSNR |
| --- | ---: | ---: | ---: |
| 不跳步 | 50 | 29.89 s | — |
| cache-dit 0.08 | 33 | 21.35 s | 34.1 dB |
| **cache-dit 0.12（选为默认）** | 25 | **17.15 s** | **32.1 dB** |
| cache-dit 0.24（cache-dit 自己的默认） | 17 | 12.93 s | 27.3 dB |
| TeaCache 0.1 | 17 | 11.54 s | 25.9 dB |
| TeaCache 0.2（vLLM-Omni 给这个模型的默认） | 10 | 7.67 s | 18.9 dB，构图和色调明显变了 |

- **结论**：速度相近时 cache-dit 画质更好。TeaCache 只看 timestep，cache-dit 每步至少算一层、看的是图像内容。
- **实现要点**：
  - 每步拆成 3 张图：前段、后段、命中缓存时的段。
  - host 每步拷回一个标量做判断；32 个 rank 用 gloo 做一次 MIN 投票，防止各自选了不同的图导致卡死。

### 阶段 11（10-12）：发版 v0.24.0.0.2.0

- 默认配置：TP32 + UNet 切分 + cache-dit 0.12，HTTP 上 17.8 s（只出图）、28.6 s（think）。
- **发版前发现的两个上游接口问题**：
  1. vLLM-Omni 0.24 的 diffusion chat 接口会**丢掉请求里的 `bot_task`**。我们用 `model_config.bot_task` 作为服务级默认值绕过去了。
  2. 接口不会设置 `guidance_scale_provided`，所以**通过 HTTP 关不掉 CFG**，离线 `run.py` 不受影响。这个只记录了，还没修。
- 仓库是**公开的**。推送前扫过 diff，确认没有桶名、账号、实例 ID、密钥。

### 阶段 12（10-13）：端到端对比，AR 为什么快、DiT 为什么慢

- 同口径（think、不开 CFG、不跳步）端到端：trn2 27.48 s，GPU 约 42 s。
- **AR 快**：
  - trn2 的 decode 是编译好的整图。
  - GPU 那边跑的是 eager 模式。
  - 而且 GPU 的 AR 和 DiT 分在两组卡上，中间还要传 KV。
- **DiT 慢**（不开 CFG、batch 1 的 profile，一步 328.5 ms）：
  - all-reduce 69 ms；
  - attention 59 ms，有 7.1 GB spill；
  - 编译器生成的数据搬运 73 ms；
  - MoE 44 ms、RMSNorm 25 ms、router 15 ms；
  - 39% 的时间核是空闲的，MFU 约 7%。
- 这份 profile 直接决定了后面三项优化的方向。

### 阶段 13（10-13）：TP16 × CP2，再快约 20%

- **思路**：TP32 下，每层 all-reduce 的数据量、RMSNorm、router 都是每个 rank 把全部 token 算一遍。如果每个 rank 只拿一半 token，这些开销都减半。
- **先测通信**：给 `bench_allreduce.py` 加了 `tpcp` 模式。
  - TP16 组内 16.8 MB 的 all-reduce 一次 0.381 ms（每步 24 ms，TP32 是 55 ms）；
  - CP 对内交换 K/V 一次 0.071 ms（每步 2.3 ms）。
  - 分组可以路由，两组同时跑也互不影响。
- **做法**：
  - TP 组是连续的 rank 块 [0..15] 和 [16..31]，正好各占环网的一行；CP 对是 (r, r+16)，位于相邻芯片、核内偏移相同。
  - 4097 个 token pad 到 2×2049。只有 CP rank 0 拿到 timestep token，rank 1 那份的 "timestep bias" 改用 image bias。
  - 每层 attention 前在 CP 对内交换 K/V；final layer 前把 token 收集回来。
  - UNet 仍然切在全部 32 个 rank 上。
- **AR 意外变慢**：TP16 时每个核每个 token 要读两倍的权重，decode 从 15.5 涨到 23.8 ms/token。
  解决办法：CP 对里两个 rank 的权重完全相同，所以让它们各算一半（一半 query 头、一半本地专家、一半共享专家的列、一半 LM head），再在全部 32 个 rank 上求和。降到 **15.7 ms/token**。
- **编译器坑**：开 CFG（batch 2）时，32 层的图报 `NCC_IBTN020 step out of int16 range`，而 2 层的图能编过。
  把"整批一次 gather 再做 `[cp, B]→[B, cp]` 转置"改成"每个 batch 行单独 gather"就好了。教训：先用截断层数（`--num-layers 2`）把流程跑通，问题只在全深度出现时，就去找跟数据布局有关的写法。
- **结果**：
  - CFG 不跳步：0.558 → 0.453 s/步；
  - 默认请求：17.15 → 14.14 s；
  - think 同口径：27.48 → 24.0 s；
  - 精度 5.26e-2，缓存跳过的 25 步和 TP32 完全一致。

### 阶段 14（10-13）：CFG 并行，以及一次判断失误

- **思路**：开 CFG 时，与其每组各算两个分支的一半 token，不如两组各算一个分支的全部 token。这样不用交换 K/V，patch embed 也只需算 batch 1。
- **失误**：我先用"16 个核单独跑一个 TP16 组、不开 CFG"做代理测量，得到 0.496 s/步，比 token 切分的 0.453 慢，就下了"CFG 并行不划算"的结论。
  用户追问"你是这样测的吗"以后，我把真正的实现做了出来：**0.442 s/步，反而更快**。
  代理测量的问题在于：UNet 只切在 16 个 rank 上，每个 rank 多算了一倍；真实方案的 UNet 切在 32 个 rank 上。
- **做法**：
  - 新增 `cfg_parallel: true`。
  - 两个分支的 latent 和 timestep 完全相同，所以 patch embed 仍在 32 个 rank 上算 batch 1。
  - final layer 前把两个分支的 hidden 拼成 batch 2，32 个 rank 一起算，每个 rank 都拿到两个预测。
  - 图通过 RoPE 表的长度判断当前是哪种模式：2049 表示 token 切分，4097 表示分支并行。
- **结果**：默认请求 14.14 → 13.97 s，精度 5.40e-2。
- **教训**：代理测量要先列清楚它和真实方案的每一处差异。拿不准就直接做真的。

### 阶段 15（10-13）：denoise attention 换 flash kernel，再快约 20%

- **问题**：denoise 的 attention 一直是 torch 写的 softmax，交给编译器自己融合。它会把完整的 fp32 分数矩阵 `[头数, 2049, 4610]` 写进 HBM，造成大量 spill。
- **又一次失误**：代码里有一条旧注释说 "attention_cte 不支持逐位置 mask"，我照着它告诉用户"没法直接换"。用户反问"不是有 NKI flash kernel 吗"，我去节点上查了实际安装的 kernel 签名：它支持 prefix caching（`k_prior`/`v_prior`，外加运行时可变的 `prior_used_len`），还支持 `bound_min/max` 和 `position_bias`。
- **做法**：
  - 右侧 pad 过的 prompt K/V 当作前缀，`prior_used_len` = prompt 实际长度，在图里从 key bias 数出来，所以不同长度的 prompt 共用一个 NEFF。
  - 本步的图像 K/V 当作 active 部分，`causal_mask=False`。
  - timestep 那一行只能看到 prompt 和它自己，继续用 torch 算。
  - pipeline 会先核对 generation mask，确认图像 token 能看到整个 step，不满足就直接报错。
- **坑**：`prior_used_len` 必须传 **float32**。kernel 会把它原样 DMA 进 fp32 的缓冲区，不做类型转换。
- **验证**：单核和 torch 对比，误差 3.5e-3。用 512 个 prompt 位置只用 8 个这种极端情况测，误差 4.1e-3，证明 pad 确实被屏蔽了。整模型精度 5.27e-2。
- **结果**：
  - CFG：0.442 → 0.357 s/步；
  - 不开 CFG：0.264 → 0.209 s/步；
  - 默认请求 12.6 s；
  - think 同口径 21.1 s，DiT 12.3 s，首次比 g7e 快。

---

## 4. 性能演进总表

**只出图、CFG 2.5 的请求**（Mars prompt；"每步"是不跳步时的数）：

| 时间 | 改动 | 每步 | 请求 |
| --- | --- | ---: | ---: |
| 10-09 | Lite+NKI 跑通（VAE 在 CPU 上） | 0.895 s | 72.8 s |
| 10-10 | MoE 换 NKI kernel（持平） | 0.911 s | 72.1 s |
| 10-11 | VAE 在设备上，bf16 | 0.90 s | 47.6 s |
| 10-11 | VAE 用 fp16，独占一个核 | 0.90 s | 47.0 s |
| 10-12 | MoE 换 hybrid 动态循环 | 0.760 s | 39.9 s |
| 10-12 | UNet 切分 | 0.558 s | 29.9 s |
| 10-12 | + cache-dit 0.12（**发版**） | 0.558 s | 17.15 s |
| 10-13 | TP16×CP2 | 0.453 s | 14.14 s |
| 10-13 | CFG 并行 | 0.442 s | 13.97 s |
| 10-13 | flash attention | **0.357 s** | **12.6 s** |

**和 GPU 同口径**（think、不开 CFG、不跳步，cabin prompt）：

| 时间 | DiT（denoise+VAE） | 整次请求 |
| --- | ---: | ---: |
| 10-12 上午（MoE hybrid） | 23.2 s | 32.0 s |
| 10-12 下午（UNet 切分） | 18.8 s | 27.5 s |
| 10-13（TP16×CP2） | 15.1 s | 24.0 s |
| 10-13（flash attention） | **12.3 s** | **21.1 s** |
| 参考：g7e / H100 | 13.68 / 6.03 s | 约 42 / 42 s |

---

## 5. 经验教训

### 5.1 方法论

1. **先测量，再优化，不靠倒推**。"通信占 0.4 s"和"MoE 负载不均"两个推断都被实测推翻了。真正的瓶颈（UNet 被每个 rank 重复计算、碎 DMA、attention spill）都是 profile 直接看到的。
2. **代理测量要列清和真实方案的差异**。CFG 并行那次，代理方案少切了一半 UNet，结论正好反了。
3. **先查实际安装的东西，再说"做不到"**。代码注释和文档可能过时，`inspect.signature` 一下就能看到 kernel 的真实能力。
4. **精度要逐步比**。50 步之后的图会因为累加顺序不同漂移约 35 dB，没法用来判断对错。要用 teacher forcing 加 fp32 gold，和"bf16 噪声底"对比。
5. **小步验证**。kernel 先单核测（`check_nki_kernels.py`），模型先截成 2 层跑通（`--num-layers 2`），最后才上 32 层全量。全量冷编译一次要 10 分钟到 2 小时。

### 5.2 Trainium / 编译器特有

1. **NEFF 缓存 key 里有源码行号**。在被追踪函数的上方加一行就会重编；换安装路径也会。所以缓存备份只对"这个提交 + 这个路径"有效。
2. **mask 用比较表达式构造**，不要在图里写新张量（会触发 `NCC_IMPR902`）。
3. **大张量的 batch 维转置可能触发 `NCC_IBTN020`**。换成逐行处理。
4. **低精度求和是精度的头号杀手**。GroupNorm、softmax 的统计量要用 fp32 累加。
5. **用 `compiler_args` 设置的编译选项会覆盖环境变量 `NEURON_CC_FLAGS`**。
6. **碎 DMA 很贵**。大布局转置尽量只转自己那一片。
7. **同一张图里，计算和集合通信不会自动重叠**。

### 5.3 运维

1. 调试前确认 `NodeRecovery=None`，否则一次设备卡死就会换掉整台节点。
2. 长任务用 `systemd-run` 或 `nerdctl exec -d` 启动，不要用 `setsid nohup`。
3. 停进程按 PID 停，不用 `pkill -f`（它会匹配并杀掉自己）。
4. 集群在 us-east-2，用 `HP_REGION`，不要继承本机为 Bedrock 设置的 `AWS_REGION`。
5. 节点没有 S3 写权限，用 presigned URL 上传。每次关机前都要把 NEFF 缓存备份到 S3，冷编译动辄几十分钟。
6. 仓库是公开的，推送前要扫描 diff 里的基础设施标识。

---

## 6. 和 GPU 的对比该怎么看

- **公平的部分**：两边输入相同，都不开 CFG、不跳步、跑满 50 步；GPU 也在用 FlashAttention。
- **不对等的部分**：
  - GPU 基准是 vLLM-Omni 的默认配置：eager 模式，DiT TP4，没开序列并行。
  - GPU 那边如果加上 CUDA graph / torch.compile、开序列并行，AR 和 DiT 都还有提升空间。AR 的空间尤其大，我们 3.6–4 倍的 AR 优势里，有一部分来自 GPU 没做优化。
- **我们的优化有多少能迁移到 GPU**：

| 优化 | 能否迁移 |
| --- | --- |
| flash attention | 不能，GPU 本来就有，我们是在补课 |
| UNet 切分 | 收益很小。大头是 Trainium 特有的碎 DMA |
| TP16×CP2 | 可能有用（g7e 卡间走 PCIe），要实测 |
| CFG 并行 | vLLM-Omni 本来就支持 |
| 跳步缓存 | vLLM-Omni 本来就支持，GPU 开了也有同样的倍数，所以同口径对比两边都关 |

- **还缺的数据**：GPU 调优后的上限。需要一台 g7e 测 TP4 对 TP2×SP2，以及开不开 torch.compile。
- trn2 是一台机器串行跑 AR 和 DiT，只用了 33/64 个核；GPU 用了 8 张卡，两组之间可以流水。所以上表比的是**单请求延迟**，不是吞吐。

---

## 7. 还没做的事

1. 发版后的提交还没合进 main，也没重新发版。
2. 这些改动之后，HTTP 服务路径还没重测过。
3. HTTP 关不掉 CFG（vLLM-Omni 0.24 的接口问题），可以像 `bot_task` 那样加一个服务级的 `guidance_scale`。
4. 只验证过 1024×1024。其他分辨率每种都要重新编译一套图（VAE 一套约 47 分钟）。
5. 下一轮 profile。剩下的大项估计是 MoE、编译器生成的数据搬运，以及空闲等待（芯片降频、等最慢的 rank）。
6. 33 个核以外还有 31 个核空闲。可以跑第二个实例做吞吐，也可以把 AR 和 DiT 流水起来。

---

## 8. 动手复现

> 下面的路径都是容器内路径。插件装在 `/workspace/plugin`，NEFF 缓存只对这个路径有效。

```bash
# 只出图，默认配置（TP16×CP2、CFG 并行、flash attention、cache-dit 0.12）
python examples/hunyuan_image3/run.py --model-path <权重目录> --seed 42 --profile --runs 3

# 和 GPU 同口径：think、不开 CFG、不跳步
python examples/hunyuan_image3/run.py --model-path <权重目录> --seed 42 \
  --stage-config examples/hunyuan_image3/hunyuan_image3_stage_ar.yaml \
  --bot-task think --guidance-scale 0 --step-cache 0 \
  --prompt "A cozy wooden cabin in a snowy pine forest at dusk, ..." --profile --runs 3

# 常用开关
#   --context-parallel 1        回到 TP32
#   --cfg-parallel off          开 CFG 时改回 token 切分
#   --step-cache 0              关掉跳步缓存
#   --num-layers 2              只跑 2 层，用来快速跑通流程（出的图没有意义）
#   model_config.denoise_attention: torch    改回 torch 版 attention

# 起服务（OpenAI 兼容接口）
bash examples/hunyuan_image3/serve.sh <权重目录> 8091 [stage yaml]

# CPU 单测（7 个）
VLLM_NEURON_CPU_MODE=1 python test/unit/test_hunyuan_image3_context_parallel.py   # 以及 numerics / sharding / unet_parallel / step_cache / tea_cache / ar_decode

# 设备上单独验证 kernel（几分钟）
NEURON_RT_VISIBLE_CORES=0 python examples/hunyuan_image3/check_nki_kernels.py --tp-size 16 --only denoise_attention

# 通信微基准（32 个 rank）
torchrun --nproc-per-node 32 examples/hunyuan_image3/bench_allreduce.py --only tpcp

# 精度：录 trace 和 teacher forcing，再对比
HY3_TRACE_FORCE_DIR=<基准 trace> HY3_TRACE_DIR=<新 trace> python examples/hunyuan_image3/run.py ... --step-cache 0
python examples/hunyuan_image3/compare_step_traces.py --reference fp32=<cpu_fp32 trace> new=<新 trace>
```
