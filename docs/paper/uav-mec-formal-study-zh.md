# 事件驱动 UAV-MEC 混合动作强化学习的可审计实证研究

## 摘要

无人机辅助移动边缘计算（UAV-MEC）同时涉及任务卸载、无人机移动、无线接入、队列与能耗，形成离散目标选择与连续三维移动耦合的混合动作问题。本文在 EdgeCloudSim 上构建事件驱动研究系统：Java 独占物理状态、事件队列和状态转移，Python 仅通过 GymBridge 1.2 接口训练和评估策略。每次任务到达构成一个决策时刻，相邻决策间隔随仿真事件变化，因此系统按半马尔可夫决策过程使用转移特定折扣。本文比较 masked parameterized-action DDPG、mixed-action PPO、mixed-action TD3，以及零移动 masked DQN 动作能力消融，并设置随机、最小估计时延、本地执行和云执行四类非学习基线。

正式实验为每种学习算法独立训练 10 个最终 checkpoint，每个 checkpoint 固定使用 200,000 次真实 GymBridge 交互，共计 8,000,000 次训练交互。每个 checkpoint 随后在 10 个城市验证种子和 10 个未参与调参的农村留出种子上评估。统计推断以独立训练 checkpoint 为学习策略的重采样单位，执行 20,000 次确定性非参数 bootstrap；配置、环境、源码、checkpoint 和原始报告哈希均通过聚合审计。

农村留出结果不支持学习策略优于简单启发式。最小估计时延基线在全部留出种子上退化为本地执行，奖励为 2140.881，截止时间成功率为 99.938%，平均时延为 1.266 s。学习策略中，DQN 的平均奖励最高（2066.346），但成功率降至 98.610%，平均时延升至 7.351 s，并出现训练种子长尾；DDPG 的成功率最高（99.883%），但相对最小估计时延基线的配对奖励差为 −655.049，95% 置信区间为 [−659.935, −650.301]。本文不将 DQN 与混合动作算法的差异解释为移动能力的因果效应，也不宣称任何学习算法具有普遍优越性。结果表明，在当前轻载配置中，本地执行已构成强基线，而动作约束、价值估计稳定性和跨配置泛化仍是主要瓶颈。

**关键词：** 无人机辅助移动边缘计算；混合动作强化学习；半马尔可夫过程；留出评估；负结果；可复现实验

## 1. 引言

UAV 可以利用三维机动性改善空地链路，并在固定基础设施不足时提供中继或边缘计算服务，但轨迹、卸载目标、无线资源、任务队列和能量约束相互耦合，使在线控制成为高维、随机且非凸的问题 [1–3]。深度强化学习能够直接从交互中优化序贯决策，但其结果容易受到随机种子、实现细节、评估单位和不完整基线的影响 [12,13]。如果仿真物理和学习环境分别在两种语言中实现，还可能出现状态复制、时间推进不一致或指标来源不清等额外风险。

本文不提出新的通用强化学习算法，而是研究一个更基础的问题：在真实事件驱动仿真边界、固定训练预算和可审计统计协议下，常用混合动作学习方法能否在未见配置中优于简单策略。实验保留并分析负结果，以区分“训练曲线改善”“系统可复算”和“留出性能优越”三个不同命题。

### 1.1 研究问题

本文研究以下问题：

1. 在 Java 事件驱动仿真边界内，混合动作强化学习能否在未见农村配置中优于非学习基线？
2. 混合动作策略与零移动 DQN 动作能力消融呈现何种行为和稳定性差异？
3. 如何让长周期强化学习实验的配置、种子、环境、代码、checkpoint、原始报告与统计推断可追溯？

第二个问题只进行描述性比较。DQN 与 DDPG、PPO、TD3 同时存在算法结构和动作能力差异，因而本实验不能识别“允许移动”本身的因果效应。严格检验该效应需要增加同算法零移动版本。

### 1.2 主要贡献

- 建立 Java 物理仿真与 Python 学习逻辑严格分离的 GymBridge 1.2 接口，防止 Python 侧复制或旁路物理状态；
- 将任务到达间隔建模为事件驱动 SMDP，并向所有学习算法提供转移特定折扣；
- 在同一观测、奖励、训练预算和种子划分下比较三种混合动作扩展，并把 DQN 明确限定为零移动动作能力消融；
- 完成 4 算法 × 10 独立训练种子的正式矩阵、配对验证/留出评估和 20,000 次 bootstrap；
- 公开没有战胜基线的证据，并用源码、配置、环境、checkpoint 和报告哈希约束可复现边界。

## 2. 相关工作

### 2.1 UAV 通信与移动边缘计算

UAV 通信相较地面网络具有更强的视距传播概率和可控三维位置，同时受到载荷、能量和干扰限制 [1]。UAV-MEC 研究通常联合优化卸载比例、资源分配与轨迹。例如，Hu 等研究了用户本地计算与 UAV 卸载的联合轨迹设计 [2]；Zhou 等研究了 UAV 使能 MEC 中的卸载与轨迹优化 [3]。这类工作多以离散时隙和显式优化模型为基础。本文关注另一边界：任务异步到达、已提交任务在事件队列中并发推进、学习策略只在任务到达时决策。

EdgeCloudSim 在 CloudSim 基础上提供边缘计算的网络、计算、移动性和负载建模能力 [4]。本文保留其事件内核和 VM 调度器，并扩展 UAV、空地信道、Gym 决策协调和可审计报告。空地链路采用 Al-Hourani 等提出的概率 LoS/NLoS 平均路损形式 [5]。

### 2.2 混合动作强化学习

任务到达之间的持续时间不固定，适合用 SMDP 的时间抽象描述 [6]。卸载目标为离散动作，UAV 位移为连续参数，属于参数化动作空间 [7]。本文以 DQN [8]、DDPG [9]、PPO [10] 和 TD3 [11] 为基础实现四种受动作 mask 约束的策略。由于原始 DDPG 和 TD3 假定连续动作，本文使用的版本是面向 `target + movement` 的参数化动作扩展，不应被理解为原算法的逐字复现或理论改进。

### 2.3 强化学习实验可靠性

深度强化学习对种子、超参数和实现细节高度敏感，单次运行或只报告最好 checkpoint 容易高估改进 [12]。少量独立运行还会使点估计和区间结论不稳定 [13]。本文因此使用固定预算最终 checkpoint、独立训练种子、配对环境种子和 checkpoint 级 bootstrap。与现有工作相比，本文的重点不是提出更高的点估计，而是把事件驱动仿真、学习实现和统计产物连接为可审计证据链。

## 3. 系统与仿真模型

### 3.1 事件驱动边界

每个 Gym step 对应一次任务到达。Java 线程在到达事件上冻结决策边界，生成观测和动作 mask；Python 返回卸载目标与 UAV 移动命令；Java 随后继续推进上传、排队、计算、下载、能耗和失败事件，直至下一次任务到达或仿真终止。此前提交的任务不会因 Gym step 结束而暂停。

```mermaid
flowchart LR
    A["Java：任务到达事件"] --> B["冻结决策边界<br>生成观测与 mask"]
    B --> C["Python：策略选择<br>target + movement"]
    C --> D["Java：约束检查与 UAV 移动"]
    D --> E["Java：网络、队列、计算和能耗事件"]
    E --> F["结算已完成/失败任务<br>累计奖励与物理指标"]
    F --> G["下一任务到达<br>effective_discount"]
    G --> B
```

Java 是仿真时钟、物理状态和指标的唯一权威来源。Python 不预测带宽、队列或能量，也不自行推进时间。每个正式训练或评估报告都启动新的 JVM，避免跨报告共享静态状态。

### 3.2 SMDP 折扣

若相邻决策时刻为 \(t_k\) 和 \(t_{k+1}\)，则

\[
\Delta t_k=t_{k+1}-t_k.
\]

基础折扣为 \(\gamma_0=0.99\)，时间单位为 \(\tau_d=1\ \mathrm{s}\)，第 \(k\) 个转移使用

\[
d_k=\gamma_0^{\Delta t_k/\tau_d}.
\]

后端将 \(d_k\) 写入 `effective_discount`，经验回放和 PPO rollout 均存储该值。训练目标不得以固定“每 Gym step 折扣”替代它。由于观测是完整 CloudSim 状态的压缩表示，本文把学习接口视为部分可观测控制接口，而不声称 31 维观测构成充分马尔可夫状态。

### 3.3 观测

当 UAV 数量 \(U=2\) 时，连续观测共 31 维，另含 4 位布尔动作 mask。

| 观测组 | 维数 | 内容 |
|---|---:|---|
| 时间 | 2 | 归一化仿真时间、相邻决策时间间隔 |
| 当前任务 | 7 | 输入/输出数据量、任务长度、核心需求、时延敏感度、截止时间等 |
| 固定资源 | 6 | 本地与云端各自的可用能力、负载和预计链路时延 |
| UAV 状态 | \(8U=16\) | 三维位置、剩余能量、队列、可用能力、状态和链路特征 |
| 动作 mask | 4 | local、cloud、UAV 0、UAV 1 的当前可执行性 |

连续特征使用运行均值与方差归一化并截断到 \([-10,10]\)。mask 只约束离散目标分布，不作为连续移动特征。

### 3.4 动作、移动与约束

动作定义为

\[
a_k=(a_k^{\mathrm{target}},a_k^{\mathrm{move}}),
\]

其中

\[
a_k^{\mathrm{target}}\in\{\mathrm{local},\mathrm{cloud},
\mathrm{uav}_0,\mathrm{uav}_1\},
\qquad
a_k^{\mathrm{move}}\in[-1,1]^{2\times3}.
\]

对第 \(u\) 架 UAV，策略输出先按当前决策间隔缩放为期望三维位移，再受速度 \(v_{\max}=5\ \mathrm{m/s}\)、区域边界和高度区间 \(20\text{–}200\ \mathrm{m}\) 约束。若期望位移为 \(\Delta \mathbf{x}_{u,k}\)，实际位移满足

\[
\|\Delta \mathbf{x}_{u,k}^{\mathrm{exec}}\|_2
\le v_{\max}\Delta t_k.
\]

被 mask 排除的目标、超出物理边界的位移以及相关协议错误都会被记录为约束事件；系统不静默回退到其他卸载目标。约束计数可在一个决策内大于 1，因此它不是失败任务数。

### 3.5 空地接入信道

设地面设备与 UAV 的水平距离为 \(r\)、垂直距离为 \(h\)，三维距离和仰角分别为

\[
d=\sqrt{r^2+h^2},\qquad
\theta=\frac{180}{\pi}\arctan\frac{\max(h,0)}{\max(r,\epsilon)}.
\]

LoS 概率采用 [5] 的逻辑函数：

\[
p_{\mathrm{LoS}}
=\frac{1}{1+a\exp[-b(\theta-a)]}.
\]

载波频率为 \(f_c\)，光速为 \(c\)，平均路损为

\[
L_{\mathrm{dB}}
=20\log_{10}\left(\frac{4\pi f_c d}{c}\right)
+p_{\mathrm{LoS}}\eta_{\mathrm{LoS}}
+(1-p_{\mathrm{LoS}})\eta_{\mathrm{NLoS}}.
\]

信道增益 \(g=10^{-L_{\mathrm{dB}}/10}\)。总接入带宽 \(B\) 被划分为 \(N_c=4\) 个正交信道，每信道带宽 \(B_c=B/N_c\)。发射功率为 \(P_t\)，噪声功率谱密度为 \(N_0\)，则

\[
\mathrm{SNR}=\frac{P_t g}{N_0B_c},
\qquad
C_c=B_c\log_2(1+\mathrm{SNR}).
\]

若同方向并发传输数为 \(n\)，确定性共享规则令每个信道最多承载

\[
q=\left\lceil\frac{n}{N_c}\right\rceil
\]

个流，有效速率为 \(R=C_c/q\)。大小为 \(L\) KB 的传输时延为

\[
T_{\mathrm{access}}
=T_0+\frac{d}{c}+\frac{8L}{1000R_{\mathrm{Mbps}}}.
\]

上传和下载分别维护并发计数。云任务在决策时选择预计接入往返时延最小且仍有能量的 UAV，同一任务的上、下行保持同一中继。

### 3.6 云回传、计算队列与能耗

UAV—云回传带宽为 \(B_{\mathrm{bh}}=100\ \mathrm{Mbps}\)，同方向并发云任务均分带宽。单程传播项为

\[
T_{\mathrm{bh,prop}}=0.02+\frac{20\,000}{c}.
\]

云任务总通信时延由接入上传、回传上传、回传下载和接入下载组成。Local 任务不产生无线传输时延；UAV 和云任务的上传完成后才进入相应计算队列。Local 与 cloud 使用 CloudSim 的 time-shared VM 调度，UAV 使用最大长度为 50 的先进先出队列并按仿真秒处理 MI。

UAV 初始能量为 \(E_{\max}=200{,}000\ \mathrm{J}\)。每个仿真秒收取 \(P_{\mathrm{hover}}=20\ \mathrm{W}\) 的悬停基线；实际飞行时间额外收取

\[
E_{\mathrm{move}}
=\max(P_{\mathrm{flight}}-P_{\mathrm{hover}},0)T_{\mathrm{flight}},
\]

其中 \(P_{\mathrm{flight}}=50\ \mathrm{W}\)。UAV 计算能耗为 \(0.001\ \mathrm{J/MI}\)。UE 上传能耗按上传时延乘 \(0.1\ \mathrm{W}\) 计算。

### 3.7 奖励

在相邻决策时刻之间完成的每个任务贡献

\[
r_i=\operatorname{clip}\left(
s_i-0.35\operatorname{clip}\left(\frac{\ell_i}{d_i},0,2\right)
-0.15\operatorname{clip}\left(\frac{e_i^{\mathrm{UE}}}{10},0,2\right),
-1,1
\right),
\]

其中 \(s_i=1\) 表示任务成功且在截止时间内完成，否则为 0；\(\ell_i\)、\(d_i\) 和 \(e_i^{\mathrm{UE}}\) 分别为任务时延、截止时间和 UE 能耗。

区间奖励为

\[
r_k=\sum_{i\in\mathcal{S}_k}r_i
-0.20\operatorname{clip}\left(
\frac{e_k^{\mathrm{UAV}}}{U E_{\max}},0,2\right)
-0.30\mathbb{1}[c_k>0],
\]

其中 \(\mathcal{S}_k\) 是该区间结算的任务集合，\(c_k\) 是目标 mask 与移动边界约束事件总数。一个区间可结算多个任务，因此 \(r_k\) 不要求落在 \([-1,1]\) 内。

### 3.8 正式参数

| 参数 | 城市训练/验证 | 农村留出 |
|---|---:|---:|
| UAV 数量 | 2 | 2 |
| 移动设备数量 | 25 | 15 |
| 区域 | \(1000\times1000\times200\ \mathrm{m}\) | \(2000\times2000\times200\ \mathrm{m}\) |
| 仿真时长 | 3600 s | 3600 s |
| UAV 初始高度 | 80 m | 80 m |
| UAV 计算能力 | 由种子确定的 1500–2500 MIPS | 同左 |
| 任务到达 | 每设备指数分布，均值 12 s | 同左 |
| 任务输入/输出 | 1500 KB / 25 KB | 同左 |
| 任务长度/截止时间 | 3000 MI / 100 s | 同左 |
| 接入带宽/信道数 | 10 Mbps / 4 | 同左 |
| 发射功率/噪声 PSD | 0.1 W / −174 dBm/Hz | 同左 |
| 载波频率 | 2 GHz | 2 GHz |
| \(a,b\) | 9.61, 0.16 | 4.88, 0.43 |
| \(\eta_{\mathrm{LoS}},\eta_{\mathrm{NLoS}}\) | 1, 20 dB | 0.1, 21 dB |
| 回传带宽/距离 | 100 Mbps / 20 km | 同左 |
| UAV 能量/功率 | 200 kJ；飞行 50 W；悬停 20 W | 同左 |

地面设备在三个固定接入点 \((500,500)\)、\((200,200)\)、\((800,800)\) 间按设备编号映射，正式场景没有连续地面移动。因此，“城市到农村”同时改变设备数、区域大小和 LoS 参数，属于混杂的跨配置评估，不应被解释为单一环境因素的因果效应。

## 4. 学习算法

### 4.1 共同处理

四种算法均接收相同连续观测和 mask，使用两层 128 单元的 MLP、Adam、CPU 训练、运行观测归一化和梯度裁剪。所有 TD 目标使用转移特定折扣 \(d_k\)。终止转移不 bootstrap；协议截断允许从最后观测 bootstrap。训练只保存固定预算后的最终 checkpoint，不按验证或留出性能挑选最好模型。

### 4.2 Masked parameterized-action DDPG

Actor 共享两层 trunk，分别输出四个目标 logits 和 \(6\) 维 tanh 移动。交互时在合法目标中执行 argmax；actor 更新时使用 hard straight-through masked softmax，使离散 one-hot 的前向选择与连续近似梯度结合。Critic 输入

\[
(o_k,\operatorname{onehot}(a_k^{\mathrm{target}}),
a_k^{\mathrm{move}})
\]

并输出单一 \(Q\) 值。目标为

\[
y_k=r_k+d_k(1-z_k)
Q_{\bar\phi}(o_{k+1},\pi_{\bar\theta}(o_{k+1})),
\]

其中 \(z_k\) 为终止标记。本文版本使用单 critic、每步 actor 更新，不使用 TD3 目标平滑。

### 4.3 Mixed-action PPO

PPO 将联合策略分解为 masked categorical 目标分布与 tanh-squashed 对角高斯移动分布：

\[
\pi(a_k|o_k)
=\pi_d(a_k^{\mathrm{target}}|o_k,m_k)
\pi_c(a_k^{\mathrm{move}}|o_k)
\]

在概率意义上为两者乘积，联合 log-probability 为两项之和。优势估计使用 \(d_k\) 替代固定折扣，并以 \(\lambda=0.95\) 递推。优化目标包含 PPO clipped surrogate、value loss 和熵正则。

### 4.4 Mixed-action TD3

TD3 与 DDPG 使用相同参数化 actor，但维护两个 critic。目标值为

\[
y_k=r_k+d_k(1-z_k)
\min_{j\in\{1,2\}}
Q_{\bar\phi_j}(o_{k+1},a'_{k+1}).
\]

目标平滑噪声只作用于连续移动，不作用于离散目标；actor 每两个 critic 更新步骤更新一次。该设计保留 TD3 的 clipped double Q、delayed policy update 和 target smoothing，同时明确其混合动作扩展边界。

### 4.5 Masked DQN 零移动消融

DQN 输出四个离散目标的 \(Q\) 值并将非法动作设为不可选，目标为

\[
y_k=r_k+d_k(1-z_k)
\max_{a'\in\mathcal{A}(m_{k+1})}Q_{\bar\phi}(o_{k+1},a').
\]

它始终输出零移动，但仍可将任务卸载到保持初始位置的 UAV。该模型用于观察“只学习目标选择”的行为，不是与三种混合动作方法同能力的主基线，也不能单独识别移动的收益。

### 4.6 超参数

| 参数 | DDPG | PPO | TD3 | DQN |
|---|---:|---:|---:|---:|
| 学习率 | actor/critic \(3\times10^{-4}\) | \(3\times10^{-4}\) | actor/critic \(3\times10^{-4}\) | \(3\times10^{-4}\) |
| Batch/minibatch | 128 | 128 | 128 | 128 |
| Replay/rollout | 100,000 | 2,048 | 100,000 | 100,000 |
| 学习起点 | 1,024 | — | 1,024 | 1,024 |
| 隐层 | 128, 128 | 128, 128 | 128, 128 | 128, 128 |
| 软更新 \(\tau\) | 0.005 | — | 0.005 | — |
| 连续探索噪声 | 0.1 | \(\log\sigma_0=-0.5\) | 0.1 | — |
| 离散探索 | 0.1 | 熵系数 0.01 | 0.1 | \(\epsilon:1\rightarrow0.05\) |
| 特有参数 | policy delay 1 | clip 0.2；10 epochs | delay 2；noise 0.2；clip 0.5 | target 每 250 步更新 |

PPO 的 value 系数为 0.5，最大梯度范数为 0.5；其余三种算法最大梯度范数为 10。DQN 的 \(\epsilon\) 在完整 200,000 次交互内线性衰减。

## 5. 实验设计与审计

### 5.1 对照、种子和预算

| 阶段 | 种子 | 用途 |
|---|---|---|
| 独立训练 | 10001, 20001, …, 100001 | 每算法 10 个 checkpoint |
| 城市验证 | 201–210 | 固定预算后评估，不选择 checkpoint |
| 农村留出 | 301–310 | 不调参、不早停、不选模型 |

每个 checkpoint 恰好训练 200,000 次真实 GymBridge 交互。非学习基线为：

- `random_masked`：在 mask 内均匀选择目标，并均匀采样移动；
- `minimum_estimated_delay`：选择观测中预计传输/固定资源时延最小的合法目标，移动为零；
- `local_only`：始终选择本地执行，移动为零；
- `cloud_only`：始终选择云执行，移动为零。

### 5.2 指标

预先指定的主指标为

\[
\mathrm{deadline\ success\ rate}
=\frac{\text{成功且按时完成任务数}}
{\text{已结算任务数}}.
\]

未结算、尚未到达和执行中任务分别记录并由审计核对。次要指标为总奖励、平均/P95 时延、每任务 UE+UAV 能耗、吞吐量、约束事件、UAV 队列和卸载比例。次要指标用于解释机制，不进行多重比较校正，不作为额外优越性声明。

`local_resource_utilization` 在全部正式报告中为 0。代码审查表明该字段是在任务到达决策边界对 CloudSim VM 调度器进行瞬时采样，而不是对忙碌时间积分；因此它不能代表区间平均本地利用率。本文从性能解释中排除该指标，不把“可复算”误写为“构念有效”。协议 1.2 的原始报告保持不变，后续协议应改为基于 busy-time 的时间积分。

### 5.3 统计单位

学习策略先在同一 checkpoint 内对 10 个环境种子取均值，再跨 10 个独立 checkpoint 均值 bootstrap。基线在同一环境种子上重复运行且通过位级确定性检查，区间直接跨 10 个环境种子 bootstrap。候选相对基线的效应先按相同环境种子相减、在 checkpoint 内平均，再跨 10 个 checkpoint bootstrap。

所有区间使用 20,000 次确定性 percentile bootstrap。基线之间以及同一候选的不同参考基线使用共同重采样索引；因此，如果两条策略的原始种子向量完全相同，其区间也必然相同。区间只描述声明种子总体下的不确定性，不证明跨任务分布的普遍优越性。

### 5.4 完整矩阵与 checkpoint

| 检查项 | 期望 | 完成 | 状态 |
|---|---:|---:|---|
| 训练报告 | 4 × 10 = 40 | 40 | 通过 |
| 每 checkpoint 交互 | 200,000 | 200,000 | 通过 |
| 总训练交互 | 8,000,000 | 8,000,000 | 通过 |
| 城市验证报告 | 40 | 40 | 通过 |
| 农村留出报告 | 40 | 40 | 通过 |
| bootstrap 重采样/区间 | 20,000 | 20,000 | 通过 |
| 原始报告审计 | 全部通过 | 全部通过 | 通过 |
| 配置/源码/环境/checkpoint 哈希 | 一致 | 一致 | 通过 |

聚合器只有在交互序列连续、预算准确、种子集合完整、原始指标可复算、重复基线确定且全部来源哈希一致时才输出 `passed`。正式矩阵使用的源码提交为 `f4ef625ac2a438de724cad12760f187ea87560dc`。

### 5.5 运行与复算环境

正式报告记录环境 XML、Java artifact、源码树和 checkpoint SHA-256。当前复算环境为 macOS aarch64、Maven 3.9.9、Maven 使用的 Java 1.8.0_432、Python 3.13.2、NumPy 2.5.1、PyTorch 2.13.0 和 Matplotlib 3.11.1。协议 1.2 未把 CPU 型号、RAM 容量和 wall-clock 时间写入报告，这是主机级复现信息的限制，不影响报告内的仿真时间和来源哈希审计。

## 6. 结果

### 6.1 城市验证与农村留出

下表先给出固定预算最终 checkpoint 的验证与留出均值。学习策略对 10 个 checkpoint 均值，基线对 10 个环境种子均值。

| 策略 | 验证奖励 | 留出奖励 | 验证成功率 (%) | 留出成功率 (%) | 验证时延 (s) | 留出时延 (s) |
|---|---:|---:|---:|---:|---:|---:|
| Random masked | 3652.940 | 2041.035 | 99.658 | 99.698 | 4.726 | 4.482 |
| Minimum estimated delay | 3801.220 | 2140.881 | 99.944 | 99.938 | 1.041 | 1.266 |
| Local only | 3801.220 | 2140.881 | 99.944 | 99.938 | 1.041 | 1.266 |
| Cloud only | 3777.432 | 2122.667 | 99.912 | 99.877 | 2.173 | 2.915 |
| DDPG | 2623.718 | 1485.832 | 99.605 | 99.883 | 2.448 | 2.332 |
| PPO | 3025.121 | 1585.020 | 99.362 | 99.783 | 3.053 | 2.824 |
| TD3 | 2645.603 | 1485.623 | 99.903 | 99.850 | 1.885 | 2.323 |
| DQN 零移动 | 3680.511 | 2066.346 | 98.579 | 98.610 | 6.087 | 7.351 |

城市到农村同时减少设备数量、扩大区域并更换 LoS 参数，因此奖励变化主要反映任务规模和配置变化，不能直接解释为单一的“农村泛化损失”。跨配置后，策略之间的相对排序比绝对总奖励更有意义。

### 6.2 农村留出总体性能

均值后的方括号为 95% bootstrap 置信区间。约束列报告每个决策的平均约束事件数，不把它解释为失败任务率。

| 策略 | 奖励 | 成功率 (%) | 平均时延 (s) | P95 时延 (s) | 能耗/任务 (J) | 约束/决策 |
|---|---:|---:|---:|---:|---:|---:|
| Random masked | 2041.035 [1820.001, 2242.757] | 99.698 [99.616, 99.783] | 4.482 [4.293, 4.639] | 13.620 [12.844, 14.367] | 154.350 [141.023, 169.412] | 0.112 |
| Minimum estimated delay | 2140.881 [1908.804, 2352.436] | 99.938 [99.914, 99.960] | 1.266 [1.218, 1.316] | 3.823 [3.623, 4.039] | 69.189 [61.684, 78.569] | 0 |
| Local only | 2140.881 [1908.804, 2352.436] | 99.938 [99.914, 99.960] | 1.266 [1.218, 1.316] | 3.823 [3.623, 4.039] | 69.189 [61.684, 78.569] | 0 |
| Cloud only | 2122.667 [1892.093, 2333.424] | 99.877 [99.847, 99.904] | 2.915 [2.717, 3.097] | 7.038 [6.531, 7.484] | 69.330 [61.827, 78.700] | 0 |
| DDPG | 1485.832 [1480.946, 1490.575] | 99.883 [99.847, 99.915] | 2.332 [1.795, 2.905] | 7.865 [5.957, 9.793] | 106.918 [97.264, 115.946] | 1.953 |
| PPO | 1585.020 [1531.114, 1644.563] | 99.783 [99.704, 99.852] | 2.824 [2.282, 3.347] | 7.911 [6.184, 9.924] | 111.038 [108.464, 113.849] | 1.270 |
| TD3 | 1485.623 [1480.732, 1490.220] | 99.850 [99.800, 99.897] | 2.323 [1.914, 2.722] | 6.421 [5.233, 7.601] | 107.543 [98.440, 116.174] | 1.935 |
| DQN 零移动 | 2066.346 [1965.436, 2121.523] | 98.610 [96.340, 99.811] | 7.351 [3.275, 14.392] | 18.302 [8.161, 34.942] | 69.994 [69.564, 70.477] | 0 |

![留出性能](figures/heldout_performance.svg)

Minimum estimated delay 在全部农村留出种子中选择 local，因此与 `local_only` 的十个原始值、均值和区间完全相同。该退化说明当前任务长度、截止时间和本地资源配置使本地执行成为强基线，而不是证明该启发式在其他负载下普遍最优。

### 6.3 相对最小估计时延基线的配对效应

| 学习策略 | 奖励差 | 成功率差（百分点） | 平均时延差 (s) | 能耗差 (J/task) |
|---|---:|---:|---:|---:|
| DDPG | −655.049 [−659.935, −650.301] | −0.055 [−0.090, −0.022] | +1.066 [+0.524, +1.633] | +37.729 [+27.994, +46.845] |
| PPO | −555.861 [−609.437, −496.318] | −0.154 [−0.235, −0.084] | +1.559 [+1.026, +2.075] | +41.849 [+39.245, +44.628] |
| TD3 | −655.257 [−660.209, −650.696] | −0.087 [−0.139, −0.041] | +1.058 [+0.648, +1.452] | +38.355 [+29.309, +46.909] |
| DQN 零移动 | −74.535 [−175.165, −19.400] | −1.328 [−3.593, −0.127] | +6.086 [+1.999, +13.116] | +0.805 [+0.373, +1.280] |

![配对效应](figures/paired_effects.svg)

四种学习策略的奖励差区间均完全低于 0，成功率差也均低于 0。所有平均时延差和能耗差区间均高于 0。DQN 在奖励和能耗上最接近本地基线，但其一个 checkpoint 的平均时延差达到 36.332 s、成功率差达到 −11.370 个百分点，使区间明显变宽。该现象支持报告训练种子分布，而不是只给总体均值。

### 6.4 卸载行为与约束诊断

| 策略 | Local (%) | Cloud (%) | UAV (%) | 总卸载 (%) | 约束/决策 |
|---|---:|---:|---:|---:|---:|
| Random masked | 25.3 | 25.0 | 49.7 | 74.7 | 0.112 |
| Minimum estimated delay / Local only | 100.0 | 0.0 | 0.0 | 0.0 | 0 |
| Cloud only | 0.0 | 100.0 | 0.0 | 100.0 | 0 |
| DDPG | 77.5 | 17.4 | 5.1 | 22.5 | 1.953 |
| PPO | 50.3 | 14.1 | 35.5 | 49.7 | 1.270 |
| TD3 | 54.9 | 37.2 | 7.9 | 45.1 | 1.935 |
| DQN 零移动 | 30.1 | 47.1 | 22.9 | 69.9 | 0 |

DDPG 和 TD3 的高约束率与其低奖励平台同时出现，但当前原始报告只保留“非法目标 + 边界约束”的合计，无法按类型分解，也不能证明约束是低性能的唯一原因。PPO 更多选择 UAV，TD3 更多选择 cloud，说明三种混合动作算法并未收敛到同一卸载模式。DQN 虽无移动能力，仍频繁选择云和固定位置 UAV；其长尾主要体现为目标选择与排队稳定性，而不是飞行越界。

### 6.5 训练与优化诊断

![训练奖励](figures/training_curves.svg)

DQN 的训练奖励整体最高，PPO 随交互增加而改善；DDPG 与 TD3 长期处于较低平台。训练奖励排序没有转化为相对留出基线的优势，说明单独观察训练曲线会高估策略质量。

![优化损失](figures/optimization_losses.svg)

损失图使用 2,000 交互分箱和跨 10 个训练种子的 bootstrap 区间。DDPG/TD3 的 actor/critic、PPO 的 policy/value 采用独立纵轴；不同算法的目标函数尺度不同，不能跨面板直接比较绝对值。损失只用于识别优化异常，不是系统性能指标。

## 7. 讨论

第一，当前正式场景更像“可审计轻载泛化基准”，而不是必须依赖 UAV/云资源的压力测试。Local only 的成功率已接近 100%，且 minimum estimated delay 完全退化为 local，使复杂策略缺少可利用的收益空间。下一版协议应在不查看留出结果的前提下预注册本地 CPU 拥塞、突发到达、严格截止时间、回传受限和 UAV 能量紧张等分层场景。

第二，混合动作算法没有在相同预算下战胜本地基线。DDPG/TD3 的高约束事件与低奖励表明动作构造需要更强的可行域归纳偏置，例如边界内参数化、投影层或分层目标—移动策略。仅扩大网络宽度无法解决目标合法性和三维边界问题。

第三，DQN 零移动消融接近本地基线的奖励和能耗，但存在显著长尾。这只能说明在当前配置中“固定 UAV、只学习目标”的策略有时接近强基线，不能说明 DQN 优于混合动作算法，更不能说明移动本身有害。需要给 DDPG、PPO 和 TD3 分别增加 movement-clamped-to-zero 版本才能隔离动作能力效应。

第四，城市与农村配置的总奖励不可直接比较。设备数从 25 降为 15 会改变任务总数，区域和 LoS 参数又同时变化。本文将农村结果解释为一个组合域的留出评估，而不是城乡因素的消融。

第五，吞吐量主要由到达过程和仿真时长限制，策略区分度弱；截止成功率、时延、能耗、卸载比例和约束事件更有解释力。由于成功率接近天花板，后续压力场景还应报告失败类型和生存/尾时延指标。

## 8. 有效性威胁与限制

- 结论仅适用于 2 UAV、三个固定地面接入点、声明任务参数和城市训练—农村留出的组合配置，不能外推到连续用户移动、真实无线信道或硬件部署。
- 10 个独立 checkpoint 允许执行预定 bootstrap，但对重尾失败模式仍有限；DQN 已显示单个不稳定 checkpoint 可以显著改变均值。
- 研究问题 2 不是同算法控制实验，不能识别移动能力的因果效应。
- `local_resource_utilization` 是决策边界瞬时采样且全部为 0，已从解释中排除；未来应改为 busy-time 积分并重新验证。
- 约束事件把非法目标和每架 UAV 的边界约束合并，可能在一个决策内累计多次，当前不能按失败机制分解。
- 评估报告保留目标比例，但没有保留可重建的逐时刻 UAV 轨迹；轨迹行为只能由新协议重新记录，不能从现有汇总推断。
- 本文没有进行超参数搜索。固定预算最终 checkpoint 避免留出泄漏，但可能低估需要不同优化配置的算法。
- 正式报告未记录 CPU 型号、RAM 与 wall-clock 时间，主机级性能复现信息不完整。
- 20,000 次 bootstrap 降低 Monte Carlo 误差，但独立单位仍只有 10 个；区间不应被解释为对所有 UAV-MEC 场景的总体推断。

## 9. 结论

本文完成了一个真实 Java 事件驱动仿真边界上的 UAV-MEC 混合动作强化学习研究，并通过独立训练种子、配对环境种子、20,000 次 bootstrap 和全链路哈希审计构建可复现证据。正式结果没有证明学习策略优于简单启发式：当前农村留出配置中，本地执行是强基线，四种学习策略在奖励、截止成功率、时延和能耗的配对比较中均处于劣势。

负结果并不否定混合动作学习在 UAV-MEC 中的潜力，而是限定了下一阶段需要回答的问题：构造真正需要卸载与移动的预注册压力场景；用同算法零移动消融识别移动价值；降低 mask 和边界约束错误；记录约束类型与 UAV 轨迹；在不泄漏留出数据的前提下提高训练稳定性。本文的主要价值是提供一个能够拒绝不受证据支持的优越性主张的实验系统。

## 10. 可复现材料

- 正式协议：`experiments/formal_protocol_1_2_10seed.json`
- 正式实验 ID：`formal_protocol_1_2_10seed_v2`
- 正式实验源码提交：`f4ef625ac2a438de724cad12760f187ea87560dc`
- 正式配置 SHA-256：`403688ade89c5b6abf13394af652892c790bd3e7289aed37ef58131720c481be`
- 原始与汇总目录：`results/formal/protocol_1_2_10seed_v2`
- 审计汇总：`summary/formal_summary.json`
- 汇总表：`summary/formal_results.md` 与 `summary/heldout_results.csv`
- 图表清单：`summary/figures/figure_manifest.json`

复算命令：

```bash
.venv/bin/python scripts/summarize_formal_experiments.py
.venv/bin/python scripts/plot_formal_results.py
```

## 参考文献

[1] Y. Zeng, Q. Wu, and R. Zhang, “Accessing From the Sky: A Tutorial on UAV Communications for 5G and Beyond,” *Proceedings of the IEEE*, vol. 107, no. 12, pp. 2327–2375, 2019. https://doi.org/10.1109/JPROC.2019.2952892

[2] Q. Hu, Y. Cai, G. Yu, Z. Qin, M. Zhao, and G. Y. Li, “Joint Offloading and Trajectory Design for UAV-Enabled Mobile Edge Computing Systems,” *IEEE Internet of Things Journal*, vol. 6, no. 2, pp. 1879–1892, 2019. https://doi.org/10.1109/JIOT.2018.2878876

[3] F. Zhou, Y. Wu, H. Sun, and Z. Chu, “UAV-Enabled Mobile Edge Computing: Offloading Optimization and Trajectory Design,” in *2018 IEEE International Conference on Communications (ICC)*, 2018. https://doi.org/10.1109/ICC.2018.8422277

[4] C. Sonmez, A. Ozgovde, and C. Ersoy, “EdgeCloudSim: An Environment for Performance Evaluation of Edge Computing Systems,” *Transactions on Emerging Telecommunications Technologies*, vol. 29, no. 11, e3493, 2018. https://doi.org/10.1002/ett.3493

[5] A. Al-Hourani, S. Kandeepan, and S. Lardner, “Optimal LAP Altitude for Maximum Coverage,” *IEEE Wireless Communications Letters*, vol. 3, no. 6, pp. 569–572, 2014. https://doi.org/10.1109/LWC.2014.2342736

[6] R. S. Sutton, D. Precup, and S. Singh, “Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning,” *Artificial Intelligence*, vol. 112, no. 1–2, pp. 181–211, 1999. https://doi.org/10.1016/S0004-3702(99)00052-1

[7] M. Hausknecht and P. Stone, “Deep Reinforcement Learning in Parameterized Action Space,” in *International Conference on Learning Representations (ICLR)*, 2016. https://arxiv.org/abs/1511.04143

[8] V. Mnih et al., “Human-Level Control through Deep Reinforcement Learning,” *Nature*, vol. 518, pp. 529–533, 2015. https://doi.org/10.1038/nature14236

[9] T. P. Lillicrap et al., “Continuous Control with Deep Reinforcement Learning,” in *International Conference on Learning Representations (ICLR)*, 2016. https://arxiv.org/abs/1509.02971

[10] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” 2017. https://arxiv.org/abs/1707.06347

[11] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” in *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, pp. 1587–1596, 2018. https://proceedings.mlr.press/v80/fujimoto18a.html

[12] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, “Deep Reinforcement Learning That Matters,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018. https://doi.org/10.1609/aaai.v32i1.11694

[13] R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. G. Bellemare, “Deep Reinforcement Learning at the Edge of the Statistical Precipice,” in *Advances in Neural Information Processing Systems 34*, 2021. https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
