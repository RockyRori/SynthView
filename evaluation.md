## 1. 模型结构

- 该模型是 **SinGAN** 的变体，采用 **MultiVanilla** 结构。
- 由多个 `Vanilla` 模块组成，每个模块都有 **BasicBlock**，其中包含：
    - 3x3 卷积（`Conv2d`）
    - 批归一化（`BatchNorm2d`）
    - LeakyReLU 激活函数（`LeakyReLU`）
- 这是一个 **多尺度生成网络**，每个 `Vanilla` 模块对应一个不同的尺度（`s0` 到 `s8`），类似于金字塔结构。
- 最终输出通过 `features_to_image` 转换到 RGB 空间，使用 `Tanh()` 作为最后的激活函数。

---

## 2. 超参数

### 生成器（Generator）

- `gen_model = "g_multivanilla"`
- `num_blocks = 5`（每个尺度有5层 BasicBlock）
- `min_features = 32, max_features = 32`（可能是设置了特征图的通道数）
- `batch_size = 16`
- `num_steps = 100`（生成时迭代步数）
- `lr = 0.0004`（学习率）
- `gen_betas = [0.5, 0.9]`（Adam 优化器的 beta 参数）
- `adversarial_weight = 1.0`（对抗损失的权重）
- `reconstruction_weight = 10.0`（重建损失的权重）

### 判别器（Discriminator）

- `dis_model = "d_vanilla"`
- `dis_betas = [0.5, 0.9]`
- `num_critic = 1`（训练判别器的步数）
- `penalty_weight = 0.1`（正则化项权重）

---

## 3. 训练参数

- 使用 **CUDA** 设备进行训练。
- **加载的模型权重**：
  ```
  .\results\2025-02-26_11-17-13\g_multivanilla.pt
  .\results\2025-02-26_11-17-13\amps.pt
  ```
- **评估结果存储路径**：
  ```
  ./results\2025-03-18_13-40-07
  ```
- `step_size = 2000, gamma = 0.1`（学习率衰减参数）

---

## 4. 参数量

- **生成器总参数量**: **1,479,966**（约 1.48M）。
- 这个参数量相对较小，适合小型图像生成任务，比如无条件图像生成或者小分辨率超分辨率任务。

---

