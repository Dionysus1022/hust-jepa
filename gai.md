# 模型改进思路总结

## 背景

原始 VLA-JEPA 的核心思想是将 Qwen3-VL、动作预测头和 V-JEPA2 latent world model 结合起来。模型输入当前观测图像和语言指令，在 prompt 中插入特殊 token，并利用这些 token 的 hidden states 完成不同任务。

原始代码中主要有两类特殊 token：

- `<|action_i|>`：并不是离散化后的真实机器人动作，而是用于建模视觉动态的 latent action / dynamics token。它参与 V-JEPA predictor，条件化未来视觉 latent 的预测。
- `<|embodied_action|>`：用于动作头，作为机器人 embodied policy 的条件 token，最终预测连续动作序列。

原始 world model 训练流程可以概括为：

```text
过去观测图像 + 语言指令 + <|action_i|>
        ↓
Qwen3-VL hidden states
        ↓
取出 <|action_i|> hidden states
        ↓
past V-JEPA latent + action token hidden states
        ↓
V-JEPA predictor
        ↓
预测 future V-JEPA latent
        ↓
与真实未来视频的 V-JEPA latent 对齐
```

这里的问题是，模型虽然通过 `<|action_i|>` 学到了与未来动态相关的条件表示，但没有一个专门的 token 显式承载“未来状态表征”。因此我们希望引入新的 future token，使模型在语言-视觉上下文和动态 token 的条件下，显式预测未来 latent representation。

## 改进核心

我们新增一类特殊 token：

```text
<|future_i|>
```

其中 `<|future_i|>` 表示未来第 `i` 个时间步的 latent future representation。它的职责与 `<|action_i|>` 明确区分：

- `<|action_i|>`：表示从当前/过去到未来的动态条件，继续参与 V-JEPA predictor。
- `<|future_i|>`：表示模型自己预测出的未来状态表征，不参与 V-JEPA predictor，而是直接对齐真实未来 V-JEPA latent。
- `<|embodied_action|>`：保持原有职责，继续服务动作头，预测连续机器人动作。

因此新的 token 顺序设计为：

```text
<|action_i|><|future_{i+1}|><|embodied_action|>
```

更具体地，对多个未来时间步，prompt 中的动态部分按照严格因果顺序交错排列：

```text
<|action_0|><|future_1|>
<|action_1|><|future_2|>
<|action_2|><|future_3|>
...
<|embodied_action|>
```

这样设计的含义是：

```text
先根据过去观测和语言形成 action/dynamics token，
再让 future token 在已有上下文和 action token 条件下预测下一时刻未来表征，
最后 embodied action token 负责动作生成。
```

由于 Qwen 是 causal language model，后面的 `<|future_{i+1}|>` 可以 attend 到前面的 `<|action_i|>`，但 `<|action_i|>` 不会看到后面的 future token。这保证了 token 顺序本身符合“先动态条件，后未来预测”的因果结构。

## 防止信息泄露

这个改进必须避免未来信息泄露。我们的约束是：

```text
真实未来视频 / 真实未来 V-JEPA latent 只能作为 loss target，
不能作为 Qwen 输入，也不能作为 future token 的条件输入。
```

训练时，Qwen 只接收：

```text
当前/过去观测图像
语言指令
特殊 token 序列
```

V-JEPA2 encoder 可以编码完整视频，但其中未来部分只用于构造监督信号：

```text
gt_future_latent = V-JEPA2(future frames)
```

这个 `gt_future_latent` 只参与 loss 计算，不回流到 prompt 或 Qwen 输入中。因此 `<|future_i|>` 学到的是在过去观测、语言和 `<|action_{i-1}|>` 条件下对未来 latent 的预测，而不是读取真实未来得到的表示。

## 新的训练目标

改进后，模型训练包含三个主要目标。

### 1. 动作预测 loss

保持原有动作头训练方式不变：

```text
<|embodied_action|> hidden states
        ↓
flow-matching / DiT action head
        ↓
预测连续动作序列
        ↓
与真实机器人动作对齐
```

这个 loss 负责保证模型仍然能够输出可执行的机器人动作。

### 2. 原始 world model loss

保持原有 V-JEPA predictor 路径：

```text
past V-JEPA latent + <|action_i|> hidden states
        ↓
V-JEPA predictor
        ↓
predicted future patch latent
        ↓
与真实 future V-JEPA patch latent 对齐
```

这里 `<|action_i|>` 仍然承担 latent dynamics token 的作用，用于条件化细粒度未来视觉 latent 预测。

### 3. 新增 future token alignment loss

新增的关键监督是直接训练 `<|future_i|>` 本身：

```text
<|future_i|> hidden state
        ↓
future projector
        ↓
predicted future latent summary
        ↓
与真实 future V-JEPA latent summary 对齐
```

由于 Qwen hidden size 与 V-JEPA latent dim 不同，因此引入一个 `future_projector`，将 `<|future_i|>` 的 hidden state 投影到 V-JEPA latent 空间。

为了让训练更稳定，第一版采用 frame-level latent summary 对齐：

```text
gt_future_frame_latent = mean_pool(gt_future_patch_latents per frame)
```

也就是说，不要求单个 future token 直接重建所有 patch-level latent，而是让它对齐对应未来帧的 V-JEPA latent 摘要。这样 `<|future_i|>` 的语义更加清晰：

```text
<|future_i|> = 对未来第 i 帧视觉状态的 latent summary prediction
```

## 总体结构

改进后的整体训练结构可以写成：

```text
输入:
  当前/过去图像
  语言指令
  <|action_0|><|future_1|> ... <|action_i|><|future_{i+1}|> <|embodied_action|>

Qwen3-VL:
  输出所有 token 的 hidden states

分支 1: 动作预测
  <|embodied_action|> hidden states
      -> action head
      -> action_loss

分支 2: V-JEPA world model
  past V-JEPA latent + <|action_i|> hidden states
      -> V-JEPA predictor
      -> wm_loss

分支 3: future token 对齐
  <|future_i|> hidden states
      -> future_projector
      -> future_token_loss
```

最终 loss 为：

```text
total_loss = action_loss + wm_loss + lambda_future * future_token_loss
```

其中 `lambda_future` 用于控制 future token 对齐 loss 的权重。

## 改进意义

这个改进的核心意义是让模型内部显式形成未来表征。

原始 VLA-JEPA 中，未来动态主要通过 `<|action_i|>` 和 V-JEPA predictor 的输出体现；而新增 `<|future_i|>` 后，模型被要求在语言-视觉上下文中直接生成一个可监督的 future representation token。

这种设计带来几个好处：

- 明确区分“动态条件 token”和“未来状态 token”。
- 让未来表征成为 Qwen token space 中可直接监督的对象。
- 保持原有动作预测路径不变，降低对策略输出接口的影响。
- 真实未来只作为监督目标，不进入模型输入，避免信息泄露。
- `<|future_i|>` 可以作为后续扩展的接口，例如用于未来状态可视化、辅助动作预测、多步规划或额外 consistency loss。

简而言之，这个改进可以概括为：

```text
在 VLA-JEPA 中新增显式 future representation token，
让模型不仅通过 world model 预测未来 latent，
还在 Qwen hidden space 中直接学习一个可对齐 V-JEPA future latent 的未来状态表征。
```

