# π0.5 Knowledge Insulation Fine-tuning 实施方案

## 1. 目标和开关语义

在当前 JAX π0.5 上增加 Knowledge Insulation（KI）训练，不替换 backbone，不在推理时生成 FAST token。

总损失：

    L_total = lambda_FAST * L_FAST + lambda_FM * L_FM

| knowledge_insulation | 模式 | FAST CE | FM | FM → VLM 梯度 |
|---|---|---:|---:|---:|
| False | 现有 baseline | 关 | 开 | 保留 |
| True | 完整 KI | 开 | 开 | detached KV 阻断 |

不增加“CE + FM 但不隔离”模式。False 必须回到原 FM-only 路径，包括原有 loss shape、RTC 和 inference 行为。

期望梯度拓扑：

    FAST CE
      ├─→ image encoder                         non-zero grad
      └─→ VLM Transformer / VLM LoRA            trainable part non-zero grad

    FM
      ├─→ detached context KV ─→ VLM/image      zero grad
      └─→ Action Expert / AE LoRA / projections non-zero grad

## 2. 实施前测试和结论

以下来自当前 /app 代码和实际运行，不是对未实现代码的推测。

### 2.1 基线测试

执行：

    uv run pytest -q src/openpi/models/pi0_test.py src/openpi/transforms_test.py
    # 14 passed

    uv run pytest -q \
      src/openpi/models/rtc_guidance_test.py \
      examples/ur10e/action_adapter_test.py
    # 17 passed

Policy 扩展回归先完成 19 passed，但发现一个与 KI 无关的既有差异：policy_test.py 仍期望 train_time_rtc_max_delay == 10，而用户后来根据实验已将当前配置调整为 4，因此这是 stale test，不是待讨论的配置问题，也不纳入 KI 修改。后续用例卡在模型下载，约 5 分钟后中止。KI 验收时将该 stale failure 单独标记，不把它计为 KI regression，也不在 KI 实现改动中顺手修正。

### 2.2 KV split 前向和 stop-gradient

用两个小型 Gemma expert 测试 joint forward 与 prefix-KV/suffix split forward，并覆盖 batch 内不同 prefix padding：

| dtype | MAE | max abs |
|---|---:|---:|
| float32 | 1.42e-7 | 5.96e-7 |
| bfloat16 | 0.0 | 0.0 |

split forward 的输入梯度：

| 模式 | prefix grad norm | suffix grad norm |
|---|---:|---:|
| 未 detach | 3.79e-8 | 1.18e-7 |
| detach 整棵 KV | 0.0 | 1.18e-7 |

结论：机制可行，必须执行：

    kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

不能只 detach 最终 hidden，也不能在 suffix forward 重新传 prefix token。正式实现后仍须在真实 Pi0 上重做等价测试。

### 2.3 UR 数据表示和 transform 顺序

当前顺序：

    repack → URInputs → AbsoluteTCPActionsToRelative
    → Normalize → model transforms

对 pick_v4_merge_crop_vid 的 100 个均匀抽样：

    dataset size             19908
    normalized state         (10,)
    normalized actions       (20, 10)
    Pad 后 state             (32,)
    Pad 后 actions           (20, 32)
    padding 区非零元素       0

结论：KI FAST tokenizer 必须在 Normalize 后、PadStatesAndActions(32) 前运行，编码 normalized 物理 10D action；FM 仍接收 Pad 后 32D action。

### 2.4 FAST 长度、mask 和 dtype

同一组 100 个真实样本，max_len=256：

    token length mean/p90/p95/p99/max  89.10/107.20/112/118.11/129
    truncated                          0 / 100
    loss-mask count mean/min/max       32.89/15/73
    zero loss mask                     0 / 100

这是初步抽样，不代替最终 5000 样本检查。

现有 FASTTokenizer.tokenize() 返回 tokens:int64 和 ar_mask:int64。JAX loader 当前会隐式转为 int32，但 KI transform 必须显式产生：

    ki_tokens      int32
    ki_token_mask  bool
    ki_ar_mask     bool
    ki_loss_mask   bool

### 2.5 loss mask 的准确语义

现有 FAST mask 监督整个 postfix：

    Action: <FAST action tokens> | EOS

零 action 样例中，Action 标签 3 tokens + FAST action 2 tokens + 结尾 2 tokens = 7 个 masked targets。新实现保持 Pi0FAST 的整个 postfix CE 语义。指标命名为 fast_target_token_count，不误称为纯 action-token count。

### 2.6 实际 trainable 参数组

当前 UR 双 LoRA config 的 abstract parameter tree：

| 参数组 | trainable leaves | frozen leaves |
|---|---:|---:|
| image encoder | 23 | 0 |
| VLM LLM | 10 | 9 |
| action expert | 10 | 11 |
| action input projection | 2 | 0 |
| time MLP | 4 | 0 |
| action output projection | 2 | 0 |

当前 freeze filter 会训练 image encoder。本任务不改 freeze filter，因此 FAST-only 时 image encoder 应有梯度，FM-only insulated 时必须为零。

### 2.7 tied decoder 和 shared-transform 风险

gemma.Embedder.decode() 已实现 hidden @ embedding.T，但 gemma.Module 未公开 decode。Pi0 使用 nnx_bridge.ToNNX，不能复制 pi0_fast.py 对另一套 gemma_fast.py API 的调用。必须增加 Module-level tied decode，并测试 ToNNX method dispatch。

训练 loader 和 Policy 当前共用 data_config.model_transforms.inputs。直接加入 KI transform 会让 KI checkpoint 推理时也构造 FAST tokenizer；实测还触发了 Hugging Face remote-code 下载。因此 KI tokenization 必须是 training-only，Policy 不得实例化它。

## 3. 配置设计

在 Pi0Config 增加：

    knowledge_insulation: bool = False
    ki_max_token_len: int = 256
    ki_fast_tokenizer_path: str = "physical-intelligence/fast"
    ki_fast_tokenizer_revision: str | None = None
    ki_fast_loss_weight: float = 1.0
    ki_flow_loss_weight: float = 1.0

不增加独立 detach 开关。KI 开启时校验 pi05=True、ki_max_token_len>=2、两个 weight 都是有限正数。weight 只是损失系数，不作为额外模式开关。

新增 pi05_ur10e_lora_ki_finetune，保持 baseline 的 repo、action dim/horizon、normalization、LoRA variants、weight loader 和 policy metadata。初始 batch_size=8，真实 single-step 显存测试后再调整。

FAST tokenizer 使用 trust_remote_code=True，正式训练应锁定 revision。本机实测 snapshot 为 ec4d7aa71691cac0b8bed6942be45684db2110f4；写入最终配置前确认它是期望版本。

## 4. 数据管线

### 4.1 Training-only construction and application

不能在 DataConfig 或 LeRobotURDataConfig.create() 的返回值中存放已经实例化的 FAST tokenizer。否则 create_trained_policy() 即使不应用该 transform，仍会在 train_config.data.create(...) 时触发 FASTTokenizer() 和 Hugging Face remote-code 加载。

在 DataConfigFactory 增加默认返回空 Group 的 factory method：

    def create_training_model_transforms(
        self,
        model_config: _model.BaseModelConfig,
    ) -> _transforms.Group:
        return _transforms.Group()

LeRobotURDataConfig override 该方法。只有当 model_config 是启用 KI 的 Pi0Config 时，才在方法体内实例化 FASTTokenizer 和 TokenizeKnowledgeInsulation。LeRobotURDataConfig.create() 本身继续返回不包含 KI tokenizer 实例的普通 DataConfig。

只有训练入口 create_data_loader(config, ...) 调用：

    training_model_transforms = (
        config.data.create_training_model_transforms(config.model)
    )

再将该 Group 显式传给 create_torch_data_loader() 和 transform_dataset()。相关 helper 增加空 Group/default，以兼容现有直接调用。

训练 loader 顺序：

    repack → robot transforms → Normalize
    → training_model_transforms          # KI tokenizer
    → shared model_transforms            # TokenizePrompt, Pad

create_trained_policy() 和 compute_norm_stats.py 永远不调用 create_training_model_transforms()。这同时保证 training-only materialization 和 training-only application：Policy 既不构造也不执行 FAST tokenizer，norm stats 定义不变。

### 4.2 TokenizeKnowledgeInsulation

要求：

1. 读取 prompt、normalized 10D state 和 clean normalized 20×10 actions。
2. 复用 FASTTokenizer 的 vocabulary mapping 和整个 postfix mask。
3. 输出四个 ki_* 字段并显式 cast dtype。
4. 不 pop prompt，不修改 state/actions。
5. 后续继续执行 TokenizePrompt(discrete_state_input=True) 和 PadStatesAndActions(32)。

KI training transform 当前运行在 shared model transforms 中的 InjectDefaultPrompt 之前。当前 UR KI config 使用 prompt_from_task=True，prompt 已在 dataset 阶段生成，因此顺序成立；但该 transform 不能依赖后续 InjectDefaultPrompt 补 prompt。必须显式检查：

    if "prompt" not in data:
        raise ValueError(
            "KI training transform requires prompt before model transforms"
        )

因此当前 KI UR config 的数据契约是：进入 training_model_transforms 前 prompt 必须已经存在。未来若改用 default_prompt，必须把默认 prompt 的注入移到 KI transform 之前，或在 training-transform factory 中显式先加入 InjectDefaultPrompt；不能依赖 shared transforms 中较晚执行的实例。

现有 tokenizer 截断时只记录 warning，非空 mask 不能证明 postfix 完整。增加非破坏性 metadata API，同时保留原 tokenize() 四元返回：

    untruncated_length
    was_truncated
    prefix_length
    postfix_length
    fast_action_code_count

## 5. Observation 和 specs

在 Observation、from_dict()、to_dict() 和 preprocess_observation() 增加并保留：

    ki_tokens: Int[..., L] | None = None
    ki_token_mask: Bool[..., L] | None = None
    ki_ar_mask: Bool[..., L] | None = None
    ki_loss_mask: Bool[..., L] | None = None

KI inputs_spec 为 [B, ki_max_token_len]，dtype 依次为 int32/bool/bool/bool。KI 关闭时字段保持 None。四个字段必须成组出现。

### 5.1 每个 training step 只预处理 observation 一次

KI 的 FAST 和 FM helper 都只接收已经 preprocess 的 Observation，helper 内不得再次调用 preprocess_observation()。compute_loss_with_aux() 是唯一的预处理入口：

    if not self.knowledge_insulation:
        # 直接走原 compute_loss()，保留原 RNG 和数值路径。
        flow_loss = self.compute_loss(rng, observation, actions, train=train)
        ...

    if self.train_time_rtc_max_delay > 0:
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
    else:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        delay_rng = None

    processed_observation = _model.preprocess_observation(
        preprocess_rng,
        observation,
        train=train,
    )

    fast = self._compute_fast_loss(processed_observation)
    flow = self._compute_flow_loss_preprocessed(
        noise_rng,
        time_rng,
        delay_rng,
        processed_observation,
        actions,
    )

因此两个分支读取完全相同的 resize/augmentation 结果：

    raw observation
          ↓
    preprocess/augmentation once
          ├─→ FAST branch
          └─→ FM branch

这样既避免 FAST/FM 使用不同随机增强，也保留当前 RTC 开启和关闭时分别 split 4/3 个 RNG 的约定。non-KI 不经过这套新拆分，直接保留当前 compute_loss() 路径。

## 6. FAST branch

### 6.1 Tied decode

在 gemma.Module 暴露无新参数的方法：

    def decode(self, hidden):
        return self.embedder.decode(hidden)

用与现有 method="embed" 相同的 ToNNX dispatch。测试 decode 等于 embedding transpose matmul，且 parameter tree 不变。

### 6.2 Embedding、attention 和 CE

embed_ki_inputs() 先构造所有 image embeddings + 固定长度 ki_tokens embedding，以及对应的完整 input mask 和 AR mask。image + FAST prefix 是 bidirectional block，FAST postfix 是 autoregressive block：

    full_embeddings, full_input_mask, full_ar_mask = self.embed_ki_inputs(
        observation
    )
    full_attn_mask = make_attn_mask(full_input_mask, full_ar_mask)
    full_positions = jnp.cumsum(full_input_mask, axis=1) - 1

next-token forward 必须同步裁剪 embedding、attention mask 和 positions：

    model_embeddings = full_embeddings[:, :-1]
    model_attn_mask = full_attn_mask[:, :-1, :-1]
    model_positions = full_positions[:, :-1]

    (vlm_out, _), _ = self.PaliGemma.llm(
        [model_embeddings, None],
        mask=model_attn_mask,
        positions=model_positions,
    )

这里的 :-1 始终删除固定物理序列的最后一个 slot，绝不能按每个样本的最后一个 valid token 动态裁剪。batch 内不同有效长度继续完全由 ki_token_mask 和 ki_loss_mask 处理。

next-token target 和 hidden 使用与 Pi0FAST 相同的尾部选择语义：

    targets = observation.ki_tokens[:, 1:]
    loss_mask = observation.ki_loss_mask[:, 1:]
    token_hidden = vlm_out[:, -targets.shape[1]:]

经过 model_embeddings = full_embeddings[:, :-1] 后，最后 L-1 个 hidden 固定对应 T0 ... T(L-2)，恰好预测 T1 ... T(L-1)。不要手算 image_token_count；尾部选择可避免未来摄像头数量、图像 token 数或图像排列改变时产生静默错位。

必须断言：

    model_embeddings.shape[1] == full_embeddings.shape[1] - 1
    model_attn_mask.shape[-2:] == (
        model_embeddings.shape[1],
        model_embeddings.shape[1],
    )
    model_positions.shape[1] == model_embeddings.shape[1]
    token_hidden.shape[:2] == targets.shape
    targets.shape == loss_mask.shape

使用 integer-label CE，不构造 one-hot。第一版使用固定序列 logits + mask；若显存不足，后续使用固定容量 packed gather，不在 JIT 内使用 data-dependent dynamic indexing。

_compute_fast_loss() 返回 per-example CE [B]、correct-target count 和 valid-target count。accuracy 用全局 correct/count。

## 7. Detached FM branch

FAST target 在 dataset transform 阶段由 clean normalized action 生成。RTC corruption 和 flow noise 只在 _compute_flow_loss() 中作用于 padded 32D action：

    clean normalized 10D actions → FAST targets
    clean normalized padded 32D actions → RTC/noise → x_t/FM

KI prefix prefill：

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=prefix_positions,
    )
    kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

suffix 复用 sample_actions() 的 cache semantics：

    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_region = repeat(prefix_mask, "b p -> b s p", s=suffix_len)
    full_attn_mask = concatenate([prefix_region, suffix_attn_mask], axis=-1)
    suffix_positions = (
        sum(prefix_mask, axis=-1)[:, None]
        + cumsum(suffix_mask, axis=-1) - 1
    )
    (_, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens],
        kv_cache=kv_cache,
        mask=full_attn_mask,
        positions=suffix_positions,
        adarms_cond=[None, adarms_cond],
    )

mask 的 key 长度必须是 prefix_physical_len + suffix_len；position offset 使用每个样本的 valid-prefix count。

knowledge_insulation=False 时直接运行现有 joint forward，不 prefill cache、不构造 FAST loss、不改变 [B,H] loss shape。

## 8. Loss contract 和训练更新

保留现有 compute_loss() 的 non-KI 行为。为 Pi0 增加：

    compute_loss_with_aux(...) -> (total_scalar, aux)

KI 内部先独立 reduce。FM helper 继续返回当前已经经过 normalize_postfix_loss() 处理的 [B,H] loss；不要为 KI 重新实现 RTC 的 sum/count normalization：

    flow_loss = self._compute_flow_loss_preprocessed(...)
    # RTC 开启时 flow_loss 已由现有 normalize_postfix_loss() 缩放。
    flow_loss_scalar = jnp.mean(flow_loss)
    fast_loss_per_example = (
        sum(masked_ce, axis=-1)
        / maximum(sum(loss_mask, axis=-1), 1)
    )
    fast_loss_scalar = jnp.mean(fast_loss_per_example)
    total = (
        ki_fast_loss_weight * fast_loss_scalar
        + ki_flow_loss_weight * flow_loss_scalar
    )

不将 [B] FAST loss直接与 [B,H] flow loss相加，避免尤其在 batch_size == action_horizon 时形状合法但语义错误的广播。

FAST loss 保持现有 Pi0FAST.compute_loss() 的“每个样本先按自己的有效 postfix target 数归一化，再对 batch 取 mean”语义。它与日志 accuracy 的聚合方式不同：loss 保持每个训练样本等权，accuracy 则按所有有效 target token 的 correct/count 统计。

train.py 在 Python 静态分支中对 Pi0 实例调用 compute_loss_with_aux()，对其他模型保留 mean(compute_loss())；不使用运行时 JAX 条件选择模型接口。loss_fn 统一返回 (loss, aux)，再通过 has_aux=True 只执行一次 gradient、tx.update、apply_updates 和 nnx.update。non-KI Pi0 wrapper 只调用原 compute_loss() 并取 mean，不改变 RNG、FM 或 RTC postfix normalization。

每个 step 的 aux 返回 fast_correct_count 和 fast_target_token_count，不只返回已经相除的 ratio。当前 train.py 会把多个 step 的 info stack 后逐项取 mean；FAST accuracy 必须特殊聚合：

    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *infos)
    fast_token_accuracy = (
        jnp.sum(stacked["fast_correct_count"])
        / jnp.maximum(jnp.sum(stacked["fast_target_token_count"]), 1)
    )
    averaged = jax.tree.map(jnp.mean, stacked)
    averaged["fast_token_accuracy"] = fast_token_accuracy

不能先计算每个 batch 的 accuracy 再对 batch accuracy 做无权平均，因为不同 batch 的有效 FAST target 数可能不同。

最终日志指标：loss、flow_loss、fast_ce_loss、fast_token_accuracy、fast_correct_count、fast_target_token_count、grad_norm、param_norm。分组 grad norm 放在独立脚本，不增加每步训练开销。

## 9. 验证脚本

### 9.1 scripts/check_ki_data.py

参数：--config-name、--num-samples、--seed、--num-examples。使用与训练相同的 UR transform、norm stats 和 training-only KI transform。

输出：

    sample count and pre-pad dimensions
    token length mean/p90/p95/p99/max
    FAST action-code count mean/min/max
    full postfix target count mean/min/max
    truncation count/ratio
    zero loss masks
    non-finite inputs
    loss mask outside token mask
    missing postfix terminator/EOS
    decoded prompt/state/action-code/mask-boundary examples

开发阶段至少运行 5000 样本；正式训练启动前扫描完整数据集。当前 pick_v4_merge_crop_vid 实测共有 19,908 个 sample positions，因此最终推荐命令不设置抽样上限，或显式使用 --num-samples 19908。要求 unexpected truncation、empty/invalid mask、missing terminator/EOS 和 non-finite input 均为零。报告必须区分“5000 抽样检查”和“full-dataset 检查”，不能用前者替代正式训练前的全量结果。

### 9.2 scripts/check_ki_gradients.py

按实际 trainable tree 分组：

    image_encoder
    vlm_lora
    action_expert_lora
    action_in_proj
    time_mlp
    action_out_proj
    other_trainable

每组输出 trainable/gradient leaf count、finite status、gradient norm、parameter norm 和 relative norm。不把 frozen 参数缺少 grad 当成 KI 成功证据。

FAST-only 预期 image encoder/VLM LoRA 非零；AE LoRA 和 action/time projections 严格零。FM-only insulated 预期相反。显式识别 NNX/JAX 对 unused leaves 的零结构，不只依赖经验 tolerance。

other_trainable 是强制警报项，不只是普通日志字段。脚本必须打印其中每一个完整 parameter path，并满足以下二者之一才能 PASS：

    other_trainable is empty

或：

    每个 path 都通过显式规则重新归入已知参数组，
    且对应分支的零/非零梯度期望已定义并验证。

不允许在 other_trainable 非空且未分类时继续给出总 PASS，避免漏分的 VLM 参数绕过 FM→VLM 零梯度检查。参数分组测试还应断言所有 trainable leaves 被且仅被一个组覆盖，没有遗漏或重复归类。

Insulation 的严格判据是 value_and_grad 直接返回的原始 gradient tree。不能根据 optimizer step 后参数是否 bitwise unchanged 推断隔离是否成功：当前使用 AdamW，decoupled weight decay 即使在梯度为零时也可能造成极小参数变化。参数更新检查只用于证明 joint KI 的目标参数确实能训练，不用于证明零梯度拓扑。

### 9.3 真实 DataLoader 多进程 smoke test

training-transform factory 解决了 Policy 侧不构造 tokenizer 的问题，但训练 transform 会持有 Hugging Face AutoProcessor，而当前 Torch DataLoader 使用 spawn worker。Phase 1 必须用真实 KI config 分两档各读取至少 3 个 batch：

    # 先隔离验证 transform/data contract
    HF_LEROBOT_HOME=/app/data/lerobot \
    uv run scripts/check_ki_dataloader.py \
      --config-name pi05_ur10e_lora_ki_finetune \
      --num-workers 0 --num-batches 3

    # 再验证正式训练使用的 spawn 多进程路径
    HF_LEROBOT_HOME=/app/data/lerobot \
    uv run scripts/check_ki_dataloader.py \
      --config-name pi05_ur10e_lora_ki_finetune \
      --num-workers 12 --num-batches 3

两档都要检查：

    Observation 四个 ki_* 字段 shape/dtype 正确
    action batch 是 [B, 20, 32]
    没有 pickle/serialization error
    没有 worker crash、hang 或异常重启
    没有每个 batch 重复下载/加载 remote code
    worker RSS 和主进程/GPU 内存处于可接受范围
    连续 batch 能正常退出或继续迭代

记录启动耗时、前三个 batch 耗时、主进程与 worker 峰值 RSS。若 AutoProcessor 不能可靠跨 spawn pickle，不在首次正式训练时临时绕过；应回到 factory/worker 初始化边界，选择明确的每-worker 一次初始化或可序列化包装，并重新验证不重复下载及内存开销。

## 10. 必需测试

Data：

- KI 四字段、dtype、shape 正确，prompt 保留。
- prompt 缺失时 KI transform 立即抛出明确错误；测试证明不能依赖后续 InjectDefaultPrompt。
- FAST 看 10D clean normalized action，FM 看 32D padded action。
- mask 与现有 FAST 整个 postfix 语义一致。
- truncation metadata 能区分刚好等于 max 和被截断。
- 调用 LeRobotURDataConfig.create() 和 create_trained_policy() 时 monkeypatch FASTTokenizer 构造器为“调用即失败”，证明二者不会 materialize KI tokenizer。
- 只有 create_data_loader() 调用 training-transform factory，并且恰好构造一次 KI tokenizer；non-KI loader 构造零次。
- 真实 KI DataLoader 分别以 num_workers=0 和 num_workers=12 连续读取至少 3 batches，无 pickle、worker、重复 remote-code 加载或异常内存问题。

FAST：

- tied decode 等于 embedding transpose matmul。
- 无新 vocabulary-head parameter。
- embedding[:, :-1]、attention_mask[:, :-1, :-1] 和 positions[:, :-1] 使用同一个固定物理末位 shift。
- 使用 vlm_out 尾部 L-1 个 hidden 与 targets[:, 1:] 对齐，不手算 image token offset；覆盖摄像头/image-token 数变化和 batch 内不同 valid token 长度。
- CE finite，并与 Pi0FAST 的 per-example valid-token normalization 一致。
- 两个不同 target count 的 step 聚合后，accuracy 等于 sum(correct)/sum(target)，而不是 batch ratio 的平均。

FM：

- 相同 params/input/noise/time 下比较 joint/split suffix_out、v_t 和 per-token loss。
- 覆盖不同 prompt padding、多图像、float32/bf16、scalar time 和 RTC per-token time。
- 确认 suffix position 不从 0 重启，padding KV 被 mask。
- 初始 float32 标准为 max_abs <= 1e-5；bf16 根据真实模型实测设 atol/rtol，并报告 MAE/max-abs。

Gradient/RTC：

- monkeypatch/计数 preprocess_observation()，证明 KI step 只调用一次，且 FAST/FM 收到同一个 processed Observation。
- 固定 RNG 验证 RTC 开/关分别维持当前 4-key/3-key split 规则。
- FAST-only 和 insulated FM-only 的完整参数组梯度拓扑。
- 所有 trainable parameter path 被且仅被一个已知组覆盖；未分类的 other_trainable 使检查失败并打印完整路径。
- joint KI 一次 backward 后 VLM 和 AE 目标参数都更新。
- frozen base weights 不更新。
- insulation 以 optimizer 前的原始 gradient tree 为准，不以 AdamW step 后的参数差值为准。
- RTC delay=0 和 delay>0 均可 forward/backward。
- FAST 始终监督整个 clean chunk，RTC prefix 只影响 FM corruption/loss。

Non-KI/inference/checkpoint：

- 固定 rng/input/params，比较 non-KI 修改前后 loss shape、loss 和 gradients。
- sample_actions()、Policy/server 和 RTC inference 不变。
- inference 不实例化 FAST tokenizer。
- KI/non-KI Pi0 parameter tree 完全一致。
- pi05_base 可直接初始化 KI config，无新 checkpoint 参数。

## 11. 实施阶段

1. Phase 0 — baseline：记录已通过项；预热并锁定 tokenizer/model cache。RTC stale test 不属于 KI 工作范围，不修改配置或旧测试。
2. Phase 1 — data：配置、training-transform factory、Observation/spec、KI transform、tokenizer metadata、check_ki_data.py；先通过 5000 样本检查，再完成 num_workers=0/12 的真实 DataLoader smoke test；正式训练前扫描完整 19,908 样本。
3. Phase 2 — FAST：Gemma tied decode、embed_ki_inputs()、完整序列统一 shift、_compute_fast_loss()、finite/alignment/mask 测试和 FAST-only gradient check。
4. Phase 3 — detached FM：_prefill_detached_context()、split _compute_flow_loss()、真实模型 joint/split 等价和 FM-only gradient check。
5. Phase 4 — joint：compute_loss_with_aux()、has_aux=True、metrics、KI UR config、真实 batch 单步 optimizer/显存测试。
6. Phase 5 — compatibility：KI + train-time RTC、non-KI 数值回归、inference/Policy/server/RTC、checkpoint、Ruff 和完整相关 Pytest。

## 12. 预计修改文件

    src/openpi/models/pi0_config.py
    src/openpi/models/model.py
    src/openpi/models/gemma.py
    src/openpi/models/pi0.py
    src/openpi/models/pi0_test.py
    src/openpi/models/tokenizer.py
    src/openpi/transforms.py
    src/openpi/transforms_test.py
    src/openpi/training/config.py
    src/openpi/training/data_loader.py
    src/openpi/training/data_loader_test.py
    src/openpi/policies/policy_test.py
    scripts/train.py
    scripts/check_ki_data.py
    scripts/check_ki_dataloader.py
    scripts/check_ki_gradients.py

policy_config.py 和 compute_norm_stats.py 原则上都不改。通过 data_loader_test.py / policy_test.py 回归证明：前者仅在训练 loader 构造 KI tokenizer，后者和 norm-stats 路径不 materialize training-only transforms。

## 13. 最终验收

执行：

    HF_LEROBOT_HOME=/app/data/lerobot \
    uv run scripts/check_ki_data.py \
      --config-name pi05_ur10e_lora_ki_finetune \
      --num-samples 5000

    # 正式训练前运行完整数据集；当前数据集为 19,908 samples。
    HF_LEROBOT_HOME=/app/data/lerobot \
    uv run scripts/check_ki_data.py \
      --config-name pi05_ur10e_lora_ki_finetune \
      --num-samples 19908

    HF_LEROBOT_HOME=/app/data/lerobot \
    uv run scripts/check_ki_dataloader.py \
      --config-name pi05_ur10e_lora_ki_finetune \
      --num-workers 0 --num-batches 3

    HF_LEROBOT_HOME=/app/data/lerobot \
    uv run scripts/check_ki_dataloader.py \
      --config-name pi05_ur10e_lora_ki_finetune \
      --num-workers 12 --num-batches 3

    HF_LEROBOT_HOME=/app/data/lerobot \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run scripts/check_ki_gradients.py \
      --config-name pi05_ur10e_lora_ki_finetune

    uv run pytest -q src/openpi/models/pi0_test.py
    uv run pytest -q src/openpi/transforms_test.py
    uv run pytest -q src/openpi/models/rtc_guidance_test.py
    # RTC delay=10 的旧断言已知过期，不属于 KI；其余 Policy 测试必须通过。
    uv run pytest -q src/openpi/policies/policy_test.py \
      -k "not test_ur_train_time_rtc_config_is_explicit_and_metadata_has_no_checkpoint_capability"
    uv run pytest -q examples/ur10e/action_adapter_test.py

交付报告必须包含：5000 抽样和 19,908 全量 token/truncation/mask 统计；num_workers=0/12 的 DataLoader smoke test 与内存/耗时；FAST-only 和 FM-only 分组 gradient norms；joint/split MAE/max-abs；single-step loss/optimizer；peak GPU memory；non-KI/inference/checkpoint 回归；实际 Ruff/Pytest 命令和结果。

只有以下条件同时成立才验收：

    False 与原 FM-only loss/gradient/sampling/RTC 一致
    FAST-only: image/VLM trainable params non-zero; AE/projections zero
    FM-only insulated: image/VLM trainable params zero; AE/projections non-zero
    split FM output ≈ joint FM output
    FAST targets 来自 clean normalized 10D GT actions
    FM 使用 padded 32D actions
    zero unexpected truncation and invalid/empty masks
    full 19,908-sample scan passes before formal training
    num_workers=0 and num_workers=12 real DataLoader smoke tests pass
    FAST/FM finite; one backward and one optimizer update
    inference 不实例化 FAST tokenizer
    sample_actions/policy server/RTC inference 不变
    base pi05 checkpoint 无需新参数
