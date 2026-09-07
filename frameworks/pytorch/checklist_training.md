# PyTorch Training Workload Readiness Checklist

> Evaluates hardware accelerator readiness for **training workloads** with PyTorch.
> Covers forward + backward pass, autograd, AMP, distributed training, and optimizer support.
> For full PyTorch integration coverage, use `checklist.md`.

| Field | Value |
|-------|-------|
| **Backend** | _[FILL: backend name]_ |
| **Backend version** | _[FILL: release version/tag]_ |
| **Integration path** | _[FILL: PU1 or Fork]_ |
| **Dispatch key** | _[FILL: PrivateUse1 or custom]_ |
| **Evaluation date** | _[FILL: date]_ |
| **Evaluator** | _[FILL: human or agent]_ |
| **Source** | _[FILL: repo URL]_ |
| **Workload** | Training |

---

## Readiness Score & Summary

### Executive Summary

_[FILL: Write a concise summary covering:
- **Overall Training Readiness**: X%
- **Key strengths** for training workloads
- **Key gaps** blocking training use
- **Recommendations**]_

### Scoring Model

Every row has a max score of **2** and a **priority** (1-3):
- **Priority 1** = Critical (blocks basic training) -- weight 1.000
- **Priority 2** = Important (expected for production training) -- weight 0.500
- **Priority 3** = Nice-to-have (polish, edge cases) -- weight 0.333

Row weight: `w = 1 / priority`

**Points column**: 2 = fully implemented, 1 = partially implemented, 0 = not implemented, N/A = excluded

**Section score**: `percentage = sum(score_i * w_i) / sum(max_i * w_i) * 100`

**Overall score**:
```
weight_r = 1 / level
Readiness = (sum(percentage * weight_r) / sum(weight_r)) * 100
```

### Section Scores

| Section | Level | Max Pts | Earned | Percentage |
|---------|-------|---------|--------|------------|
| Device Registration & Management | 1 | | | |
| Memory & Allocator | 1 | | | |
| Operator Registration (Training) | 1 | | | |
| Autograd | 1 | | | |
| AMP | 1 | | | |
| Distributed Training | 1 | | | |
| Accelerator Hooks [PU1] | 2 | | | |
| Device Guard | 2 | | | |
| torch.compile | 2 | | | |
| Python Frontend | 2 | | | |
| Serialization (Checkpointing) | 2 | | | |
| Testing & Validation | 2 | | | |
| Dtype Support | 2 | | | |
| Numerical Accuracy | 2 | | | |
| Streams & Events | 3 | | | |
| RNG & Generator | 3 | | | |
| Autoload [PU1] | 3 | | | |
| DataLoader Integration | 3 | | | |
| Ecosystem Compatibility | 3 | | | |

**Readiness**: _____ %

---

## 0. Source Discovery & Integration Path Detection

Follow the same source discovery procedure as `checklist.md` Section 0.

| Source Discovery | Value | Notes |
|-----------------|-------|-------|
| **Source location** | | |
| **How found** | | |
| **Repository URL** | | |
| **Backend version** | | |

| Signal | Detected | Value | Notes |
|--------|----------|-------|-------|
| `rename_privateuse1_backend()` call found | `[ ]` | | |
| `entry_points` for `torch.backends` found | `[ ]` | | |
| Custom entry in `DeviceType.h` | `[ ]` | | |
| Custom entry in `DispatchKey.h` | `[ ]` | | |
| **Detected path** | | _[PU1 / Fork / Hybrid]_ | |

---

## 1. Device Registration & Management -- Level: **1**

### 1.1 Backend Registration

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 1.1.1 | Backend name registered via `rename_privateuse1_backend("<name>")` | PU1 | 1 | | |
| 1.1.2 | `torch._register_device_module("<name>", module)` | PU1 | 1 | | |
| 1.1.3 | Device type added to `c10::DeviceType` enum | Fork | 1 | | |
| 1.1.4 | Custom dispatch key registered in `DispatchKey.h` | Fork | 1 | | |
| 1.1.5 | `torch.device("<name>")` and `torch.device("<name>:0")` work | Both | 1 | | |
| 1.1.6 | `generate_methods_for_privateuse1_backend("<name>")` | PU1 | 2 | | |

### 1.2 Device Management

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 1.2.1 | `deviceCount()` returns correct count | 1 | | |
| 1.2.2 | `setCurrentDevice()` / `getCurrentDevice()` | 1 | | |
| 1.2.3 | `torch.<name>.is_available()` | 1 | | |
| 1.2.4 | `torch.<name>.device_count()` | 1 | | |
| 1.2.5 | `exchangeDevice()` / `maybeExchangeDevice()` | 2 | | |
| 1.2.6 | Multi-device indexing (`<name>:0`, `<name>:1`, ...) | 2 | | |
| 1.2.7 | `torch.<name>.synchronize()` | 2 | | |

---

## 2. Accelerator Hooks **[PU1]** -- Level: **2**

### Mandatory hooks

| # | Hook | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 2.1 | `hasPrimaryContext(device_index)` | 1 | | |
| 2.2 | `getDefaultGenerator(device_index)` | 1 | | |
| 2.3 | `getDeviceFromPtr(void* data)` | 1 | | |
| 2.4 | `isBuilt()` | 1 | | |
| 2.5 | `isAvailable()` | 1 | | |
| 2.6 | `getNewGenerator(device_index)` | 2 | | |
| 2.7 | `getPinnedMemoryAllocator()` | 2 | | |
| 2.8 | `resizePrivateUse1Bytes(storage, newsize)` | 2 | | |

### Hooks with safe defaults

| # | Hook | Default | Priority | Points | Notes |
|---|------|---------|----------|--------|-------|
| 2.9 | `deviceCount()` | returns 0 | 1 | | |
| 2.10 | `setCurrentDevice(device)` | throws | 1 | | |
| 2.11 | `getCurrentDevice()` | throws | 1 | | |
| 2.12 | `init()` | no-op | 2 | | |
| 2.13 | `exchangeDevice(device)` | throws | 2 | | |
| 2.14 | `isPinnedPtr(data)` | returns false | 3 | | |
| 2.15 | `maybeExchangeDevice(device)` | throws | 3 | | |

---

## 3. Device Guard -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 3.1 | `c10::impl::DeviceGuardImpl` subclass implemented | Both | 1 | | |
| 3.2 | Registered via `C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardClass)` | PU1 | 1 | | |
| 3.3 | Registered via `C10_REGISTER_GUARD_IMPL(<Key>, GuardClass)` | Fork | 1 | | |
| 3.4 | `getDevice()` / `setDevice()` / `uncheckedSetDevice()` | Both | 1 | | |
| 3.5 | Guard saves/restores device+stream on scope exit | Both | 1 | | |
| 3.6 | `getStream()` / `setStream()` / `exchangeStream()` | Both | 2 | | |

---

## 4. Memory & Allocator -- Level: **1**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 4.1 | Device allocator (`c10::Allocator` subclass) | 1 | | |
| 4.2 | Allocator registered globally | 1 | | |
| 4.3 | Host/pinned memory allocator | 2 | | |
| 4.4 | `torch.<name>.memory_allocated()` | 2 | | |
| 4.5 | `torch.<name>.empty_cache()` | 2 | | |
| 4.6 | OOM produces a clear error message (not a segfault) | 2 | | |
| 4.7 | Pinned memory (`pin_memory=True` in DataLoader) | 2 | | |
| 4.8 | `torch.<name>.max_memory_allocated()` | 3 | | |
| 4.9 | `torch.<name>.memory_reserved()` (if caching allocator) | 3 | | |
| 4.10 | `memory_summary()` or equivalent | 3 | | |

---

## 5. Operator Registration (Training) -- Level: **1**

### 5.1 Minimal Kernel Set

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 5.1.1 | `empty.memory_format` -- tensor factory | 1 | | |
| 5.1.2 | `_copy_from` / `_copy_from_and_resize` -- device<->CPU | 1 | | |
| 5.1.3 | `_local_scalar_dense` -- scalar extraction (`.item()`) | 1 | | |
| 5.1.4 | Tensor creation ops (`torch.zeros`, `ones`, `randn`, `full`) | 1 | | |

### 5.2 Training-Critical Operator Coverage

| Category | Example Ops | Priority | Points | Pass/Total | Notes |
|----------|-------------|----------|--------|------------|-------|
| Elementwise unary | `abs`, `neg`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu` | 1 | | | |
| Elementwise binary | `add`, `sub`, `mul`, `div`, `remainder`, `pow`, `maximum`, `minimum` | 1 | | | |
| Reduction | `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `any`, `all`, `prod` | 1 | | | |
| Linear algebra | `mm`, `bmm`, `addmm`, `matmul`, `linear`, `einsum` | 1 | | | |
| Normalization | `batch_norm`, `layer_norm`, `group_norm`, `instance_norm` | 1 | | | |
| Activation | `relu`, `gelu`, `silu`, `sigmoid`, `softmax`, `log_softmax` | 1 | | | |
| Attention | `scaled_dot_product_attention` | 1 | | | |
| Memory ops | `clone`, `copy_`, `fill_`, `zero_`, `empty_like`, `zeros_like` | 1 | | | |
| Loss functions | `cross_entropy`, `mse_loss`, `nll_loss`, `binary_cross_entropy_with_logits` | 1 | | | |
| Convolution | `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d` | 2 | | | |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d` | 2 | | | |
| Indexing | `index`, `index_put`, `index_select`, `gather`, `scatter`, `masked_fill` | 2 | | | |
| Shape ops | `reshape`, `view`, `permute`, `transpose`, `contiguous`, `cat`, `stack`, `chunk`, `split` | 2 | | | |
| Random | `uniform_`, `normal_`, `bernoulli_`, `dropout`, `rand`, `randn` | 2 | | | |
| Embedding | `embedding`, `embedding_bag` | 2 | | | |
| Type casting | `to(dtype)`, `float()`, `half()`, `bfloat16()` | 2 | | | |
| Comparison | `eq`, `ne`, `lt`, `gt`, `le`, `ge` | 2 | | | |
| Optimizer ops | `addcmul_`, `addcdiv_`, `lerp_`, `clamp_` | 2 | | | |

### 5.3 Fallback Mechanisms

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 5.3.1 | `Autograd<Key>` fallback for autograd | Both | 1 | | |
| 5.3.2 | Per-operator CPU fallback (device->CPU->compute->device) | Both | 2 | | |
| 5.3.3 | Global fallback via `torch::Library::fallback()` | Both | 2 | | |

---

## 6. Autograd -- Level: **1**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 6.1 | Dispatch key has corresponding `Autograd<Key>` registered | Both | 1 | | |
| 6.2 | `.backward()` completes on a simple loss | Both | 1 | | |
| 6.3 | Gradients match CUDA numerically (within tolerance) | Both | 1 | | |
| 6.4 | `torch.autograd.grad()` works | Both | 2 | | |
| 6.5 | Gradient accumulation (`.backward()` multiple times) | Both | 2 | | |
| 6.6 | `torch.autograd.Function` custom forward/backward on device | Both | 2 | | |
| 6.7 | Gradient checkpointing (`torch.utils.checkpoint.checkpoint`) | Both | 2 | | |
| 6.8 | Mixed precision backward (fp16/bf16 forward, fp32 grad) | Both | 2 | | |
| 6.9 | Higher-order gradients | Both | 3 | | |

---

## 7. Automatic Mixed Precision (AMP) -- Level: **1**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 7.1 | `Autocast<Key>` dispatch key kernels registered | Both | 1 | | |
| 7.2 | `torch.autocast(device_type="<name>")` works | Both | 1 | | |
| 7.3 | Training loop with AMP converges (loss decreases) | Both | 1 | | |
| 7.4 | `get_amp_supported_dtype()` returns supported dtypes | Both | 2 | | |
| 7.5 | Ops correctly cast to lower precision inside autocast | Both | 2 | | |
| 7.6 | Ops that need fp32 (softmax, layer_norm, loss) stay in fp32 | Both | 2 | | |
| 7.7 | `torch.amp.GradScaler("<name>")` works | Both | 2 | | |

---

## 8. torch.compile -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 8.1 | `DeviceInterface` subclass implemented | Both | 1 | | |
| 8.2 | Registered via `register_interface_for_device("<name>", Interface)` | Both | 1 | | |
| 8.3 | `torch.compile(model)` does not error on a simple training loop | Both | 1 | | |
| 8.4 | Compiled model produces correct gradients | Both | 1 | | |
| 8.5 | No unexpected graph breaks on standard training models | Both | 2 | | |
| 8.6 | FakeTensor / meta tensor support for device | Both | 2 | | |
| 8.7 | `torch.compile(model, mode="reduce-overhead")` works | Both | 3 | | |

---

## 9. Distributed Training -- Level: **1**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 9.1 | Custom `ProcessGroup` subclass implemented | Both | 1 | | |
| 9.2 | Registered via `torch.distributed.Backend.register_backend()` | Both | 1 | | |
| 9.3 | `init_process_group(backend="<name>")` works | Both | 1 | | |

### Collective Operations

| Collective | Priority | Points | Multi-node | Notes |
|-----------|----------|--------|------------|-------|
| `all_reduce` | 1 | | `[ ]` | |
| `broadcast` | 1 | | `[ ]` | |
| `all_gather` | 2 | | `[ ]` | |
| `reduce_scatter` | 2 | | `[ ]` | |
| `barrier` | 2 | | `[ ]` | |
| `send` / `recv` (P2P) | 3 | | `[ ]` | |
| `all_to_all` | 3 | | `[ ]` | |

### Distributed Strategies

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 9.4 | DDP training (2+ devices, loss converges) | 1 | | |
| 9.5 | FSDP training | 2 | | |
| 9.6 | Tensor Parallel | 2 | | |
| 9.7 | Multi-node training (2+ nodes) | 2 | | |
| 9.8 | Pipeline Parallel | 3 | | |
| 9.9 | `DeviceMesh` works | 3 | | |

---

## 10. Python Frontend & Device-Agnostic APIs -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 10.1 | `torch.<name>.is_available()` | Both | 1 | | |
| 10.2 | `Tensor.to(device)` / `Tensor.<name>()` / `Tensor.is_<name>` | Both | 1 | | |
| 10.3 | `nn.Module.to(device)` | Both | 1 | | |
| 10.4 | `torch.<name>.device_count()` | Both | 2 | | |
| 10.5 | `torch.<name>.synchronize()` | Both | 2 | | |
| 10.6 | `torch.accelerator.current_device()` | Both | 2 | | |

---

## 11. Serialization (Checkpointing) -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 11.1 | `torch.save(model.state_dict())` works with device tensors | Both | 1 | | |
| 11.2 | `torch.load(..., map_location="<name>")` works | Both | 1 | | |
| 11.3 | Resume training from checkpoint (optimizer state restored correctly) | Both | 1 | | |
| 11.4 | `torch.load(..., weights_only=True)` works | Both | 2 | | |
| 11.5 | `TensorBackendMetaRegistry` registered for save/load | PU1 | 2 | | |

---

## 12. Testing & Validation -- Level: **2**

### 12.1 Device-Generic Test Framework

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 12.1.1 | OpInfo-based operator compliance tests pass | Both | 1 | | |
| 12.1.2 | `instantiate_device_type_tests` runs on your device | PU1 | 2 | | |
| 12.1.3 | Common device dtype tests pass | Both | 2 | | |

### 12.2 Training-Specific Tests

| Test Area | Priority | Points | Notes |
|-----------|----------|--------|-------|
| Backward pass / autograd correctness | 1 | | |
| Gradient accumulation | 1 | | |
| AMP / autocast training loop | 2 | | |
| Distributed collectives | 2 | | |
| Optimizer convergence (SGD, Adam) | 2 | | |
| Gradient checkpointing | 2 | | |
| RNG reproducibility | 3 | | |
| Serialization round-trip (checkpoint/resume) | 2 | | |

---

## 13. Dtype Support Matrix -- Level: **2**

| Dtype | Priority | Points | Compute | Storage | AMP Target | Notes |
|-------|----------|--------|---------|---------|------------|-------|
| `float32` | 1 | | `[ ]` | `[ ]` | -- | |
| `float16` | 1 | | `[ ]` | `[ ]` | `[ ]` | |
| `bfloat16` | 1 | | `[ ]` | `[ ]` | `[ ]` | |
| `int32` | 2 | | `[ ]` | `[ ]` | -- | |
| `int64` | 2 | | `[ ]` | `[ ]` | -- | |
| `bool` | 2 | | `[ ]` | `[ ]` | -- | |
| `float64` | 3 | | `[ ]` | `[ ]` | -- | |

---

## 14. Numerical Accuracy -- Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 14.1 | float32 ops match CUDA within 1e-5 atol | 1 | | |
| 14.2 | float16 ops match CUDA within 1e-3 atol | 1 | | |
| 14.3 | Loss convergence curve matches CUDA on reference models | 1 | | |
| 14.4 | bfloat16 ops match CUDA within 1e-2 atol | 2 | | |
| 14.5 | Reduction ops handle large tensors without overflow | 2 | | |
| 14.6 | Matmul results are deterministic (same input -> same output) | 2 | | |

---

## 15. Streams & Events -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 15.1 | `torch.<name>.Stream` class functional | 2 | | |
| 15.2 | `torch.<name>.current_stream()` / `set_stream()` | 2 | | |
| 15.3 | `stream.synchronize()` | 2 | | |
| 15.4 | Event creation and recording | 2 | | |
| 15.5 | Non-blocking H2D/D2H transfer with stream overlap | 2 | | |
| 15.6 | `stream.wait_stream(other)` | 3 | | |

---

## 16. RNG & Generator -- Level: **3**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 16.1 | Custom `at::Generator` subclass | Both | 1 | | |
| 16.2 | `torch.<name>.manual_seed(seed)` | Both | 2 | | |
| 16.3 | Generator fork safety (fork handler registered) | Both | 2 | | |
| 16.4 | `torch.Generator(device='<name>')` works | Both | 2 | | |
| 16.5 | `get_rng_state()` / `set_rng_state()` for checkpointing | Both | 2 | | |
| 16.6 | `torch.use_deterministic_algorithms(True)` respected | Both | 3 | | |

---

## 17. Autoload **[PU1]** -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 17.1 | `entry_points` registered in `setup.py`/`pyproject.toml` | 2 | | |
| 17.2 | `_autoload()` callable initializes backend on `import torch` | 2 | | |
| 17.3 | `torch.device("<name>")` resolves without explicit import | 2 | | |

---

## 18. DataLoader Integration -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 18.1 | `DataLoader(pin_memory=True)` works | 2 | | |
| 18.2 | `tensor.to(device, non_blocking=True)` overlaps with compute | 2 | | |
| 18.3 | Multi-worker DataLoader doesn't deadlock with device | 2 | | |

---

## 19. Ecosystem Compatibility (Training) -- Level: **3**

| Library | Priority | Points | Version Tested | Notes |
|---------|----------|--------|----------------|-------|
| HuggingFace transformers | 1 | | | |
| HuggingFace accelerate | 1 | | | |
| PEFT (LoRA, QLoRA) | 2 | | | |
| DeepSpeed | 2 | | | |
| Megatron-LM | 2 | | | |
| torchvision | 2 | | | |
| Flash Attention | 2 | | | |
| triton (if applicable) | 2 | | | |
| bitsandbytes | 3 | | | |
| torchtune | 3 | | | |

---

## Appendix: Discovered APIs Not in Checklist

_[FILL: List any backend APIs or training-relevant features discovered during evaluation that are not covered by this checklist.]_
