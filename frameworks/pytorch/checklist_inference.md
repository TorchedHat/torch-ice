# PyTorch Inference Workload Readiness Checklist

> Evaluates hardware accelerator readiness for **inference workloads** with PyTorch.
> Covers forward pass, model loading, torch.compile, quantization, and deployment.
> For full PyTorch integration coverage, use `checklist.md`.

| Field | Value |
|-------|-------|
| **Backend** | _[FILL: backend name]_ |
| **Backend version** | _[FILL: release version/tag]_ |
| **Integration path** | _[FILL: PU1 or Fork]_ |
| **Dispatch key** | _[FILL: PrivateUse1 or custom]_ |
| **Evaluation date** | _[FILL: date]_ |
| **Evaluator** | _[FILL: human or agent]_ |
| **Source** | _[FILL: backend repo URL]_ |
| **Workload** | Inference |

---

## Readiness Score & Summary

### Executive Summary

_[FILL: Write a concise summary covering:
- **Overall Inference Readiness**: X%
- **Key strengths** for inference workloads
- **Key gaps** blocking inference use
- **Recommendations**]_

### Scoring Model

Every row has a max score of **2** and a **priority** (1-3):
- **Priority 1** = Critical (blocks basic inference) -- weight 1.000
- **Priority 2** = Important (expected for production inference) -- weight 0.500
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
| Operator Registration (Inference) | 1 | | | |
| torch.compile / Inductor | 1 | | | |
| Accelerator Hooks [PU1] | 2 | | | |
| Device Guard | 2 | | | |
| Serialization & Model Loading | 2 | | | |
| Python Frontend | 2 | | | |
| Dtype Support | 2 | | | |
| Numerical Accuracy | 2 | | | |
| Testing & Validation | 2 | | | |
| Quantization | 2 | | | |
| Streams & Events | 3 | | | |
| Autoload [PU1] | 3 | | | |
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
| 4.3 | OOM produces a clear error message (not a segfault) | 2 | | |
| 4.4 | `torch.<name>.memory_allocated()` | 2 | | |
| 4.5 | `torch.<name>.empty_cache()` | 2 | | |
| 4.6 | Host/pinned memory allocator | 3 | | |
| 4.7 | `torch.<name>.max_memory_allocated()` | 3 | | |
| 4.8 | `memory_summary()` or equivalent | 3 | | |

---

## 5. Operator Registration (Inference) -- Level: **1**

### 5.1 Minimal Kernel Set

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 5.1.1 | `empty.memory_format` -- tensor factory | 1 | | |
| 5.1.2 | `_copy_from` / `_copy_from_and_resize` -- device<->CPU | 1 | | |
| 5.1.3 | `_local_scalar_dense` -- scalar extraction (`.item()`) | 1 | | |
| 5.1.4 | Tensor creation ops (`torch.zeros`, `ones`, `randn`, `full`) | 1 | | |

### 5.2 Inference-Critical Operator Coverage

| Category | Example Ops | Priority | Points | Pass/Total | Notes |
|----------|-------------|----------|--------|------------|-------|
| Elementwise unary | `abs`, `neg`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu` | 1 | | | |
| Elementwise binary | `add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum` | 1 | | | |
| Linear algebra | `mm`, `bmm`, `addmm`, `matmul`, `linear`, `einsum` | 1 | | | |
| Normalization | `batch_norm`, `layer_norm`, `group_norm`, `instance_norm` | 1 | | | |
| Activation | `relu`, `gelu`, `silu`, `sigmoid`, `softmax`, `log_softmax` | 1 | | | |
| Attention | `scaled_dot_product_attention` | 1 | | | |
| Memory ops | `clone`, `copy_`, `fill_`, `zero_`, `empty_like`, `zeros_like` | 1 | | | |
| Reduction | `sum`, `mean`, `max`, `min`, `argmax`, `argmin` | 1 | | | |
| Shape ops | `reshape`, `view`, `permute`, `transpose`, `contiguous`, `cat`, `stack`, `chunk`, `split`, `unsqueeze`, `squeeze` | 2 | | | |
| Convolution | `conv1d`, `conv2d`, `conv3d` | 2 | | | |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d` | 2 | | | |
| Indexing | `index`, `index_select`, `gather` | 2 | | | |
| Type casting | `to(dtype)`, `float()`, `half()`, `bfloat16()`, `int()` | 2 | | | |
| Embedding | `embedding`, `embedding_bag` | 2 | | | |
| Sorting | `sort`, `topk`, `argsort` | 3 | | | |

### 5.3 Fallback Mechanisms

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 5.3.1 | Per-operator CPU fallback (device->CPU->compute->device) | Both | 2 | | |
| 5.3.2 | Global fallback via `torch::Library::fallback()` | Both | 2 | | |
| 5.3.3 | Fallthrough registration for metadata/shape-only dispatch | Both | 3 | | |

---

## 6. torch.compile / Inductor -- Level: **1**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 6.1 | `DeviceInterface` subclass implemented | Both | 1 | | |
| 6.2 | Registered via `register_interface_for_device("<name>", Interface)` | Both | 1 | | |
| 6.3 | `torch.compile(model)` does not error on a simple model | Both | 1 | | |
| 6.4 | Compiled model produces correct output | Both | 1 | | |
| 6.5 | No unexpected graph breaks on standard inference models | Both | 2 | | |
| 6.6 | FakeTensor / meta tensor support for device | Both | 2 | | |
| 6.7 | `torch.compile(model, mode="reduce-overhead")` works | Both | 2 | | |
| 6.8 | Custom Inductor codegen registered (if applicable) | Both | 3 | | |
| 6.9 | AOTInductor export works | Both | 3 | | |

---

## 7. Serialization & Model Loading -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 7.1 | `torch.save(model.state_dict())` works with device tensors | Both | 1 | | |
| 7.2 | `torch.load(..., map_location="<name>")` works | Both | 1 | | |
| 7.3 | Load a state_dict saved on CUDA onto your device | Both | 1 | | |
| 7.4 | Load a state_dict saved on your device onto CPU | Both | 2 | | |
| 7.5 | `torch.load(..., weights_only=True)` works | Both | 2 | | |
| 7.6 | `TensorBackendMetaRegistry` registered for save/load | PU1 | 2 | | |
| 7.7 | `safetensors` load/save | Both | 3 | | |

---

## 8. Python Frontend & Device-Agnostic APIs -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 8.1 | `torch.<name>.is_available()` | Both | 1 | | |
| 8.2 | `Tensor.to(device)` / `Tensor.<name>()` / `Tensor.is_<name>` | Both | 1 | | |
| 8.3 | `nn.Module.to(device)` | Both | 1 | | |
| 8.4 | `torch.<name>.device_count()` | Both | 2 | | |
| 8.5 | `torch.<name>.synchronize()` | Both | 2 | | |
| 8.6 | `torch.accelerator.current_device()` | Both | 2 | | |

---

## 9. Dtype Support Matrix -- Level: **2**

| Dtype | Priority | Points | Compute | Storage | Notes |
|-------|----------|--------|---------|---------|-------|
| `float32` | 1 | | `[ ]` | `[ ]` | |
| `float16` | 1 | | `[ ]` | `[ ]` | |
| `bfloat16` | 1 | | `[ ]` | `[ ]` | |
| `int8` | 2 | | `[ ]` | `[ ]` | quantization |
| `int32` | 2 | | `[ ]` | `[ ]` | |
| `int64` | 2 | | `[ ]` | `[ ]` | |
| `bool` | 2 | | `[ ]` | `[ ]` | |
| `float8_e4m3fn` | 3 | | `[ ]` | `[ ]` | |
| `float8_e5m2` | 3 | | `[ ]` | `[ ]` | |
| `uint8` | 3 | | `[ ]` | `[ ]` | |
| `float64` | 3 | | `[ ]` | `[ ]` | |

---

## 10. Numerical Accuracy -- Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 10.1 | float32 ops match CUDA within 1e-5 atol | 1 | | |
| 10.2 | float16 ops match CUDA within 1e-3 atol | 1 | | |
| 10.3 | Forward pass output matches CUDA on reference models | 1 | | |
| 10.4 | bfloat16 ops match CUDA within 1e-2 atol | 2 | | |
| 10.5 | Reduction ops handle large tensors without overflow | 2 | | |
| 10.6 | Matmul results are deterministic (same input -> same output) | 2 | | |

---

## 11. Testing & Validation -- Level: **2**

### 11.1 Device-Generic Test Framework

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 11.1.1 | OpInfo-based operator compliance tests pass | Both | 1 | | |
| 11.1.2 | `instantiate_device_type_tests` runs on your device | PU1 | 2 | | |
| 11.1.3 | Common device dtype tests pass | Both | 2 | | |

### 11.2 Inference-Specific Tests

| Test Area | Priority | Points | Notes |
|-----------|----------|--------|-------|
| Forward pass correctness (no backward) | 1 | | |
| Model load + forward (state_dict from CUDA) | 1 | | |
| torch.compile correctness (inference mode) | 2 | | |
| Quantized model forward pass | 2 | | |
| Dtype casting (fp32->fp16->bf16) | 2 | | |
| Serialization round-trip (save/load) | 2 | | |
| Throughput benchmark vs baseline | 3 | | |

---

## 12. Streams & Events -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 12.1 | `torch.<name>.Stream` class functional | 2 | | |
| 12.2 | `torch.<name>.current_stream()` / `set_stream()` | 2 | | |
| 12.3 | `stream.synchronize()` | 2 | | |
| 12.4 | Non-blocking H2D/D2H transfer with stream overlap | 2 | | |
| 12.5 | Event creation and recording | 2 | | |
| 12.6 | `event.elapsed_time(end_event)` -- latency measurement | 3 | | |

---

## 13. Autoload **[PU1]** -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 13.1 | `entry_points` registered in `setup.py`/`pyproject.toml` | 2 | | |
| 13.2 | `_autoload()` callable initializes backend on `import torch` | 2 | | |
| 13.3 | `torch.device("<name>")` resolves without explicit import | 2 | | |

---

## 14. Quantization -- Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 14.1 | int8 quantized op kernels | 2 | | |
| 14.2 | `torch.quantization.quantize_dynamic` works on device | 2 | | |
| 14.3 | `torch.quantization.quantize_static` works on device | 2 | | |
| 14.4 | float8 quantized ops (e4m3fn, e5m2) | 3 | | |
| 14.5 | bitsandbytes 4-bit / 8-bit quantization | 3 | | |

---

## 15. Ecosystem Compatibility (Inference) -- Level: **3**

| Library | Priority | Points | Version Tested | Notes |
|---------|----------|--------|----------------|-------|
| HuggingFace transformers | 1 | | | |
| HuggingFace accelerate | 2 | | | |
| vLLM / TGI | 2 | | | |
| torchvision | 2 | | | |
| ONNX Export (`torch.onnx.export`) | 2 | | | |
| diffusers | 3 | | | |
| bitsandbytes | 3 | | | |
| Flash Attention | 3 | | | |

---

## Appendix: Discovered APIs Not in Checklist

_[FILL: List any backend APIs or inference-relevant features discovered during evaluation that are not covered by this checklist.]_
