# PyTorch Accelerator Integration Readiness Checklist (Unified)

> A comprehensive template for evaluating hardware accelerator integration
> readiness with PyTorch -- covers both **PrivateUse1 (out-of-tree)** and
> **Fork (in-tree/vendor fork)** integration paths.
>
> Items marked with **[PU1]** apply only to PrivateUse1 backends.
> Items marked with **[Fork]** apply only to fork-based backends.
> Unmarked items apply to **both** paths.

| Field | Value |
|-------|-------|
| **Backend** | _[FILL: backend name]_ |
| **Backend version** | _[FILL: release version/tag, e.g. v0.3.0]_ |
| **Integration path** | _[FILL: PU1 or Fork]_ |
| **Dispatch key** | _[FILL: PrivateUse1 or custom]_ |
| **Evaluation date** | _[FILL: date]_ |
| **Evaluator** | _[FILL: human or agent]_ |
| **torch-air version** | _[FILL: read from VERSION in torch-air repo, or git describe --tags --always]_ |
| **Model** | _[FILL: AI model name used to generate this report]_ |
| **Source** | _[FILL: repo URL]_ |
| **PyTorch version** | _[FILL: PyTorch version supported/tested by the backend — from setup.py, pyproject.toml, requirements, README compatibility table, or CI config]_ |
| **Fork base** | _[FILL: version/commit, or omit row for PU1]_ |


---

## Readiness Score & Summary

### Executive Summary

_[FILL: Write a concise summary covering:
- **Overall Readiness**: X%
- **Notable insights**:
  - Surprising finding or important context 1
  - Surprising finding or important context 2
- **Key strengths**:
  - Top strength 1
  - Top strength 2
  - Top strength 3
- **Key gaps**:
  - Critical gap 1
  - Critical gap 2
  - Critical gap 3
- **Upstream candidates**: N features identified as generic enough for PyTorch core (see below)]_

### Scoring Model

Every row has a max score of **2** and a **priority** (1-3) reflecting its importance:
- **Priority 1** = Critical (blocks basic functionality) -- weight 1.000
- **Priority 2** = Important (expected for production use) -- weight 0.500
- **Priority 3** = Nice-to-have (polish, edge cases) -- weight 0.333

Row weight: `w = 1 / priority`

Each section has a **level** (1-3). Level 1 sections are weighted highest.

**Points column**: 2 = fully implemented, 1 = partially implemented, 0 = not implemented, N/A = excluded

**Section score**: `percentage = sum(score_i * w_i) / sum(max_i * w_i) * 100` (max_i = 2 for non-N/A rows)

**Overall score formula**:
```
weight_r = 1 / level
Readiness = (sum(percentage * weight_r) / sum(weight_r)) * 100
```
where the sums are over all applicable sections (excluding fully N/A sections).

### Section Scores

| Section | Max Pts | Earned | Percentage |
|---------|---------|--------|------------|
| **Level 1** | | | |
| Device Registration & Management | | | |
| Operator Registration | | | |
| Autograd | | | |
| Device Guard | | | |
| Accelerator Hooks [PU1] | | | |
| Memory & Allocator | | | |
| **Level 2** | | | |
| Serialization & Model Portability | | | |
| Python Frontend & Device-Agnostic APIs | | | |
| AMP | | | |
| torch.compile / Inductor | | | |
| Distributed Training | | | |
| Dtype Support Matrix | | | |
| Numerical Accuracy | | | |
| Testing & Validation | | | |
| **Level 3** | | | |
| Streams & Events | | | |
| RNG & Generator | | | |
| Autoload [PU1] | | | |
| DataLoader Integration | | | |
| Profiler | | | |
| Ecosystem Compatibility | | | |
| Additional PyTorch APIs | | | |

### Calculation

```
Row weight: w_i = 1 / priority_i
Percentage = sum(score_i * w_i) / sum(max_i * w_i) * 100  (excluding N/A rows, max_i = 2)
weight_r = 1 / level
Readiness = (sum(Percentage * weight_r) / sum(weight_r for applicable sections)) * 100
```

**Readiness**: _____ %

### Upstream Candidates (Advisory)

> Features discovered in this backend that are generic enough to benefit
> all PU1/accelerator backends if upstreamed to PyTorch core.
> Advisory only -- does not affect the readiness score.

**Classification key**:
- **Generic** -- no backend-specific references; ready to upstream into `torch/accelerator/` as-is for any PU1 backend
- **Needs Abstraction** -- solves a problem every backend faces, but implementation references backend-specific internals; upstream path is to define an interface/hook in core

_[FILL: For each discovered feature, use this format:]_

#### _[Feature Name]_

_[Brief description of what the feature does]_

**Classification**: _[Generic | Needs Abstraction | Hardware-Specific]_

**Relevant files**: _[source files and key symbols]_

**Current state in PyTorch**: _[whether PyTorch core has an equivalent, partial equivalent, or nothing]_

**Motivation**: _[why this would benefit all backends if upstreamed]_

---

## 0. Source Discovery & Integration Path Detection

The agent handles everything -- the user only provides a backend name (e.g.,
"ascend npu", "habana gaudi", "graphcore ipu"). The agent locates the source
code and determines the integration path automatically.

### Step 1: Locate the source code

The user may not have the repo locally. The agent should:

1. **Check locally first**:
   - `pip show torch_<name>` or `pip show <name>` to find installed package location
   - `python -c "import torch_<name>; print(torch_<name>.__file__)"` if importable
   - Search current directory and parents for the backend source

2. **Search GitHub/GitLab if not found locally**:
   - Search `github.com/<vendor>/` for vendor backends (e.g., `huawei/torch_npu`, `RBLN-SW/torch-rbln`)
   - Try common naming patterns: `torch-<name>`, `torch_<name>`, `pytorch-<name>`, `<vendor>-pytorch`
   - Use `gh search repos "torch <name> pytorch backend"` or web search

3. **Clone if needed**: Once found, clone to a scratch location:
   - `git clone <url> /tmp/<name>` or `agent_space/<name>`
   - Shallow clone is fine: `git clone --depth 1`

4. **Ask the user** only if the source cannot be found after searching.

Record the source location and proceed.

| Source Discovery | Value | Notes |
|-----------------|-------|-------|
| **Source location** | _[local path or URL]_ | |
| **How found** | _[local / pip / GitHub / user-provided]_ | |
| **Repository URL** | _[if applicable]_ | |
| **Backend version** | _[latest release tag or pip version]_ | Use `gh release list`, `git describe --tags`, or `pip show` |

### Step 2: Detect integration path

1. Check if the backend is a **separate package** (e.g., `torch_npu`, `torch_xla`):
   - Search for `rename_privateuse1_backend` or `register_privateuse1_backend` in the source
   - Search for `entry_points` with `torch.backends` group in `setup.py`/`pyproject.toml`
   - If found -> **PrivateUse1 (out-of-tree)**

2. Check if the backend is **inside a PyTorch fork**:
   - Check if `c10/core/DeviceType.h` contains a custom device type entry
   - Check if `c10/core/DispatchKey.h` contains a custom dispatch key
   - Check if `aten/src/ATen/native/native_functions.yaml` has dispatch entries for the backend
   - If found -> **Fork (in-tree)**

3. **Hybrid** (rare): Some backends start as a fork and migrate to PrivateUse1, or
   maintain both paths. If both signals are present, note it and evaluate both.

| Signal | Detected | Value | Notes |
|--------|----------|-------|-------|
| `rename_privateuse1_backend()` call found | `[ ]` | | |
| `entry_points` for `torch.backends` found | `[ ]` | | |
| Custom entry in `DeviceType.h` | `[ ]` | | |
| Custom entry in `DispatchKey.h` | `[ ]` | | |
| Dispatch entries in `native_functions.yaml` | `[ ]` | | |
| Separate pip-installable package | `[ ]` | | |
| Modifies PyTorch core source files | `[ ]` | | |
| **Detected path** | | _[PU1 / Fork / Hybrid]_ | |
| **Dispatch key** | | _[PrivateUse1 / custom]_ | |
| **Fork base** (if fork) | | _[version/commit]_ | |

Once detected, the agent should:
- Mark all items for the **other** path as `[N/A]`
- Fill the header metadata fields above with detected values
- Proceed with evaluation of applicable items

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

## 2. Accelerator Hooks **[PU1]** -- Level: **1**

### Mandatory hooks (throw if not overridden)

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

### Hooks with safe defaults (override recommended)

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

## 3. Device Guard -- Level: **1**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 3.1 | `c10::impl::DeviceGuardImpl` subclass implemented | Both | 1 | | |
| 3.2 | Registered via `C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardClass)` | PU1 | 1 | | |
| 3.3 | Registered via `C10_REGISTER_GUARD_IMPL(<Key>, GuardClass)` | Fork | 1 | | |
| 3.4 | `getDevice()` / `setDevice()` / `uncheckedSetDevice()` | Both | 1 | | |
| 3.5 | Guard saves/restores device+stream on scope exit | Both | 1 | | |
| 3.6 | `getStream()` / `setStream()` / `exchangeStream()` | Both | 2 | | |

---

## 4. Autoload **[PU1]** -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 4.1 | `entry_points` registered in `setup.py`/`pyproject.toml` (group: `torch.backends`) | 2 | | |
| 4.2 | `_autoload()` callable initializes backend on `import torch` | 2 | | |
| 4.3 | `torch.device("<name>")` resolves without explicit import of backend package | 2 | | |

---

## 5. Memory & Allocator -- Level: **1**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 5.1 | Device allocator (`c10::Allocator` subclass) | 1 | | |
| 5.2 | Allocator registered globally | 1 | | |
| 5.3 | Host/pinned memory allocator | 2 | | |
| 5.4 | `torch.<name>.memory_allocated()` | 2 | | |
| 5.5 | `torch.<name>.empty_cache()` | 2 | | |
| 5.6 | OOM produces a clear error message (not a segfault) | 2 | | |
| 5.7 | Pinned memory (`pin_memory=True` in DataLoader) | 2 | | |
| 5.8 | `torch.<name>.max_memory_allocated()` | 3 | | |
| 5.9 | `torch.<name>.memory_reserved()` (if caching allocator) | 3 | | |
| 5.10 | `memory_summary()` or equivalent | 3 | | |

---

## 6. Streams & Events -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 6.1 | Stream implementation (pool management, priority support) | 2 | | |
| 6.2 | `torch.<name>.Stream` class functional | 2 | | |
| 6.3 | `torch.<name>.current_stream()` / `set_stream()` | 2 | | |
| 6.4 | `stream.synchronize()` | 2 | | |
| 6.5 | Event creation and recording | 2 | | |
| 6.6 | Non-blocking H2D/D2H transfer with stream overlap | 2 | | |
| 6.7 | `stream.wait_stream(other)` | 3 | | |
| 6.8 | `event.wait()` / `event.synchronize()` | 3 | | |
| 6.9 | `event.elapsed_time(end_event)` | 3 | | |

---

## 7. RNG & Generator -- Level: **3**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 7.1 | Custom `at::Generator` subclass | Both | 1 | | |
| 7.2 | Registered via `REGISTER_GENERATOR_PRIVATEUSE1(GeneratorClass)` | PU1 | 2 | | |
| 7.3 | `torch.<name>.manual_seed(seed)` | Both | 2 | | |
| 7.4 | Generator fork safety (fork handler registered) | Both | 2 | | |
| 7.5 | `torch.Generator(device='<name>')` works | Both | 2 | | |
| 7.6 | `get_rng_state()` / `set_rng_state()` for checkpointing | Both | 2 | | |
| 7.7 | `torch.<name>.initial_seed()` | Both | 3 | | |
| 7.8 | `torch.use_deterministic_algorithms(True)` respected | Both | 3 | | |

---

## 8. Operator Registration -- Level: **1**

### 8.1 Minimal Kernel Set

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 8.1.1 | `empty.memory_format` -- tensor factory | 1 | | |
| 8.1.2 | `_copy_from` / `_copy_from_and_resize` -- device<->CPU | 1 | | |
| 8.1.3 | `_local_scalar_dense` -- scalar extraction (`.item()`) | 1 | | |
| 8.1.4 | Tensor creation ops (`torch.zeros`, `ones`, `randn`, `full`, `arange`, `linspace`) | 1 | | |

### 8.2 Extended Operator Coverage

| Category | Example Ops | Priority | Points | Pass/Total | Notes |
|----------|--------|-------------|----------|------------|-------|
| Elementwise unary | `abs`, `neg`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu` | 1 | | | |
| Elementwise binary | `add`, `sub`, `mul`, `div`, `remainder`, `pow`, `maximum`, `minimum` | 1 | | | |
| Reduction | `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `any`, `all`, `prod` | 1 | | | |
| Linear algebra | `mm`, `bmm`, `addmm`, `matmul`, `linear`, `einsum` | 1 | | | |
| Normalization | `batch_norm`, `layer_norm`, `group_norm`, `instance_norm` | 1 | | | |
| Activation | `relu`, `gelu`, `silu`, `sigmoid`, `softmax`, `log_softmax` | 1 | | | |
| Attention | `scaled_dot_product_attention` | 1 | | | |
| Memory ops | `clone`, `copy_`, `fill_`, `zero_`, `empty_like`, `zeros_like` | 1 | | | |
| Comparison | `eq`, `ne`, `lt`, `gt`, `le`, `ge` | 2 | | | |
| Convolution | `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d` | 2 | | | |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d` | 2 | | | |
| Indexing | `index`, `index_put`, `index_select`, `gather`, `scatter`, `masked_fill` | 2 | | | |
| Shape ops | `reshape`, `view`, `permute`, `transpose`, `contiguous`, `cat`, `stack`, `chunk`, `split`, `unsqueeze`, `squeeze`, `expand`, `repeat` | 2 | | | |
| Random | `uniform_`, `normal_`, `bernoulli_`, `dropout`, `rand`, `randn` | 2 | | | |
| Embedding | `embedding`, `embedding_bag` | 2 | | | |
| Loss functions | `cross_entropy`, `mse_loss`, `nll_loss`, `binary_cross_entropy_with_logits` | 2 | | | |
| Type casting | `to(dtype)`, `float()`, `half()`, `bfloat16()`, `int()`, `bool()` | 2 | | | |
| Sorting | `sort`, `topk`, `argsort` | 3 | | | |

### 8.3 Model-Level Validation

| Model | Framework | Priority | Points | Notes |
|-------|-----------|----------|--------|-------|
| Llama-2-7B (inference) | HuggingFace transformers | 1 | | |
| ResNet-50 | torchvision | 2 | | |
| BERT-base | HuggingFace transformers | 2 | | |
| GPT-2 | HuggingFace transformers | 2 | | |
| Vision Transformer (ViT) | torchvision / timm | 3 | | |
| Stable Diffusion (UNet) | diffusers | 3 | | |
| Whisper | HuggingFace transformers | 3 | | |
| T5 | HuggingFace transformers | 3 | | |

For each: forward pass / backward pass / numerics match CUDA (within tolerance)

### 8.4 Fallback Mechanisms

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 8.4.1 | `Autograd<Key>` fallback for autograd | Both | 1 | | |
| 8.4.2 | Per-operator CPU fallback (device->CPU->compute->device) | Both | 2 | | |
| 8.4.3 | Global fallback via `torch::Library::fallback()` | Both | 2 | | |
| 8.4.4 | Fallthrough registration for metadata/shape-only dispatch | Both | 3 | | |

### 8.5 Custom Operators

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 8.5.1 | `TORCH_LIBRARY(<ns>, m)` schema registration | 2 | | |
| 8.5.2 | Kernel implementation + dispatch registration | 2 | | |
| 8.5.3 | Meta (shape-inference) kernel for torch.compile | 2 | | |
| 8.5.4 | `torch.autograd.Function` for custom backward | 2 | | |

---

## 9. Python Frontend & Device-Agnostic APIs -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 9.1 | `torch.<name>.is_available()` | Both | 1 | | |
| 9.2 | `Tensor.to(device)` / `Tensor.<name>()` / `Tensor.is_<name>` | Both | 1 | | |
| 9.3 | `nn.Module.to(device)` | Both | 1 | | |
| 9.4 | `torch.<name>.device_count()` | Both | 2 | | |
| 9.5 | `torch.<name>.synchronize()` | Both | 2 | | |
| 9.6 | `torch.accelerator.current_device()` | Both | 2 | | |
| 9.7 | `torch.accelerator.device_count()` | Both | 3 | | |
| 9.8 | `torch.accelerator.is_available()` | Both | 3 | | |
| 9.9 | `torch.accelerator.synchronize()` | Both | 3 | | |
| 9.10 | `torch.accelerator.current_stream()` / `set_stream()` | Both | 3 | | |

---

## 10. Autograd (Training Support) -- Level: **1**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 10.1 | Dispatch key has corresponding `Autograd<Key>` registered | Both | 1 | | |
| 10.2 | `.backward()` completes on a simple loss | Both | 1 | | |
| 10.3 | Gradients match CUDA numerically (within tolerance) | Both | 1 | | |
| 10.4 | `torch.autograd.grad()` works | Both | 2 | | |
| 10.5 | Gradient accumulation (`.backward()` multiple times) | Both | 2 | | |
| 10.6 | `torch.autograd.Function` custom forward/backward on device | Both | 2 | | |
| 10.7 | Gradient checkpointing (`torch.utils.checkpoint.checkpoint`) | Both | 2 | | |
| 10.8 | Mixed precision backward (fp16/bf16 forward, fp32 grad) | Both | 2 | | |
| 10.9 | Higher-order gradients (if needed) | Both | 3 | | |

---

## 11. Automatic Mixed Precision (AMP) -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 11.1 | `Autocast<Key>` dispatch key kernels registered | Both | 1 | | |
| 11.2 | `torch.autocast(device_type="<name>")` works | Both | 1 | | |
| 11.3 | Training loop with AMP converges (loss decreases) | Both | 1 | | |
| 11.4 | `get_amp_supported_dtype()` returns supported dtypes | Both | 2 | | |
| 11.5 | Ops correctly cast to lower precision inside autocast | Both | 2 | | |
| 11.6 | Ops that need fp32 (softmax, layer_norm, loss) stay in fp32 | Both | 2 | | |
| 11.7 | `torch.amp.GradScaler("<name>")` works (if needed) | Both | 2 | | |

---

## 12. torch.compile / Inductor -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 12.1 | `DeviceInterface` subclass implemented | Both | 1 | | |
| 12.2 | Registered via `register_interface_for_device("<name>", Interface)` | Both | 1 | | |
| 12.3 | `torch.compile(model)` does not error on a simple model | Both | 1 | | |
| 12.4 | Compiled model produces correct output | Both | 1 | | |
| 12.5 | No unexpected graph breaks on standard models | Both | 2 | | |
| 12.6 | FakeTensor / meta tensor support for device | Both | 2 | | |
| 12.7 | `torch.compile(model, mode="reduce-overhead")` works | Both | 3 | | |
| 12.8 | Custom Inductor codegen registered (if applicable) | Both | 3 | | |
| 12.9 | AOTInductor export works (if targeting inference deployment) | Both | 3 | | |

---

## 13. Serialization & Model Portability -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 13.1 | `TensorBackendMetaRegistry` registered for save/load | PU1 | 1 | | |
| 13.2 | `torch.save(model.state_dict())` works with device tensors | Both | 1 | | |
| 13.3 | `torch.load(..., map_location="<name>")` works | Both | 1 | | |
| 13.4 | Load a state_dict saved on CUDA onto your device | Both | 1 | | |
| 13.5 | Load a state_dict saved on your device onto CPU | Both | 2 | | |
| 13.6 | `torch.load(..., weights_only=True)` works | Both | 2 | | |
| 13.7 | `safetensors` load/save (if supported by ecosystem) | Both | 3 | | |

---

## 14. Distributed Training -- Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 14.1 | Custom `ProcessGroup` subclass implemented | Both | 1 | | |
| 14.2 | Registered via `torch.distributed.Backend.register_backend()` | Both | 1 | | |
| 14.3 | `init_process_group(backend="<name>")` works | Both | 1 | | |

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
| 14.4 | DDP training (2+ devices, loss converges) | 1 | | |
| 14.5 | FSDP training | 2 | | |
| 14.6 | Tensor Parallel | 2 | | |
| 14.7 | Multi-node training (2+ nodes) | 2 | | |
| 14.8 | Pipeline Parallel | 3 | | |
| 14.9 | `DeviceMesh` works | 3 | | |

---

## 15. Profiler -- Level: **3**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 15.1 | `ProfilerStubs` registered (operator-level timing) | PU1 | 2 | | |
| 15.2 | `torch.profiler.profile` collects device traces | Both | 2 | | |
| 15.3 | Kernel-level timing visible | Both | 2 | | |
| 15.4 | Kineto `IActivityProfiler` plugin | PU1 | 3 | | |
| 15.5 | Correlation-ID plumbing for kernel/op linking | Both | 3 | | |
| 15.6 | Traces viewable in TensorBoard / Chrome tracing | Both | 3 | | |
| 15.7 | Memory profiling | Both | 3 | | |

---

## 16. DataLoader Integration -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 16.1 | `DataLoader(pin_memory=True)` works | 2 | | |
| 16.2 | `tensor.to(device, non_blocking=True)` overlaps with compute | 2 | | |
| 16.3 | Multi-worker DataLoader doesn't deadlock with device | 2 | | |

---

## 17. Additional PyTorch APIs -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 17.1 | Quantization -- quantized dtype support and op kernels | 2 | | |
| 17.2 | ONNX Export -- `torch.onnx.export` with device tensors | 3 | | |
| 17.3 | Tensor/Storage Customization | 3 | | |

---

## 18. Testing & Validation -- Level: **2**

### 18.1 Device-Generic Test Framework

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 18.1.1 | OpInfo-based operator compliance tests pass | Both | 1 | | |
| 18.1.2 | `instantiate_device_type_tests` runs on your device | PU1 | 2 | | |
| 18.1.3 | `PrivateUse1TestBase` auto-included in test framework | PU1 | 2 | | |
| 18.1.4 | Common device dtype tests pass | Both | 2 | | |

### 18.2 Module-Level Tests

| Test Area | Priority | Points | Notes |
|-----------|----------|--------|-------|
| Tensor creation & transfer | 2 | | |
| Memory allocation & cleanup | 2 | | |
| AMP / autocast | 2 | | |
| Backward pass / autograd | 2 | | |
| torch.compile correctness | 2 | | |
| Distributed collectives | 2 | | |
| Operator coverage | 2 | | |
| Stream correctness | 3 | | |
| Event correctness | 3 | | |
| Storage operations | 3 | | |
| RNG reproducibility | 3 | | |
| Profiler output | 3 | | |
| Serialization round-trip | 3 | | |

### 18.3 CI Integration

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 18.3.1 | CI pipeline runs device-generic test suite | 2 | | |
| 18.3.2 | Regression tests on each upstream PyTorch update | 2 | | |
| 18.3.3 | Integration with `pytorch-integration-tests` framework | 3 | | |

---

## 19. Dtype Support Matrix -- Level: **2**

| Dtype | Priority | Points | Compute | Storage | AMP Target | CUDA Parity | Notes |
|-------|----------|--------|---------|---------|------------|-------------|-------|
| `float16` | 1 | | `[ ]` | `[ ]` | `[ ]` | `[ ]` | |
| `bfloat16` | 1 | | `[ ]` | `[ ]` | `[ ]` | `[ ]` | |
| `float32` | 1 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `int8` | 2 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `int32` | 2 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `int64` | 2 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `bool` | 2 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `float64` | 3 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `int16` | 3 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `uint8` | 3 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `complex64` | 3 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `complex128` | 3 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `float8_e4m3fn` | 3 | | `[ ]` | `[ ]` | -- | `[ ]` | |
| `float8_e5m2` | 3 | | `[ ]` | `[ ]` | -- | `[ ]` | |

---

## 20. Numerical Accuracy -- Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 20.1 | float32 ops match CUDA within 1e-5 atol | 1 | | |
| 20.2 | float16 ops match CUDA within 1e-3 atol | 1 | | |
| 20.3 | Loss convergence curve matches CUDA on reference models | 1 | | |
| 20.4 | bfloat16 ops match CUDA within 1e-2 atol | 2 | | |
| 20.5 | Reduction ops handle large tensors without overflow | 2 | | |
| 20.6 | Matmul results are deterministic (same input -> same output) | 2 | | |

---

## 21. Ecosystem Compatibility -- Level: **3**

| Library | Priority | Points | Version Tested | Notes |
|---------|----------|--------|----------------|-------|
| HuggingFace transformers | 1 | | | |
| HuggingFace accelerate | 2 | | | |
| DeepSpeed | 2 | | | |
| torchvision | 2 | | | |
| PEFT (LoRA, QLoRA) | 2 | | | |
| Flash Attention | 2 | | | |
| triton (if applicable) | 2 | | | |
| vLLM / TGI (inference) | 2 | | | |
| Megatron-LM | 3 | | | |
| torchaudio | 3 | | | |
| torchtune | 3 | | | |
| bitsandbytes | 3 | | | |

---

## 22. Registration API Quick Reference

### PrivateUse1 Path

| What | API / Macro | When |
|------|-------------|------|
| Backend name | `torch.utils.rename_privateuse1_backend("<name>")` | Python init |
| Device module | `torch._register_device_module("<name>", mod)` | Python init |
| Method generation | `generate_methods_for_privateuse1_backend("<name>")` | Python init |
| Hooks | `RegisterPrivateUse1HooksInterface(hooks_ptr)` | C++ static init |
| Guard | `C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardClass)` | C++ static init |
| Generator | `REGISTER_GENERATOR_PRIVATEUSE1(GenClass)` | C++ static init |
| Kernels | `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` | C++ per-op |
| Autograd | `TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m)` | C++ per-op |
| AMP | `TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m)` | C++ per-op |
| Serialization | `TensorBackendMetaRegistry(PrivateUse1, ...)` | C++ static init |
| Profiler | `REGISTER_PRIVATEUSE1_PROFILER(StubClass)` | C++ static init |
| ProcessGroup | `torch.distributed.Backend.register_backend(...)` | Python init |
| torch.compile | `register_interface_for_device("<name>", DevInterface)` | Python init |
| Autoload | `entry_points = {"torch.backends": ["<name> = ..."]}` | setup.py |

### Fork Path

| What | API / Mechanism | Where |
|------|----------------|-------|
| Device type | Add to `c10::DeviceType` enum | `c10/core/DeviceType.h` |
| Dispatch key | Add to `DispatchKey` enum | `c10/core/DispatchKey.h` |
| Guard | `C10_REGISTER_GUARD_IMPL(<Key>, GuardClass)` | C++ static init |
| Kernels | `TORCH_LIBRARY_IMPL(aten, <Key>, m)` | C++ per-op |
| Autograd | `TORCH_LIBRARY_IMPL(aten, Autograd<Key>, m)` | C++ per-op |
| AMP | `TORCH_LIBRARY_IMPL(aten, Autocast<Key>, m)` | C++ per-op |
| native_functions | Add dispatch entries to `native_functions.yaml` | ATen codegen |

---

## Sources

- [Accelerator Integration Guide][accel-guide]
- [OpenReg Reference Implementation][openreg]
- [Tracking Issue #158917][issue]
- [PrivateUse1 Tutorial][pu1-tutorial]
- [PyTorch Multi-Device Blog Post][blog]
- [ProcessGroup Extension Tutorial][pg-tutorial]
- [Running and Writing Tests Wiki][tests-wiki]
- [Accelerator Integration Working Group][accel-wg]

[accel-guide]: https://docs.pytorch.org/docs/stable/accelerator/index.html
[openreg]: https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
[issue]: https://github.com/pytorch/pytorch/issues/158917
[pu1-tutorial]: https://docs.pytorch.org/tutorials/advanced/privateuseone.html
[blog]: https://pytorch.org/blog/pt-multidevice-integration/
[pg-tutorial]: https://docs.pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html
[tests-wiki]: https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
[accel-wg]: https://github.com/pytorch-fdn/accelerator-integration-wg
[pit]: https://github.com/cosdt/pytorch-integration-tests
