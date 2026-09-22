# PyTorch Accelerator Integration Readiness — RBLN

| Field | Value |
|-------|-------|
| **Backend** | RBLN (Rebellions NPU) — `torch-rbln` |
| **Backend version** | v0.4.0 (evaluated at `6b7cee8`, dev branch) |
| **Integration path** | PrivateUse1 (out-of-tree) |
| **Dispatch key** | `PrivateUse1` |
| **Evaluation date** | 2026-09-01 |
| **Evaluator** | agent (torch-air `/torch-accelerator-readiness`) |
| **Source** | https://github.com/RBLN-SW/torch-rbln (local working copy) |
| **PyTorch base** | 2.11.0+cpu (pinned in `pyproject.toml`) |

---

## Readiness Score & Summary

### Executive Summary

**Overall Readiness: 69.5%**  (previous run: 64.4% at `299488a`, 2026-08-11)

**What moved since the last run** — 24 commits, +3,368/−756 lines across `c10/`, `aten/`,
`torch_rbln/`, and +4,026 lines of tests.

| Section | Was | Now | Why |
|---------|-----|-----|-----|
| Streams & Events | 33.3% | **85.4%** | #154 added a real `torch.Stream` / `torch.Event` surface over a 32-slot per-device pool |
| Accelerator Hooks [PU1] | 54.5% | **88.8%** | `deviceCount` / `getCurrentDevice` / `setCurrentDevice` / `exchangeDevice` / `maybeExchangeDevice` are now overridden |
| Device Guard | 94.4% | **100.0%** | stream methods are no longer default-only |
| Python Frontend | 88.6% | **91.4%** | `torch.accelerator.current_stream()` / `set_stream()` are real |
| Testing & Validation | 83.0% | **86.6%** | stream suites, a PU1 contract-conformance suite, a PyTorch-nightly CI job |
| Profiler | 61.8% | **67.6%** | #194 wires run → triggered-op correlation as kineto flow arrows |

Nothing regressed. The remaining Level-1/2 gaps are the same ones: autograd, AMP, `DeviceInterface`,
serialization hooks, and the RNG generator.

**Notable insights**

- RBLN is an **inference-first eager backend built on `torch.compile`**: every dispatched
  ATen op is lowered per-call through a Dynamo/rebel compile, cached in a C++ warm-runtime
  cache, and short-circuited by a C++ dispatch shim that pre-checks dtype/contiguity before
  ever entering Python. This is architecturally unlike XPU/NPU-style kernel-library backends,
  and it is the source of most of the upstreamable infrastructure below.
- The device is **fp16/bf16-only for on-device residency**. Everything else (int metadata,
  fp32) is accepted under `device="rbln"` but is host-backed and runs through the CPU
  fallback. Consequently the CPU-fallback path is a *first-class hot path*, not an escape
  hatch — and it has been optimized far past what `at::native::cpu_fallback` does today.
- The backend ships a **hidden-overhead explainer** (`torch.rbln.explain()`, `docs/EXPLAIN.md`)
  that has no equivalent anywhere in PyTorch core. It is the single most valuable upstream
  candidate in this evaluation.
- New this run: `test/rbln/test_privateuse1_contract.py` (872 lines) pins **one upstream clause
  per test, each citing its source in torch**, and runs every probe in a fresh subprocess across
  five degraded-environment scenarios. It is the second-most valuable upstream candidate here —
  PyTorch has no device-generic conformance suite for the paths core calls into a backend from.
- Several gaps are *deliberate*, not missing work: AMP advertises an **empty** dtype catalog
  on purpose so `torch.autocast("rbln")` disables itself with a warning rather than crashing
  on a missing `AutocastPrivateUse1` kernel.

**Key strengths**

- Level-1 plumbing is essentially complete: registration, device guard (including events and
  `getDeviceCapability`), allocator (`c10::DeviceAllocator` with real stats), pinned-host
  allocator, autoload entry point — all 94–100%.
- Distributed is unusually mature for a beta backend: a full `ProcessGroupRBLN` over RCCL with
  allreduce/broadcast/allgather/reduce_scatter/barrier/P2P, a Gloo fallback for non-fp16
  collectives, TP and PP tested, and RDMA IP auto-discovery.
- Test infrastructure is strong: PyTorch's `test_ops.py` OpInfo suite is ported and run with
  `instantiate_device_type_tests(only_for="privateuse1")`, plus C++ gtest suites, a walkthrough
  suite, and real-model logits-vs-fp32 tests on Llama-3.2-1B/3B, Qwen2.5, EXAONE-3.5.

**Key gaps**

- **Accelerator hooks: mostly closed (88.8%), one row left.** The five device-selection hooks
  were overridden this cycle. `getDefaultGenerator` is the last mandatory hook still inheriting
  `FAIL_PRIVATEUSE1HOOKS_FUNC`, and `init()` still takes the inherited no-op.
- **Autograd (38.6%) and Numerical Accuracy (38.9%)** are the weakest Level-1/2 sections.
  Backward kernels exist only for `linear`, `silu`, `_softmax`, and SDPA; there is no
  training test, no gradient-correctness comparison, and no convergence validation.
- **`torch.compile` integration is partial (47.1%).** `torch.compile(backend="rbln")` works and
  is the backend's own execution engine, but no `DeviceInterface` is registered via
  `register_interface_for_device("rbln", ...)`, so Inductor-side device-generic code paths do
  not see the device.
- **No serialization hooks (40.6%).** No `TensorBackendMetaRegistry` registration and no
  save/load round-trip test in the repo.
- **RNG is a stub (24.0%).** `RBLNGeneratorImpl::set_state` is a no-op and `get_state` returns
  uninitialized bytes; there is no `torch.rbln.manual_seed`, no fork handler, and every random
  op falls back to CPU.

**Upstream candidates**: **23 features** identified as generic or near-generic enough for
PyTorch core (19 carried forward, 4 new this run), plus 2 asks on core's own hook surface.

### Section Scores

| Section | Max Pts | Earned | Percentage |
|---------|---------|--------|------------|
| **Level 1** | | | |
| Device Registration & Management | 18.0 | 17.0 | 94.4% |
| Operator Registration | 50.0 | 35.0 | 70.0% |
| Autograd | 11.7 | 4.5 | 38.6% |
| Device Guard | 9.0 | 9.0 | 100.0% |
| Accelerator Hooks [PU1] | 22.3 | 19.8 | 88.8% |
| Memory & Allocator | 11.0 | 10.3 | 93.9% |
| **Level 2** | | | |
| Serialization & Model Portability | 10.7 | 4.3 | 40.6% |
| Python Frontend & Device-Agnostic APIs | 11.7 | 10.7 | 91.4% |
| AMP | 10.0 | 2.0 | 20.0% |
| torch.compile / Inductor | 11.3 | 5.3 | 47.1% |
| Distributed Training | 20.7 | 17.2 | 83.1% |
| Dtype Support Matrix | 14.7 | 8.0 | 54.5% |
| Numerical Accuracy | 9.0 | 3.5 | 38.9% |
| Testing & Validation | 18.7 | 16.2 | 86.6% |
| **Level 3** | | | |
| Streams & Events | 8.0 | 6.8 | 85.4% |
| RNG & Generator | 8.3 | 2.0 | 24.0% |
| Autoload [PU1] | 3.0 | 3.0 | 100.0% |
| DataLoader Integration | 3.0 | 2.5 | 83.3% |
| Profiler | 5.7 | 3.8 | 67.6% |
| Ecosystem Compatibility | 9.7 | 3.0 | 31.0% |
| Additional PyTorch APIs | 2.3 | 0.7 | 28.6% |

**Readiness**: **69.5 %**

---

### Upstream Candidates (Advisory)

> Features discovered in this backend that are generic enough to benefit all
> PU1/accelerator backends if upstreamed to PyTorch core.
> Advisory only — does not affect the readiness score.

**Classification key**

- **Generic** — no backend-specific references; ready to upstream into core as-is for any backend
- **Needs Abstraction** — solves a problem every backend faces, but the implementation references
  backend-specific internals; the upstream path is to define an interface/hook in core

**Summary**

| # | Feature | Category | Classification |
|---|---------|----------|----------------|
| 1 | Hidden-overhead explainer (`explain()`) | 24.2 | Needs Abstraction |
| 2 | Per-schema arg-kind + alias cache in boxed CPU fallback | 24.1 | Generic |
| 3 | Skip the D2H copy of pure `out=` arguments | 24.1 | Generic |
| 4 | `Tensor?` (optional-tensor) arg classification in CPU fallback | 24.1 | Generic |
| 5 | Zero-copy host borrow of device memory for CPU fallback | 24.1 | Needs Abstraction |
| 6 | Per-op / per-reason CPU-fallback attribution counters | 24.1 | Generic |
| 7 | Per-op host fast-path registry (`CPUFastPathRegistry`) | 24.1 | Generic |
| 8 | View-recipe reconstruction from strides (BFS + simulator) | 24.3 | Generic |
| 9 | Nested-compile chromium event-state isolation | 24.6 | Generic (core bug) |
| 10 | Dynamo guard-build `repr(tensor)` materialization | 24.6 | Generic (backport) |
| 11 | Reentrancy guard for `torch.compile` inside an ATen kernel | 24.6 | Generic |
| 12 | Compiled-callable wrapper attribute/descriptor semantics | 24.6 | Generic |
| 13 | Warm-cache key must include strides + storage_offset | 24.6 | Generic (contract) |
| 14 | `empty_cache()` auxiliary-cache registration hook | 24.4 | Generic |
| 15 | Host-backed "dummy device" compile-only mode | 24.4 | Needs Abstraction |
| 16 | `set_device_layout_like()` allocator layout affinity | 24.4 | Needs Abstraction |
| 17 | Logical↔physical device topology + `device_summary()` | 24.5 | Needs Abstraction |
| 18 | Device-index normalization (incl. `torch.device` int8 wraparound) | 24.5 | Generic |
| 19 | Standard fallback-disable env contract | 24.7 | Generic |

Plus two **hook-surface asks** on core itself (24.7).

---

#### 24.1 CPU Fallback Infrastructure

RBLN's `cpu_fallback_rbln` is a heavily reworked descendant of `at::native::cpu_fallback`.
Because fp32/int dtypes are host-backed on this device, the fallback is a hot path, and the
optimizations found there are ones every PU1 backend inherits for free if upstreamed.

##### 2. Per-schema arg-kind + alias cache

The boxed fallback walks `schema.arguments()` on **every** dispatch to classify each argument
(Tensor / TensorList / OptionalTensorList / Device) and read its `alias_info()`. RBLN caches
that classification keyed on the `const c10::FunctionSchema*` (stable for process lifetime),
under a `shared_mutex`.

**Classification**: Generic
**Relevant files**: `aten/src/ATen/native/rbln/RBLNCPUFallback.cpp:41-109`
(`CpuFbArgKind`, `CpuFbSchemaInfo`, `get_or_populate_schema_info`)
**Current state in PyTorch**: `aten/src/ATen/native/CPUFallback.cpp` re-derives this per call.
**Motivation**: Zero backend references. Their comment measures LLaMA-1B eager hitting this
10,066× across fewer than 10 distinct ops. Every out-of-tree backend that uses the boxed CPU
fallback pays the same repeated schema walk.

##### 3. Skip the D2H copy of pure `out=` arguments

Core's fallback copies every write-alias tensor device→host before running the CPU kernel —
including a kwarg-only `out=` whose contents the kernel is about to overwrite entirely. RBLN
detects the "pure out" case (`kwarg_only && name == "out" && is_write_alias`) and hands the
kernel fresh empty CPU storage instead, then writes back. The narrowness is deliberate:
in-place ops like `add_(self, other)` have a write-alias `self` the kernel also *reads*.

**Classification**: Generic
**Relevant files**: `aten/src/ATen/native/rbln/RBLNCPUFallback.cpp:104-105, 379-419`
(`is_pure_out`), plus the grow-retry at `:484-544`
**Current state in PyTorch**: Nothing; core D2H-copies the soon-to-be-discarded contents.
**Motivation**: Their measurement: ~22% of `cpu_fallback_rbln` host time on LLaMA-1B eager
(`mean`/`rsqrt`/`pow.out` each hit the slow path 1,122×). Pure bandwidth saved, no semantics
changed, applies to XLA/openreg/NPU/MTIA identically. Ship with the accompanying
"grow-retry on `not resizable`" mechanic, which is the only correctness wrinkle.

##### 4. `Tensor?` (optional-tensor) argument classification

Core's fallback classifies `Tensor`, `Tensor[]`, and `Tensor?[]` but an `OptionalType<Tensor>`
positional (e.g. `linear`'s `bias`) can stay on the stack as a device tensor under CPU
dispatch and fail at the next hop. RBLN classifies it explicitly and only stages the
has-value case.

**Classification**: Generic (bug fix)
**Relevant files**: `aten/src/ATen/native/rbln/RBLNCPUFallback.cpp:38-40, 96-98, 313-321`
**Current state in PyTorch**: Not handled.
**Motivation**: Straight correctness fix for any backend that falls back on an op with a
`Tensor?` argument. Has a matching in-repo test (`test_internal_op_utils.py:765` —
"linear fp32 with bias through cpu fallback").

##### 5. Zero-copy host borrow of device memory for the CPU fallback

Where core copies device→host, RBLN asks the runtime for a host pointer into the *existing*
device allocation (`try_borrow_host_ptr` / `try_acquire_host_ptr_for_overwrite`), wraps it with
`at::from_blob`, lets the CPU kernel write straight into it, and returns the borrow with an
`updated` flag so the next device consumer triggers a lazy H2D sync instead of an eager copy.
Guards are conservative and well-documented (contiguous, `storage_offset == 0`, non-zero bytes),
with an RAII release guard so a throwing kernel cannot strand a borrow.

**Classification**: Needs Abstraction
**Relevant files**: `aten/src/ATen/native/rbln/RBLNCPUFallback.cpp:187-262` (`borrow_rbln_as_cpu`,
`borrow_rbln_list_as_cpu`), `:446-475` (`BorrowReleaseGuard`), `:546-670` (write-back)
**Current state in PyTorch**: Nothing. Core assumes device memory is unreachable from the host.
**Motivation**: Any backend with host-mapped, unified, or virtual-memory-backed device
allocations (RBLN, openreg, CPU-emulated backends, several NPU stacks) can serve the entire
fallback without a single DMA. Upstream path: add optional
`AcceleratorHooksInterface::borrowHostPtr(void*, size_t) -> optional<BorrowedHostPtr>` /
`returnBorrowed(id, bool updated)` (default `nullopt`, i.e. today's copy path), and teach
`at::native::cpu_fallback` to use it when present. Backends that cannot map host memory are
unaffected.

##### 6. Per-op / per-reason CPU-fallback attribution

RBLN records, on the already-slow fallback branch only (one relaxed atomic via a cached
pointer), which op fell back, *why* (`dtype-not-fp16` / `nan-inf` / `all-scalar`), which
op recompiled, and — the useful one — which fallback ops have **no fast-path handler**
(`cpu_fallback_unaccelerated`, i.e. the optimization work list).

**Classification**: Generic
**Relevant files**: `torch_rbln/csrc/rbln/DispatchShim.h:50-72`
(`diag_dump_fallback_by_op`, `diag_dump_recompile_by_op`, `diag_dump_fallback_reasons`)
**Current state in PyTorch**: Core's `cpu_fallback` has no counters at all. Users discover
fallbacks only by `TORCH_WARN` spam or by profiling.
**Motivation**: "Which ops fell back and how often" is the single most-asked question by every
out-of-tree-backend user. The ON==OFF design (counters live only on the slow branch, reads are
lazy) is what makes it shippable as always-on in core.

##### 7. Per-op host fast-path registry

A self-registering table of host micro-kernels that bypass `redispatchBoxed(CPU)`'s
TensorIterator + boxed dispatcher for individual ops. Adding a fast path is one new `.cpp`
under `fast_paths/` with a `REGISTER_RBLN_CPU_FAST_PATH("aten::rsqrt.out", fn)` — no central
edit. Handler returns `false` to decline and fall through.

**Classification**: Generic
**Relevant files**: `aten/src/ATen/native/rbln/RBLNCPUFastPaths.h` (whole file),
`aten/src/ATen/native/rbln/fast_paths/*.cpp` (4 handlers)
**Current state in PyTorch**: Nothing equivalent next to `CPUFallback.h`.
**Motivation**: The mechanism has zero RBLN references — it is a name→function registry with a
`FunctionSchema*` cache. Any backend with a "some dtypes are host-backed" story wants it.

---

#### 24.2 Profiler / Explain Framework

##### 1. `torch.rbln.explain()` — the hidden-overhead explainer

The flagship candidate. A region context manager that answers *"what did my code make the
backend do that I never asked for and cannot see?"* — host bounces, CPU fallbacks, recompiles,
device↔host crossings — with a `torch.profiler`-style table, a cause/remedy note per row, a
`[clean]`/`[overhead: N signals]` factual marker (deliberately **not** a severity grade),
opt-in Python call-site capture, `diff()` / `explain_steady()` to separate one-time from
recurring cost, and stable `dump()` / `verdict()` dicts for CI gating.

Design properties that make it upstreamable rather than a vendor toy:

- **ON == OFF when idle.** Counters sit only on branches that already pay a host round-trip;
  reads are lazy at `report()` time.
- **Facts, not grades.** The marker says *what fired*, never how bad it is; cost is read from
  the physical-transfer line the user judges.
- **Context vs finding separation.** Host oversubscription, runtime share, and device-memory
  high-water are explicitly *context*, never marker drivers.
- **Honest `checked:` list.** When the runtime cannot expose a counter, the signal drops off
  the "checked" list rather than silently reporting clean.

**Classification**: Needs Abstraction
**Relevant files**: `torch_rbln/profiler.py` (1,232 lines), `docs/EXPLAIN.md` (the design doc —
worth reading in full as the upstream RFC), `c10/rbln/RBLNProfiler.h` (the counter contract and
its scope-decision rationale), `torch_rbln/csrc/rbln/DispatchShim.h:50-83` (the diag surface)
**Current state in PyTorch**: `torch._dynamo.explain()` covers graph breaks only.
`torch.profiler` covers timing only. Nothing covers hidden host overhead. XLA has
`torch_xla.debug.metrics` (raw counters, no attribution or remedy); MPS has nothing; JAX has
`transfer_guard` (a hard error, not an explainer). Every backend's users hit this exact
problem and every backend re-invents a partial answer.
**Motivation**: Upstream shape: `torch/accelerator/explain.py` holding the region manager,
report renderer, diff, and verdict; plus a small c10 registry
(`register_hidden_overhead_signal(name, reader)`) that backends feed with
`(count, bytes, note)` tuples. The report format, the marker semantics, the context/finding
split, the `with_stack` capture, and the CI-gating dict are all backend-agnostic. Only the
signal *sources* are per-backend. Two sub-pieces are independently generic:
**host-oversubscription detection** (N threads on M cores inflates every host op — they measured
~5× on a one-core-pinned vLLM worker) and the **non-overlapping-region counter contract**.

---

#### 24.3 View-Aware Dispatch

##### 8. View-recipe reconstruction from strides

Given an arbitrary strided tensor, recover a chain of view steps
(`permute` / `narrow` / `select` / `expand` / `squeeze` / `unsqueeze` / `reshape`) that produces
it from a contiguous, offset-0 base — found by bounded BFS (`max_steps=4`) and **verified** by
replaying the candidate recipe through a pure metadata simulator, so a wrong recipe can never
be applied. Fast paths for the common single-step pure-permute case; a cache with hit/miss
stats and an explicit reset.

**Classification**: Generic (with one pluggable policy hook)
**Relevant files**: `torch_rbln/_internal/ops_utils.py:483-1300` —
`_detect_permute`, `_classify_single_step_view`, `_simulate_recipe`, `_gen_step_candidates`,
`_bfs_search_recipe`, `_detect_view_recipe_safe`, `view_recipe_cache_stats/reset`
**Current state in PyTorch**: Nothing. `torch/_prims_common` has stride helpers but no
inverse-view reconstruction. Functionalization goes the other direction (recording views as
they happen, which needs the ops to pass through it — not available at the dispatch boundary
where a backend receives an already-strided tensor).
**Motivation**: Every graph-compiler backend that cannot consume arbitrary strides faces the
same choice: materialize `.contiguous()` (a copy) or reconstruct the view chain and lower it.
XLA, rebel, and most NPU stacks all hit this. The algorithm is pure metadata algebra with
**zero** backend references. The one coupled piece is `_is_known_bad_pattern`
(`ops_utils.py:1070-1114`), a hard-coded list of recipe shapes the rebel compiler mislowers —
that becomes a registered policy predicate, not a reason to keep the whole thing downstream.
Also worth upstreaming as documentation: the BFS deliberately prefers an
algebraically-equivalent alternative ordering (`narrow → permute` over `permute → narrow`) when
a pattern is rejected, which is a generally useful trick.

---

#### 24.4 Memory Management Patterns

##### 14. `empty_cache()` auxiliary-cache registration hook

RBLN's `empty_cache()` means "let go of everything the caller is not still holding" — it drops
the device caching allocator **and** the C++ warm-runtime cache (which holds strong refs to
compiled runtimes and their device buffers) **and** the view-recipe cache (unbounded, one entry
per distinct view geometry). It also documents one cache it deliberately does *not* flush,
with the reasoning.

**Classification**: Generic
**Relevant files**: `torch_rbln/memory.py:97-137`
**Current state in PyTorch**: `torch.accelerator.empty_cache()` drains only the allocator. Every
backend that caches anything else re-implements the divergence, and users are surprised when
`empty_cache()` doesn't free what `memory_stats()` still reports.
**Motivation**: Add `torch.accelerator.register_cache_clear_callback(fn)` so
`torch.accelerator.empty_cache()` and backend-specific `empty_cache()` converge. Small,
uncontroversial, immediately useful to XLA/XPU/NPU.

##### 15. Host-backed "dummy device" compile-only mode

`RBLN_DUMMY_DEVICE` is a real device backend **without hardware**: it allows tensor
construction, host↔device copies, and building compiled artifacts with
`torch.compile(options={"mode": ["compile_only"]})`, but refuses to *execute* — through a
single policy gate that produces one clear error rather than silently returning zeros or
silently running on CPU.

**Classification**: Needs Abstraction
**Relevant files**: `torch_rbln/_internal/dummy_device.py` (whole file — 44 lines, one
`raise_if_dummy_execution()` gate consumed by both the compiled-graph path and the eager
fallback path)
**Current state in PyTorch**: meta/fake tensors model shapes but are not a device backend;
`torch_openreg` is a full CPU emulator (it *does* execute). Nothing occupies the middle.
**Motivation**: Hardware-less CI is a universal problem for accelerator backends. The generic
piece is the **pattern and its policy gate**: one place that owns "may I execute here?", one
error message, an explicit compile-only exemption. Worth upstreaming as a documented
convention plus a `torch.accelerator` helper, even if each backend supplies its own flag.

##### 16. `set_device_layout_like(target, ref)`

Make one whole device allocation adopt another's device-side layout without copying, so a
later device↔device copy between them stays on the fast path. Motivating case: make a
host→device staging buffer match a KV cache's layout so both the bulk upload and the per-slot
scatter are fast.

**Classification**: Needs Abstraction
**Relevant files**: `torch_rbln/memory.py:77-89`, `torch_rbln/csrc/rbln/Module.cpp:111-171`
(validation: whole base allocation, same device, same dtype)
**Current state in PyTorch**: Nothing. `memory_format` covers logical layout, not
allocator-side device layout.
**Motivation**: Layout-sensitive DMA is not RBLN-specific — any backend whose fast copy path
requires matching device-side tiling hits it. Upstream path is an optional allocator hook
plus a `torch.accelerator` wrapper; backends without layout affinity implement it as a no-op.

Also noted but lower-priority: **`offload()` / `release_offload_temp_storage()`**
(`torch_rbln/memory.py:252-300`) — a nesting-counted, thread-safe scoped switch that lets host
backing of device tensors page to disk, plus an explicit temp-file reclaim for shutdown paths
that may be killed. The paging is runtime-specific (Needs Abstraction), but the **API shape**
is a good template for the CPU-offload pattern that every serving stack reinvents.

---

##### 20. `huge_host_empty(nbytes)` — DMA-ready host slabs *(new this run)*

`torch.rbln.huge_host_empty()` returns a 2 MiB-aligned, prefaulted, zero-filled host buffer
that the device can DMA into with no staging copy. An ordinary CPU tensor is 64 B aligned, so
the runtime stages any non-page-aligned host address through a bounce buffer; and even an
aligned one pays a page fault per 4 KiB the first time the runtime resolves its host
addresses — which lands *inside the transfer*, not in setup. It also requests transparent huge
pages, best effort.

**Classification**: **Needs Abstraction** — the concept and the failure mode are universal; only
the provider (`rebel.host_memory.HugeHostBuffer`) is backend-specific.
**Relevant files**: `torch_rbln/memory.py:293-341`.
**Current state in PyTorch**: nothing equivalent. `pin_memory=True` gives page-locked memory
with no alignment or prefault guarantee, and `torch.empty` gives neither. Every backend whose
DMA path is alignment-sensitive re-solves this.
**Motivation**: `torch.accelerator.host_empty(nbytes, alignment=…, prefault=True)` backed by an
allocator hook would let backends declare their alignment requirement once instead of each
framework discovering the bounce-buffer cliff empirically. Note the range check in the RBLN
implementation is itself a bug report: past `sys.maxsize` the provider's round-up to alignment
wraps to zero, allocating nothing and then prefaulting the original size over it — a segfault
rather than an error.

##### 21. `bind_device_memory(tensor)` — materialize a lazy allocation for out-of-band consumers *(new this run)*

A device allocation reserves a virtual address and materializes physical memory on first use
*through a torch op*. A consumer that reads those buffers out of band — a collective library,
NVMe/direct-storage DMA — never runs such an op, so it finds nothing there.
`bind_device_memory()` forces the binding, flat and 1:1 with no dtype transform.

**Classification**: **Needs Abstraction**.
**Relevant files**: `torch_rbln/memory.py:260-290`, `torch_rbln/csrc/rbln/Module.cpp`,
and the consumer that proves the point: `ProcessGroupRBLN.cpp` was refactored onto it in the
same change (#213), replacing a raw `rbln_set_raw_memory_alloc()` call.
**Current state in PyTorch**: nothing. Core assumes an allocation is physically backed the
moment it is returned.
**Motivation**: any backend with lazy, virtual-memory-backed, or on-demand-paged allocations
faces exactly this when handing a buffer to a collective or a storage DMA path — and the
failure mode is the worst kind: silent garbage rather than an error. An optional
`Allocator::bindPhysical(ptr, nbytes)` hook with a no-op default would let `ProcessGroup` and
storage integrations call it unconditionally.

##### 22. Batched strided copy accumulators (H2V / V2H / V2V) *(expanded this run)*

A copy batch takes a *description* — `enqueue_strided(dst, src, inner_block_bytes, outer_sizes,
src_byte_strides, dst_byte_strides)` — and expands it row-major into per-entry slab copies
submitted as one bulk call, instead of staging a contiguous temporary. What was a single
`V2VBatch` at the last run is now a three-direction family (#197, #199) over a shared
`detail/RBLNCopyBatchImpl.h`, parameterized by which end of the copy pins the batch to a
device (`DeviceAnchor::{kBothEnds, kDstOnly, kSrcOnly}`).

**Classification**: **Needs Abstraction** — `submit()` is runtime-specific; the descriptor, the
row-major expansion, the device-homogeneity logic, and the contract are not.
**Relevant files**: `c10/rbln/RBLNHostBatch.{h,cpp}`, `c10/rbln/RBLNV2VBatch.{h,cpp}`,
`c10/rbln/detail/RBLNCopyBatchImpl.h`, `aten/src/ATen/native/rbln/RBLNCopy.cpp`;
tests `test/cpp/core/RBLNHostBatchTest.cpp` (523 lines), `test_foreach_copy_host.py`,
`test_strided_host_copy.py`.
**Current state in PyTorch**: nothing. `copy_` on a non-contiguous cross-device pair
materializes a contiguous staging tensor.
**Motivation**: "describe a strided copy, submit it as one batch" is the generic form of a
problem every backend with a bulk-copy entrypoint solves privately. The written contract is
worth upstreaming as much as the code: destinations must not overlap (the runtime does not
check), entries are unordered, a failed submit may have partially applied with no rollback, and
the destructor must never call the backend because a rejection during unwind terminates the
process.

#### 24.5 Device Topology & Utilities

##### 17. Logical↔physical device topology and `device_summary()`

One logical `rbln:N` may aggregate several physical NPUs, configured by environment
(`RBLN_DEVICE_MAP="[0,1],[2,3]"` or `RBLN_NPUS_PER_DEVICE=2`). The backend exposes a
`DeviceTopology` (per-logical-device physical IDs, aggregated flag, plus **unused** physical
devices left over by grouping constraints) and prints an ASCII summary table. The topology is
then consumed to auto-determine `num_devices` for `torch.compile` and to drive a
tensor-parallel failover.

**Classification**: Needs Abstraction
**Relevant files**: `c10/rbln/DeviceMappingManager.h` / `.cpp`,
`torch_rbln/device/device.py:294-395` (`_device_summary_string`, `device_summary`),
`torch_rbln/_internal/rsd_utils.py` (`get_physical_device_ids`, `auto_determine_num_devices`)
**Current state in PyTorch**: Nothing generic. NVIDIA MIG, Intel tiles/sub-devices, and AMD
partitions all present the same logical/physical split, and each exposes it through its own
vendor namespace.
**Motivation**: A `torch.accelerator.device_topology()` returning logical→physical entries plus
a `device_summary()` printer would let device-agnostic launchers (torchrun, accelerate, vLLM)
stop special-casing per vendor. The "unused physical devices" warning is a nice touch worth
keeping — silent under-utilization from grouping constraints is a real footgun.

##### 18. Device-index normalization

A single `_normalize_device()` that accepts `None` / `int` / `str` / `torch.device`, resolves
`None` and bare `"rbln"` to the current device, rejects non-matching device types, range-checks
the index — and handles two traps most backends get wrong:

- `torch.device`'s index field is an **`int8_t`**, so `torch.device("rbln", 256)` silently wraps
  to `rbln:0` and slips past a naive range check. RBLN reads the index from the raw int or the
  original string suffix instead of a constructed device's `.index`.
- `bool` is an `int` subclass, so `True` would silently mean `rbln:1`. Explicitly rejected.

**Classification**: Generic
**Relevant files**: `torch_rbln/memory.py:31-74`
**Current state in PyTorch**: Each backend hand-rolls this (`torch.cuda._get_device_index`,
`torch.xpu._get_device_index`, …) and the int8 wraparound is not defended anywhere.
**Motivation**: Belongs in `torch/accelerator/_utils.py` as a shared helper — or, better, the
wraparound should be fixed in `torch.device` itself so it raises instead of truncating.

Also noted: **`python -m torch_rbln.diagnose`** (`torch_rbln/_internal/env_diagnostic.py`,
`torch_rbln/_internal/env_utils.py:is_diagnose_mode`) — a diagnose-only import mode that skips
backend initialization so the diagnostic still runs when the native extension is broken. That
last property is the generic insight: `torch.utils.collect_env` cannot help you when the thing
that failed is the import.

---

#### 24.6 Compile Integration Patterns

##### 9. Nested-compile chromium event-state isolation — a core bug

Because RBLN lowers ATen ops through `torch.compile`, dispatching one during an outer compile's
`build_guards` re-enters Dynamo. The nested compile's exit resets the **shared** thread-local
chromium event stack, wiping the outer compile's events and crashing it with
*"No toplevel event active"*. RBLN works around it by swapping in throwaway containers for the
nested compile and restoring unconditionally.

**Classification**: Generic — this is a PyTorch core bug, not a backend concern
**Relevant files**: `torch_rbln/_internal/torch_compile_patch_helpers.py:47-87`
(`_isolate_chromium_event_state`)
**Current state in PyTorch**: `torch._dynamo.utils.get_chromium_event_logger()` keeps a single
flat TLS stack with no nesting discipline.
**Motivation**: Highest-value "just fix it upstream" item in this list — the fix belongs in
`torch._dynamo.utils`, is small, and removes a monkey patch from at least one backend. Any
backend or user code that triggers a nested `torch.compile` hits it.

##### 10. Dynamo guard-build `repr(tensor)` materialization

On torch 2.11/2.12, `GuardBuilder.id_match_unchecked` `repr`s the guarded *value*; for a device
tensor that forces a full device→host materialization during guard construction. Fixed on
`main` (reprs the type instead); RBLN backports it locally.

**Classification**: Generic — upstream ask is a **backport to the 2.11/2.12 release branches**
**Relevant files**: `torch_rbln/_internal/monkey_patches.py:149-192` (`patch_dynamo_guard_repr`,
explicitly a no-op on torch ≥ 2.13)
**Current state in PyTorch**: Fixed on main only.
**Motivation**: Silent, hard-to-attribute host transfers during guard build affect every
non-CPU backend on those releases.

##### 11. Reentrancy guard for `torch.compile` inside an ATen kernel

A thread-local depth counter so that any nested dispatch reaching the same op (from a compiled
graph calling back into `torch.add`, or from `print`/`repr` triggering dispatch) takes the CPU
fallback instead of recursing forever.

**Classification**: Generic
**Relevant files**: `torch_rbln/_internal/torch_compile_patch_helpers.py:27-44`,
consumed at `:446-456`
**Current state in PyTorch**: Nothing. Backends that compile inside a kernel must invent it.
**Motivation**: Fifteen lines, no backend references. Belongs next to the dispatch-mode
guards in `torch/_dynamo` or `torch/_C/_dispatch`.

##### 12. Compiled-callable wrapper attribute/descriptor semantics

`CompiledFunctionWrapper` transparently delegates attribute get/set/del to the wrapped
`nn.Module`, **re-wraps** bound methods that return the module so `compiled = compiled.eval()`
keeps the wrapper, routes an explicit `.forward(...)` through the compiled path, and implements
`__get__` so a decorated *function* binds its owner while a compiled module or already-bound
method does not get an owner injected as a spurious first argument.

**Classification**: Generic
**Relevant files**: `torch_rbln/_internal/torch_compile_patch_helpers.py:321-390`
**Current state in PyTorch**: `torch._dynamo.eval_frame.OptimizedModule` implements a subset;
several of these exact edge cases recur as upstream bug reports (`.eval()` unwrapping,
descriptor binding on decorated methods, `setattr` landing on the wrapper instead of the model).
**Motivation**: Not a backend concern at all — it is "how should a compiled-callable wrapper
behave". Worth upstreaming as hardening plus tests for `OptimizedModule`.

##### 13. Warm-cache key must include strides and storage_offset

A per-op compile cache keyed only on shape+dtype is **silently wrong**: a later call with a
non-contiguous or `storage_offset > 0` input of the same shape hits the contig-compiled entry,
and the runtime reads `numel * itemsize` contiguous bytes from a stride-aware pointer — wrong
values for `permute`/`transpose`/`expand`, wrong-offset reads for `narrow(dim, k>0, …)`.

**Classification**: Generic (contract / documentation, not code)
**Relevant files**: `torch_rbln/csrc/rbln/WarmCache.h:52-84` (`TensorProfile` and its rationale
comment, which is the best write-up of this hazard I found anywhere)
**Current state in PyTorch**: Undocumented. Backends writing eager-op compile caches keep
rediscovering it — RBLN hit it twice (input side, and an earlier `out=` mirror).
**Motivation**: Belongs in the PrivateUse1 tutorial / accelerator integration guide as an
explicit warning, ideally with a helper that builds a correct cache key.

Lower priority but noted: **`FailOnRecompileLimitHit` → reset-and-retry policy**
(`torch_compile_patch_helpers.py:438-444`) and the **auto-determined `num_devices` +
tensor-parallel failover** (`:137-294`) — the latter is coupled to RBLN's topology, but the
*policy shape* ("only failover when the device count was inferred, never when the caller pinned
it") is generic and worth documenting.

---

#### 24.7 New Hooks / API Extensions

##### 19. Standard fallback-disable environment contract

`TORCH_RBLN_DISABLE_FALLBACK` takes a comma-separated category list —
`all`, `compile_error`, `non_blocking_copy`, `strided_copy_error`, `unsupported_op` — and turns
each silent CPU fallback into a hard error with a message that names the category to remove.
Categories are honoured at both the C++ dispatcher registration and the Python compile-wrapper
layer.

**Classification**: Generic
**Relevant files**: `c10/rbln/RBLNFallbackConfig.h` / `.cpp`,
`aten/src/ATen/native/RBLNRegisterOps.cpp:118-124` (registration-time branch),
`torch_rbln/_internal/ops_utils.py:1811-1840` (Python-side category parsing)
**Current state in PyTorch**: Ad hoc and inconsistent — MPS has `PYTORCH_ENABLE_MPS_FALLBACK`
(a single boolean, inverted sense), XLA and NPU have their own variants, and core's
`cpu_fallback` has no switch at all.
**Motivation**: A core `TORCH_ACCELERATOR_DISABLE_FALLBACK={all,unsupported_op,…}` honoured by
`at::native::cpu_fallback` would give users one lever across every backend. This is exactly the
kind of cross-backend UX unification the accelerator WG exists for.

#### 24.8 Conformance Testing

*None of the seven template categories fits this one; it is listed separately rather than
forced into 24.7.*

##### 23. PrivateUse1 contract-conformance suite *(new this run)*

`test/rbln/test_privateuse1_contract.py` (872 lines) tests a proposition the template never
asks about: torch does not merely *offer* a backend module and hooks, it **calls into them**
from paths that have nothing to do with wanting an accelerator — `DataLoader(pin_memory=True)`,
`torch.load(map_location=…)`, importing `torch.testing._internal.common_utils`,
`torch._utils._get_available_device_type()`. A backend that raises, or that claims a device,
on any of those breaks callers who never asked for it.

Four properties make it a suite rather than a pile of tests:

- **One upstream clause per test, each citing its source in torch.** A torch upgrade or a new
  call site fails on the named clause, not on a downstream symptom.
- **Every probe runs in a fresh subprocess**, because the state involved is process-global and
  one-shot, and reports a structured verdict: `raised`, `ctx` (did it open a device context?),
  and `remap` (would a later visible-device remap still apply, or is the mapping frozen?).
- **A scenario matrix of degraded environments** — healthy, no-visible-device, bad device map,
  dummy device, dummy + bad map — because a contract clause has to hold on an unhealthy host
  too, which is exactly where these clauses get violated.
- **Unmet clauses are `strict=True` xfails naming the work that closes them**, so the suite
  states the roadmap and an unexpected pass is itself a failure.

**Classification**: **Generic**. The harness has no RBLN-specific logic beyond the
`RBLN_*` scenario variables and the `rbln-stat` context probe — both of which are one hook
each (`env vars for this backend`, `does this pid hold the device?`).
**Relevant files**: `test/rbln/test_privateuse1_contract.py`, `test/utils.py`.
**Current state in PyTorch**: nothing comparable. `instantiate_device_type_tests` covers
*operator* correctness on a device; there is no suite for the integration contract itself, and
`torch_openreg` demonstrates the happy path rather than asserting the clauses. The relevant
requirements are scattered across `torch/serialization.py` docstrings,
`torch/utils/backend_registration.py`, and `torch/random.py` comments.
**Motivation**: this is the single highest-leverage thing on the list for the accelerator WG.
Every out-of-tree backend rediscovers these clauses one production incident at a time — the
report's own §7 and §13 gaps were both found *by this suite*, not by reading the docs. Ported
into `torch/testing/_internal/` as a device-generic conformance suite, with the two hooks above
supplied per backend, it would turn a folklore contract into a runnable one.

##### Two asks on core's own hook surface

**(a) Default-delegate `AcceleratorHooksInterface` device methods to the registered guard impl.**
`deviceCount()` returns `0` by default and `setCurrentDevice`/`getCurrentDevice`/`exchangeDevice`/
`maybeExchangeDevice` throw — yet a correct backend has already implemented all of them on its
`DeviceGuardImplInterface` (`torch.accelerator.*` routes through the guard, per
`aten/src/ATen/DeviceAccelerator.cpp`).

**This ask got stronger since the last run, not weaker.** At `299488a`, `RBLNGuardImpl` implemented
all five and `RBLNHooksInterface` overrode none, so `at::accelerator::deviceCount()` reported zero
devices while `torch.rbln.device_count()` reported the real number, and
`torch._C._accelerator_hooks_get_current_device()` raised on a healthy RBLN host. The backend has
now hand-written all five overrides (`RBLNHooksInterface.cpp:41-64`) — each one a two-line forward
to the same `c10::rbln::` function the guard already calls. That is precisely the boilerplate the
default should have supplied: every PU1 backend must write it, the delegation target is fixed, and
until a backend notices, the failure is silent. Making `PrivateUse1HooksInterface` delegate to
`c10::impl::VirtualGuardImpl(kPrivateUse1)` by default would delete this boilerplate — and this
whole class of bug — for every PU1 backend.
**Relevant files**: `c10/rbln/RBLNHooksInterface.h:40-60` (the five hand-written overrides, with
comments naming the exact symptom each one fixed), `c10/rbln/impl/RBLNGuardImpl.h`,
`aten/src/ATen/detail/AcceleratorHooksInterface.h` (defaults),
`test/rbln/test_privateuse1_contract.py::test_hooks_device_count_agrees_with_module` (the
regression test a core default would make unnecessary).

**(b) Settle the semantics of `getDeviceCapability()`'s supported-dtypes.**
RBLN had to decide unilaterally that "capability dtypes" means *dtypes actually resident in
device memory* (fp16/bf16) rather than *dtypes with native op dispatch* — other dtypes accept
`device="rbln"` but are host-backed with zero device bytes.
**Relevant files**: `c10/rbln/RBLNSupportedDtypes.h:21-25` (`kCapabilityDtypes` and its rationale),
`c10/rbln/impl/RBLNGuardImpl.h:56-65`. Documenting the intended semantic in core would stop
backends diverging on what `torch.accelerator.get_device_capability()` promises.

(The standalone `V2VBatch` note from the previous run is superseded by candidate 22 above, which
covers the expanded three-direction family.)

---

## 0. Source Discovery & Integration Path Detection

| Source Discovery | Value | Notes |
|-----------------|-------|-------|
| **Source location** | `/Users/taesu.kim/Projects/docs/torch-rbln` | Local working copy, `dev` branch |
| **How found** | local | User-provided working directory |
| **Repository URL** | https://github.com/RBLN-SW/torch-rbln | Public; wheels on PyPI as `torch-rbln` |
| **Backend version** | v0.4.0 (`git describe`: `v0.4.0`) | Beta; `rebel-compiler>=0.11.2,<0.20.0` |

| Signal | Detected | Value | Notes |
|--------|----------|-------|-------|
| `rename_privateuse1_backend()` call found | `[x]` | `"rbln"` | `torch_rbln/__init__.py:53` |
| `entry_points` for `torch.backends` found | `[x]` | `autoload = "torch_rbln:torch_backends_entry_point"` | `pyproject.toml:26` |
| Custom entry in `DeviceType.h` | `[ ]` | — | No PyTorch fork |
| Custom entry in `DispatchKey.h` | `[ ]` | — | No PyTorch fork |
| Dispatch entries in `native_functions.yaml` | `[x]` | 49 funcs, `PrivateUse1:` dispatch | Own vendored YAML consumed by `tools/codegen`, **not** a core fork |
| Separate pip-installable package | `[x]` | `torch-rbln` | Depends on stock `torch==2.11.0+cpu` |
| Modifies PyTorch core source files | `[ ]` | — | Mirrors core paths (`aten/`, `c10/`) inside its own tree |
| **Detected path** | | **PU1** | |
| **Dispatch key** | | **PrivateUse1** | |

---

## 1. Device Registration & Management — Level: **1**

### 1.1 Backend Registration

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 1.1.1 | Backend name registered via `rename_privateuse1_backend("rbln")` | PU1 | 1 | 2 | `torch_rbln/__init__.py:53` |
| 1.1.2 | `torch._register_device_module("rbln", module)` | PU1 | 1 | 2 | `torch_rbln/__init__.py:58` → `torch_rbln.device` |
| 1.1.3 | Device type added to `c10::DeviceType` enum | Fork | 1 | N/A | Not a fork |
| 1.1.4 | Custom dispatch key registered in `DispatchKey.h` | Fork | 1 | N/A | Not a fork |
| 1.1.5 | `torch.device("rbln")` and `torch.device("rbln:0")` work | Both | 1 | 2 | `test/rbln/walkthrough/test_01_device_type.py:26,42` |
| 1.1.6 | `generate_methods_for_privateuse1_backend("rbln")` | PU1 | 2 | 0 | Never called — `Tensor.rbln()` / `Tensor.is_rbln` are not generated |

### 1.2 Device Management

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 1.2.1 | `deviceCount()` returns correct count | 1 | 2 | `RBLNGuardImpl::deviceCount`; `c10::rbln::get_device_count` |
| 1.2.2 | `setCurrentDevice()` / `getCurrentDevice()` | 1 | 2 | `torch_rbln/device/device.py:39,143`; guard `getDevice`/`setDevice` |
| 1.2.3 | `torch.rbln.is_available()` | 1 | 2 | `device.py:80` — `device_count() > 0 && runtime_available()` |
| 1.2.4 | `torch.rbln.device_count()` | 1 | 2 | `device.py:55` |
| 1.2.5 | `exchangeDevice()` / `maybeExchangeDevice()` | 2 | 2 | `device.py:200,219`; `RBLNGuardImpl::exchangeDevice` |
| 1.2.6 | Multi-device indexing (`rbln:0`, `rbln:1`, …) | 2 | 2 | `DeviceMappingManager`; `test/rbln/test_multi_device.py`, `test_device_mapping.py` |
| 1.2.7 | `torch.rbln.synchronize()` | 2 | 2 | `device.py:122`; `RBLNGuardImpl::synchronizeDevice` |

---

## 2. Accelerator Hooks **[PU1]** — Level: **1**

### Mandatory hooks (throw if not overridden)

| # | Hook | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 2.1 | `hasPrimaryContext(device_index)` | 1 | 2 | `RBLNHooksInterface.cpp:74` — CUDA parity, per-device, nothrow (logging removed so it cannot throw) |
| 2.2 | `getDefaultGenerator(device_index)` | 1 | 0 | Not overridden → inherits `FAIL_PRIVATEUSE1HOOKS_FUNC` (throws) |
| 2.3 | `getDeviceFromPtr(void* data)` | 1 | 2 | `RBLNHooksInterface.cpp:66` |
| 2.4 | `isBuilt()` | 1 | 2 | `RBLNHooksInterface.cpp:20` |
| 2.5 | `isAvailable()` | 1 | 2 | `RBLNHooksInterface.cpp:31` — same predicate Python's `is_available()` is bound to |
| 2.6 | `getNewGenerator(device_index)` | 2 | 2 | `RBLNHooksInterface.cpp:121` → `RBLNGeneratorImpl` |
| 2.7 | `getPinnedMemoryAllocator()` | 2 | 2 | `RBLNHooksInterface.cpp:113` → `RBLNPinnedAllocator` |
| 2.8 | `resizePrivateUse1Bytes(storage, newsize)` | 2 | 2 | `RBLNHooksInterface.cpp:82` — handles `new_nbytes == 0` |

### Hooks with safe defaults (override recommended)

| # | Hook | Default | Priority | Points | Notes |
|---|------|---------|----------|--------|-------|
| 2.9 | `deviceCount()` | returns 0 | 1 | 2 | **Now overridden** — `RBLNHooksInterface.cpp:41` → `get_device_count_nothrow()` (#192/#207) |
| 2.10 | `setCurrentDevice(device)` | throws | 1 | 2 | **Now overridden** — `RBLNHooksInterface.cpp:51` |
| 2.11 | `getCurrentDevice()` | throws | 1 | 2 | **Now overridden** — `RBLNHooksInterface.cpp:45`; pinned by `test_privateuse1_contract.py::test_hooks_get_current_device_is_implemented` |
| 2.12 | `init()` | no-op | 2 | 1 | Inherits PU1 default no-op; no explicit lazy-init hook |
| 2.13 | `exchangeDevice(device)` | throws | 2 | 2 | **Now overridden** — `RBLNHooksInterface.cpp:56` |
| 2.14 | `isPinnedPtr(data)` | returns false | 3 | 2 | `RBLNHooksInterface.cpp:96` → real pinned-range check |
| 2.15 | `maybeExchangeDevice(device)` | throws | 3 | 2 | **Now overridden** — `RBLNHooksInterface.cpp:61`; equal to `exchangeDevice` (selection creates no context) |

---

## 3. Device Guard — Level: **1**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 3.1 | `c10::impl::DeviceGuardImpl` subclass implemented | Both | 1 | 2 | `c10/rbln/impl/RBLNGuardImpl.h` |
| 3.2 | Registered via `C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardClass)` | PU1 | 1 | 2 | `RBLNGuardImpl.cpp:14` |
| 3.3 | Registered via `C10_REGISTER_GUARD_IMPL(<Key>, GuardClass)` | Fork | 1 | N/A | Not a fork |
| 3.4 | `getDevice()` / `setDevice()` / `uncheckedSetDevice()` | Both | 1 | 2 | `RBLNGuardImpl.h:33-48` |
| 3.5 | Guard saves/restores device+stream on scope exit | Both | 1 | 2 | `exchangeDevice`/`exchangeStream` implemented |
| 3.6 | `getStream()` / `setStream()` / `exchangeStream()` | Both | 2 | 2 | Real multi-stream support (#154): `getStream`/`getDefaultStream`/`getNewStream`/`getStreamFromGlobalPool`/`exchangeStream`/`queryStream`/`synchronizeStream`, `RBLNGuardImpl.cpp:119-155` |

Also present beyond the checklist: `getDeviceCapability()` (supported-dtype reporting) and a
full event surface — `record` / `block` / `queryEvent` / `synchronizeEvent` / `destroyEvent`,
mapped onto a whole-device drain because the UMD exposes a single in-order copy queue.

---

## 4. Autoload **[PU1]** — Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 4.1 | `entry_points` registered (group: `torch.backends`) | 2 | 2 | `pyproject.toml:25-26` |
| 4.2 | `_autoload()` callable initializes backend on `import torch` | 2 | 2 | `torch_backends_entry_point()`, idempotent via `status` guard |
| 4.3 | `torch.device("rbln")` resolves without explicit import | 2 | 2 | Autoload path; `test/rbln/test_import_rbln_devices_seal.py` |

---

## 5. Memory & Allocator — Level: **1**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 5.1 | Device allocator (`c10::Allocator` subclass) | 1 | 2 | `RBLNAllocator : c10::DeviceAllocator`, `c10/rbln/RBLNAllocator.cpp` |
| 5.2 | Allocator registered globally | 1 | 2 | `REGISTER_ALLOCATOR(c10::kPrivateUse1, &allocator)` |
| 5.3 | Host/pinned memory allocator | 2 | 2 | `c10/rbln/RBLNPinnedAllocator.{h,cpp}` — page-aligned + best-effort `mlock` |
| 5.4 | `torch.rbln.memory_allocated()` | 2 | 2 | `memory.py:164`, reads `allocated.current` |
| 5.5 | `torch.rbln.empty_cache()` | 2 | 2 | `memory.py:97` — also drops warm-runtime and view-recipe caches |
| 5.6 | OOM produces a clear error message (not a segfault) | 2 | 2 | `RBLNFunctions.cpp:482,493`; pinned OOM at `RBLNPinnedAllocator.cpp:52` |
| 5.7 | Pinned memory (`pin_memory=True` in DataLoader) | 2 | 2 | `test/rbln/test_pin_memory.py`; `isPinnedPtr` wired |
| 5.8 | `torch.rbln.max_memory_allocated()` | 3 | 2 | `memory.py:178` |
| 5.9 | `torch.rbln.memory_reserved()` | 3 | 2 | `memory.py:192`; `getDeviceStats` override |
| 5.10 | `memory_summary()` or equivalent | 3 | 0 | Absent. `device_summary()` is topology, not memory |

Beyond the checklist: `offload()` / `release_offload_temp_storage()` (file offloading),
`set_device_layout_like()`, `reset_accumulated_memory_stats()`, `_mark_zeros()`
(zero-init without host allocation, for KV caches), and two APIs new since the last run
(#213): `bind_device_memory()` — materialize a lazy device allocation so an out-of-band
consumer (RCCL, NVMe DMA) that never runs a torch op finds physical memory behind the
virtual address — and `huge_host_empty()` — a 2 MiB-aligned, prefaulted host slab that
avoids both the runtime's bounce-staging of unaligned host addresses and the per-4 KiB page
fault that would otherwise land inside the transfer. `ProcessGroupRBLN` was refactored onto
`bind_device_memory()` in the same change, which is the clearest evidence that this is the
general "out-of-band consumer" problem and not a one-off.

---

## 6. Streams & Events — Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 6.1 | Stream implementation (pool management, priority) | 2 | 1 | Fixed 32-slot per-device round-robin pool (`RBLNFunctions.cpp:855-905`), filled lazily; **priorities do not exist** — accepted and ignored, `priority_range()` is `(0, 0)` |
| 6.2 | `torch.rbln.Stream` class functional | 2 | 2 | `torch_rbln/device/streams.py:36` — `torch.Stream` subclass, `torch.cuda.Stream` parity; `test/rbln/test_stream.py` (19 tests) |
| 6.3 | `torch.rbln.current_stream()` / `set_stream()` | 2 | 2 | `streams.py:112,120,128`; plus `default_stream()`, `stream()` / `StreamContext` |
| 6.4 | `stream.synchronize()` | 2 | 2 | `RBLNGuardImpl::synchronizeStream` / `queryStream` (`RBLNGuardImpl.cpp:149-155`) |
| 6.5 | Event creation and recording | 2 | 2 | `RBLNGuardImpl::record`; `torch.rbln.Event`; `test_event.py`, `RBLNEventTest.cpp` |
| 6.6 | Non-blocking H2D/D2H transfer with stream overlap | 2 | 2 | `test_stream.py::test_copy_on_stream_gated_by_event`, `test_compute_gated_by_async_copy_event`, `test_matmul_on_non_default_stream` |
| 6.7 | `stream.wait_stream(other)` | 3 | 2 | `test_stream.py::test_wait_stream_and_wait_event_do_not_error` |
| 6.8 | `event.wait()` / `event.synchronize()` | 3 | 2 | `block` / `synchronizeEvent`; cross-device waits degrade to a host-side synchronize (documented, `test_event.py:103`) |
| 6.9 | `event.elapsed_time(end_event)` | 3 | 0 | Explicitly raises — RBLN has no event timing (`streams.py:105`) |

Added wholesale by #154 (`torch.Stream` / `torch.Event` support), which is the largest single
score movement in this re-run: 33.3% → 85.4%. Also not supported and documented as such:
`torch.Stream.native_handle`, and cross-process event sharing (`ipc_handle` /
`from_ipc_handle` raise, so `interprocess=True` grants no capability — #212). Both attributes
are deliberately left present so a capability probe that only looks them up succeeds and the
failure surfaces at the call.

---

## 7. RNG & Generator — Level: **3**

Unchanged this run, and now explicitly tracked rather than merely absent:
`test_privateuse1_contract.py::test_manual_seed_reaches_the_backend` is a `strict=True`
xfail citing `torch/random.py::_seed_custom_device`, which needs **both** `_is_in_bad_fork`
and `manual_seed_all` on the device module and silently no-ops without either.

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 7.1 | Custom `at::Generator` subclass | Both | 1 | 1 | `RBLNGeneratorImpl` exists but `set_state` is a no-op and `get_state` returns uninitialized bytes |
| 7.2 | Registered via `REGISTER_GENERATOR_PRIVATEUSE1(GeneratorClass)` | PU1 | 2 | 1 | Not used; registered through the newer `getNewGenerator` hook instead, but `getDefaultGenerator` is missing |
| 7.3 | `torch.rbln.manual_seed(seed)` | Both | 2 | 0 | Not exposed on the device module |
| 7.4 | Generator fork safety (fork handler registered) | Both | 2 | 0 | No fork handler |
| 7.5 | `torch.Generator(device='rbln')` works | Both | 2 | 1 | Constructs via `getNewGenerator`, but the seed is inert — all random ops CPU-fallback |
| 7.6 | `get_rng_state()` / `set_rng_state()` for checkpointing | Both | 2 | 0 | `set_state` is `{}`; `get_state` returns uninitialized CPU bytes |
| 7.7 | `torch.rbln.initial_seed()` | Both | 3 | 0 | Absent |
| 7.8 | `torch.use_deterministic_algorithms(True)` respected | Both | 3 | 0 | Not wired |

---

## 8. Operator Registration — Level: **1**

### 8.1 Minimal Kernel Set

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 8.1.1 | `empty.memory_format` — tensor factory | 1 | 2 | `RBLNTensorFactories.cpp` via `RBLNRegisterOps.cpp:139`; `empty_strided` also native |
| 8.1.2 | `_copy_from` / `_copy_from_and_resize` | 1 | 2 | `RBLNCopy.cpp`; `_foreach_copy_` also native |
| 8.1.3 | `_local_scalar_dense` — `.item()` | 1 | 2 | `RBLNRegisterOps.cpp:184`; `test/rbln/test_native_local_scalar_dense.py` |
| 8.1.4 | Tensor creation ops (`zeros`, `ones`, `randn`, `full`, `arange`, `linspace`) | 1 | 2 | `arange.start_out` native; the rest via `empty` + fill/fallback |

### 8.2 Extended Operator Coverage

Scoring convention: **2** = native device kernel; **1** = correct but via CPU fallback (or a
mix of native and fallback within the category); **0** = broken.

| Category | Example Ops | Priority | Points | Pass/Total | Notes |
|----------|--------|-------------|----------|------------|-------|
| Elementwise unary | `abs`, `neg`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu` | 1 | 1 | ~6/12 native | Native: `silu`, `rsqrt`, `neg`, `ceil`, `abs`, `log`, `floor`, `sigmoid`. Fallback: `exp`, `sin`, `cos`, `sqrt`, `tanh`, `relu`, `gelu` |
| Elementwise binary | `add`, `sub`, `mul`, `div`, `remainder`, `pow`, `maximum`, `minimum` | 1 | 2 | 7/8 native | `add/sub/mul/div/div.out_mode/pow.Tensor_Scalar/maximum/minimum` native; `remainder` falls back |
| Reduction | `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `any`, `all`, `prod` | 1 | 1 | 3/9 native | `mean.out`, `max`, `min` native; `sum.IntList_out`, `argmax`, `any`, `all` explicitly registered to fallback |
| Linear algebra | `mm`, `bmm`, `addmm`, `matmul`, `linear`, `einsum` | 1 | 2 | 4/4 + composites | `mm.out`, `bmm.out`, `addmm.out`, `linear` native; `matmul`/`einsum` decompose onto them |
| Normalization | `batch_norm`, `layer_norm`, `group_norm`, `instance_norm` | 1 | 1 | 0 native | `native_batch_norm` → fallback; `layer_norm` decomposes onto native `mean`/`rsqrt`/`mul` |
| Activation | `relu`, `gelu`, `silu`, `sigmoid`, `softmax`, `log_softmax` | 1 | 1 | 3/6 native | `silu.out`, `sigmoid.out`, `_softmax.out` native; `relu` → fallback |
| Attention | `scaled_dot_product_attention` | 1 | 2 | native | `_scaled_dot_product_fused_attention_overrideable` + `_fused_sdp_choice` dispatch; custom paged/flash attention ops |
| Memory ops | `clone`, `copy_`, `fill_`, `zero_`, `empty_like`, `zeros_like` | 1 | 2 | native | `clone`, `zero_`, `fill_.Scalar`, `_copy_from`, `resize_`, `set_` all native |
| Comparison | `eq`, `ne`, `lt`, `gt`, `le`, `ge` | 2 | 2 | 12/12 native | Both `.Tensor_out` and `.Scalar_out` overloads |
| Convolution | `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d` | 2 | 1 | 0 native | CPU fallback only — no vision-model target today |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d` | 2 | 1 | 0 native | CPU fallback only |
| Indexing | `index`, `index_put`, `index_select`, `gather`, `scatter`, `masked_fill` | 2 | 1 | 4/6 native | `index_select(.out)`, `index_copy.out`, `index.Tensor_out` (single-axis), `masked_fill_` native; `gather`/`scatter`/`_index_put_impl_` fallback |
| Shape ops | `reshape`, `view`, `permute`, `transpose`, `contiguous`, `cat`, `stack`, `chunk`, `split`, `unsqueeze`, `squeeze`, `expand`, `repeat` | 2 | 2 | native | View ops metadata-only via `at::native::*`; `cat.out` and `repeat_interleave.Tensor` native, with a strided v2v engine |
| Random | `uniform_`, `normal_`, `bernoulli_`, `dropout`, `rand`, `randn` | 2 | 1 | 0 native | All explicitly registered to CPU fallback |
| Embedding | `embedding`, `embedding_bag` | 2 | 1 | 0 native | Via generic fallback / `index_select` decomposition |
| Loss functions | `cross_entropy`, `mse_loss`, `nll_loss`, `bce_with_logits` | 2 | 1 | 0 native | CPU fallback; training is not a current target |
| Type casting | `to(dtype)`, `float()`, `half()`, `bfloat16()`, `int()`, `bool()` | 2 | 1 | fp16/bf16 on device | Int casts round-trip host (a documented `host_bounce/d2d_copy` signal) |
| Sorting | `sort`, `topk`, `argsort` | 3 | 1 | 0 native | `topk.values` registered to fallback |

### 8.3 Model-Level Validation

| Model | Framework | Priority | Points | Notes |
|-------|-----------|----------|--------|-------|
| Llama-2-7B (inference) | HuggingFace transformers | 1 | 2 | Llama-3.2-1B and 3B in `test/models/test_transformers.py`; logits compared against CPU fp32 across a dtype × attn covering array |
| ResNet-50 | torchvision | 2 | 0 | Not tested; conv/pooling have no native kernels |
| BERT-base | HuggingFace transformers | 2 | 0 | Not tested |
| GPT-2 | HuggingFace transformers | 2 | 0 | Not tested |
| Vision Transformer (ViT) | torchvision / timm | 3 | 0 | Not tested |
| Stable Diffusion (UNet) | diffusers | 3 | 0 | Not tested |
| Whisper | HuggingFace transformers | 3 | 0 | Not tested |
| T5 | HuggingFace transformers | 3 | 0 | Not tested |

Additional decoder-only models validated beyond the template: Qwen2.5-1.5B, Qwen3-0.6B (vLLM),
EXAONE-3.5-2.4B / 7.8B (optimum). Forward-pass and numerics only — no backward validation.

### 8.4 Fallback Mechanisms

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 8.4.1 | `AutogradPrivateUse1` fallback for autograd | Both | 1 | 2 | `RBLNRegisterOps.cpp:129-131`, handles the no-VariableHooks build |
| 8.4.2 | Per-operator CPU fallback | Both | 2 | 2 | ~60 explicit `m.impl(..., fallback_rbln)` registrations |
| 8.4.3 | Global fallback via `torch::Library::fallback()` | Both | 2 | 2 | `TORCH_LIBRARY_IMPL(_, PrivateUse1)`; switchable to a hard error via env |
| 8.4.4 | Fallthrough for metadata/shape-only dispatch | Both | 3 | 2 | View ops mapped to `at::native::view`/`as_strided`/`alias`/`unfold`, no device round-trip |

### 8.5 Custom Operators

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 8.5.1 | `TORCH_LIBRARY(<ns>, m)` schema registration | 2 | 2 | `rbln_custom_ops` namespace (schema defined in `rebel-compiler`) |
| 8.5.2 | Kernel implementation + dispatch registration | 2 | 2 | `register_custom_ops.py:449-455` — 6 paged/flash attention kernels on `PrivateUse1` |
| 8.5.3 | Meta (shape-inference) kernel for `torch.compile` | 2 | 0 | No `register_fake` / `impl_abstract` anywhere in the repo |
| 8.5.4 | `torch.autograd.Function` for custom backward | 2 | 1 | Backward supplied as native ops (`linear_backward`, `silu_backward`, `_softmax_backward_data`, SDPA backward), not `autograd.Function` |

---

## 9. Python Frontend & Device-Agnostic APIs — Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 9.1 | `torch.rbln.is_available()` | Both | 1 | 2 | `device.py:80` |
| 9.2 | `Tensor.to(device)` / `Tensor.rbln()` / `Tensor.is_rbln` | Both | 1 | 1 | `.to("rbln")` works; `Tensor.rbln()` / `is_rbln` absent (see 1.1.6) |
| 9.3 | `nn.Module.to(device)` | Both | 1 | 2 | `test/rbln/walkthrough/test_06_nn_module.py` |
| 9.4 | `torch.rbln.device_count()` | Both | 2 | 2 | `device.py:55` |
| 9.5 | `torch.rbln.synchronize()` | Both | 2 | 2 | `device.py:122` |
| 9.6 | `torch.accelerator.current_device()` | Both | 2 | 2 | Via guard impl; `test_04_accelerator_interface.py:45` |
| 9.7 | `torch.accelerator.device_count()` | Both | 3 | 2 | `test_04_accelerator_interface.py:38` — agrees with `torch.rbln.device_count()` |
| 9.8 | `torch.accelerator.is_available()` | Both | 3 | 2 | `test_04_accelerator_interface.py:28` |
| 9.9 | `torch.accelerator.synchronize()` | Both | 3 | 2 | `RBLNGuardImpl::synchronizeDevice` |
| 9.10 | `torch.accelerator.current_stream()` / `set_stream()` | Both | 3 | 2 | Real pooled streams since #154; `test/rbln/test_accelerator_stream.py` (7 tests, incl. multi-device) |

Beyond the checklist: `torch.rbln.device` / `device_of` context managers, `physical_device_count()`,
`is_dummy_device()`, `is_initialized()`, `device_summary()`, and the full
`torch.accelerator.memory.*` surface.

---

## 10. Autograd (Training Support) — Level: **1**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 10.1 | Dispatch key has corresponding `AutogradPrivateUse1` registered | Both | 1 | 2 | `RBLNRegisterOps.cpp:129` |
| 10.2 | `.backward()` completes on a simple loss | Both | 1 | 1 | Backward kernels only for `linear`, `silu`, `_softmax`, SDPA; everything else via CPU fallback. No in-repo training test |
| 10.3 | Gradients match CUDA numerically | Both | 1 | 0 | No gradient-correctness comparison anywhere in the repo |
| 10.4 | `torch.autograd.grad()` works | Both | 2 | 1 | Plausible via the same path; untested |
| 10.5 | Gradient accumulation | Both | 2 | 1 | Untested |
| 10.6 | `torch.autograd.Function` custom fwd/bwd on device | Both | 2 | 1 | Exercised only through `test/ops/test_ops.py` OpInfo autograd cases |
| 10.7 | Gradient checkpointing | Both | 2 | 0 | Untested |
| 10.8 | Mixed precision backward | Both | 2 | 0 | Blocked on AMP (§11) |
| 10.9 | Higher-order gradients | Both | 3 | 0 | Untested |

---

## 11. Automatic Mixed Precision (AMP) — Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 11.1 | `AutocastPrivateUse1` dispatch key kernels registered | Both | 1 | 0 | None. Deliberate — see below |
| 11.2 | `torch.autocast(device_type="rbln")` works | Both | 1 | 1 | Does not raise: the empty dtype catalog makes torch disable autocast with a warning |
| 11.3 | Training loop with AMP converges | Both | 1 | 0 | N/A today — no AMP, no training validation |
| 11.4 | `get_amp_supported_dtype()` returns supported dtypes | Both | 2 | 2 | `device.py:112`; returns `[]` by design, kept because transformers/accelerate assert its presence |
| 11.5 | Ops correctly cast to lower precision inside autocast | Both | 2 | 0 | No cast policy |
| 11.6 | Ops that need fp32 stay in fp32 | Both | 2 | 0 | No cast policy |
| 11.7 | `torch.amp.GradScaler("rbln")` works | Both | 2 | 0 | Untested |

The empty catalog is an intentional, tested contract (`c10/rbln/RBLNSupportedDtypes.h:14-19`,
`test/rbln/test_amp_autocast.py`): advertising fp16/bf16 without an `AutocastPrivateUse1` policy
used to raise `NotImplementedError` on the first op, so an empty set was chosen to degrade to a
warning instead. Restore the dtypes when the cast policy lands.

---

## 12. torch.compile / Inductor — Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 12.1 | `DeviceInterface` subclass implemented | Both | 1 | 0 | No `torch._inductor` integration in the repo |
| 12.2 | Registered via `register_interface_for_device("rbln", …)` | Both | 1 | 0 | Not called |
| 12.3 | `torch.compile(model)` does not error on a simple model | Both | 1 | 2 | `backend="rbln"` registered by `rebel.core.torch_compile`; this is the backend's own execution engine |
| 12.4 | Compiled model produces correct output | Both | 1 | 2 | `test/rbln/test_graph_eager_mode.py`, `test_torch_compile_patch.py`, model tests |
| 12.5 | No unexpected graph breaks on standard models | Both | 2 | 1 | `cache_size_limit` raised to 64 and `fail_on_cache_limit_hit=True` with a reset-and-retry wrapper — i.e. recompile pressure is managed, not absent |
| 12.6 | FakeTensor / meta tensor support for device | Both | 2 | 1 | `TestFakeTensor` instantiated for `privateuse1` in `test/ops/test_ops.py`; no meta kernels for the custom ops |
| 12.7 | `torch.compile(model, mode="reduce-overhead")` works | Both | 3 | 0 | Not supported |
| 12.8 | Custom Inductor codegen registered | Both | 3 | N/A | RBLN uses its own Dynamo backend, not an Inductor codegen path |
| 12.9 | AOTInductor export works | Both | 3 | 1 | Equivalent capability via `options={"mode": ["compile_only"]}` producing `.rbln` artifacts; not AOTInductor |

---

## 13. Serialization & Model Portability — Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 13.1 | `TensorBackendMetaRegistry` registered for save/load | PU1 | 1 | 0 | No registration found |
| 13.2 | `torch.save(model.state_dict())` with device tensors | Both | 1 | 1 | Expected to work via `_copy_from` to CPU; no test in the repo |
| 13.3 | `torch.load(..., map_location="rbln")` | Both | 1 | 1 | Untested |
| 13.4 | Load a CUDA-saved state_dict onto RBLN | Both | 1 | 1 | Untested |
| 13.5 | Load an RBLN-saved state_dict onto CPU | Both | 2 | 1 | Untested |
| 13.6 | `torch.load(..., weights_only=True)` | Both | 2 | 1 | Untested |
| 13.7 | `safetensors` load/save | Both | 3 | 1 | Exercised indirectly — HF model tests load safetensors to CPU, then `.to("rbln")` |

This is still the weakest-evidence section in the report: nothing is known to be broken, but
nothing is verified either. A round-trip test would move most of these rows immediately.

One row did gain a citation without gaining a score. `test_privateuse1_contract.py::test_serialization_device_index_helper_exists`
is a `strict=True` xfail quoting `torch/serialization.py`, which documents
`device_module._utils._get_device_index(location, True)` as required of a PrivateUse1 backend and
uses it to resolve `map_location="rbln:N"`; `torch.rbln` has no `_utils` submodule, so torch falls
back to a looser path that cannot honour the backend's own index normalisation. That is a concrete,
cheap fix and it is the one to do first in this section.

---

## 14. Distributed Training — Level: **2**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 14.1 | Custom `ProcessGroup` subclass implemented | Both | 1 | 2 | `torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLN.cpp` (2,290 lines) over RCCL |
| 14.2 | Registered via `Backend.register_backend()` | Both | 1 | 2 | `torch_rbln/__init__.py:251` — `"rbln-ccl"`, `extended_api=True`, `devices=["rbln","cpu"]` |
| 14.3 | `init_process_group(backend="rbln-ccl")` works | Both | 1 | 2 | `test/distributed/test_process_group.py`, `test/cpp/c10d/ProcessGroupRBLNTest.cpp` |

### Collective Operations

| Collective | Priority | Points | Multi-node | Notes |
|-----------|----------|--------|------------|-------|
| `all_reduce` | 1 | 2 | `[x]` | Gloo fallback for non-fp16 dtypes |
| `broadcast` | 1 | 2 | `[x]` | Also used for `rccl_unique_id` distribution via the store |
| `all_gather` | 2 | 2 | `[x]` | Plus `_allgather_base`, `allgather_into_tensor_coalesced` |
| `reduce_scatter` | 2 | 2 | `[x]` | Plus `_reduce_scatter_base`; Gloo fallback for non-fp16 |
| `barrier` | 2 | 2 | `[x]` | |
| `send` / `recv` (P2P) | 3 | 2 | `[x]` | |
| `all_to_all` | 3 | 0 | `[ ]` | Not implemented |

### Distributed Strategies

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 14.4 | DDP training (2+ devices, loss converges) | 1 | 1 | No DDP test; training is not a current target |
| 14.5 | FSDP training | 2 | 0 | Untested |
| 14.6 | Tensor Parallel | 2 | 2 | `test/distributed/test_tp_pp.py`; plus compile-time auto-`num_devices` and TP failover |
| 14.7 | Multi-node training (2+ nodes) | 2 | 1 | RDMA IP auto-discovery + control-plane IP defaults present; only autoport tests in repo |
| 14.8 | Pipeline Parallel | 3 | 2 | `test/distributed/test_tp_pp.py`, `test_tp_pp_autoport.py` |
| 14.9 | `DeviceMesh` works | 3 | 1 | `is_initialized()` is explicitly shaped so `init_device_mesh` behaves on a no-NPU host; no direct mesh test |

---

## 15. Profiler — Level: **3**

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 15.1 | `ProfilerStubs` registered (operator-level timing) | PU1 | 2 | 0 | No `REGISTER_PRIVATEUSE1_PROFILER` |
| 15.2 | `torch.profiler.profile` collects device traces | Both | 2 | 2 | Kineto bridge registered at import (`_register_kineto_profiler`) |
| 15.3 | Kernel-level timing visible | Both | 2 | 1 | Via the rbln kineto emitter; inactive on ATOM devices by runtime gate |
| 15.4 | Kineto `IActivityProfiler` plugin | PU1 | 3 | 2 | `torch_rbln/csrc/rbln/profiler/kineto/rbln_kineto_adapter.cc` |
| 15.5 | Correlation-ID plumbing for kernel/op linking | Both | 3 | 2 | #194 draws run → triggered-op links as kineto flow arrows (`rbln_kineto_emitter.cc:43-125`), one flow id per host launch fanning 1:N to device slices, with counts logged for unwired launches |
| 15.6 | Traces viewable in TensorBoard / Chrome tracing | Both | 3 | 2 | Standard kineto trace output |
| 15.7 | Memory profiling | Both | 3 | 1 | `memory_stats()` and `explain()`'s device-memory high-water; not `torch.profiler`'s memory timeline |

Beyond the checklist: `torch.rbln.explain()` — a hidden-overhead explainer with no core
equivalent (see Upstream Candidates, §24.2) — plus per-op fallback/recompile attribution,
fallback reason histograms, opt-in call-site capture, and a librbln boundary timer.

---

## 16. DataLoader Integration — Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 16.1 | `DataLoader(pin_memory=True)` works | 2 | 2 | Pinned allocator + `isPinnedPtr` wired through the hooks; `test/rbln/test_pin_memory.py` |
| 16.2 | `tensor.to(device, non_blocking=True)` overlaps with compute | 2 | 2 | Pinned + `non_blocking` takes the async DMA path (`docs/EXPLAIN.md` §6) |
| 16.3 | Multi-worker DataLoader doesn't deadlock with device | 2 | 1 | No multi-worker DataLoader test; no fork handler registered (cf. 7.4) |

---

## 17. Additional PyTorch APIs — Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 17.1 | Quantization — quantized dtype support and op kernels | 2 | 0 | No quantized dtype support |
| 17.2 | ONNX Export — `torch.onnx.export` with device tensors | 3 | 0 | Untested |
| 17.3 | Tensor/Storage Customization | 3 | 2 | `resizePrivateUse1Bytes`, `set_.source_Storage_storage_offset`, `_create_tensor_from_ptr`, `set_device_layout_like`, `_efficientzerotensor` |

---

## 18. Testing & Validation — Level: **2**

### 18.1 Device-Generic Test Framework

| # | Item | Path | Priority | Points | Notes |
|---|------|------|----------|--------|-------|
| 18.1.1 | OpInfo-based operator compliance tests pass | Both | 1 | 2 | `test/ops/test_ops.py` — PyTorch v2.11 `test_ops.py` ported, 3,244 lines |
| 18.1.2 | `instantiate_device_type_tests` runs on the device | PU1 | 2 | 2 | `only_for="privateuse1"` on `TestCommon`, `TestCompositeCompliance`, `TestMathBits`, `TestFakeTensor`, `TestTags`, `TestForwardADWithScalars` |
| 18.1.3 | `PrivateUse1TestBase` auto-included in test framework | PU1 | 2 | 2 | Handled in `test/utils.py:71-73` (documents the `device_type` mutation trap) |
| 18.1.4 | Common device dtype tests pass | Both | 2 | 2 | `test/filters.py` custom instantiation with a documented xfail/skip filter set |

### 18.2 Module-Level Tests

| Test Area | Priority | Points | Notes |
|-----------|----------|--------|-------|
| Tensor creation & transfer | 2 | 2 | `test_tensor_copy.py`, `test_copy_v2v.py`, `test_v2v_cross_chiplet.py`, walkthrough 02/03 |
| Memory allocation & cleanup | 2 | 2 | `test_tensor_memory.py`, `test_memory_stats.py`, `RBLNAllocatorTest.cpp` |
| AMP / autocast | 2 | 2 | `test_amp_autocast.py` — locks the deliberate empty-catalog contract end to end |
| Backward pass / autograd | 2 | 1 | Only via OpInfo autograd cases; no dedicated suite |
| torch.compile correctness | 2 | 2 | `test_torch_compile_patch.py`, `test_graph_eager_mode.py`, walkthrough 07 |
| Distributed collectives | 2 | 2 | `test_process_group.py`, `benchmark_collective_ops.py`, `test_tp_pp*.py`, walkthrough 08 |
| Operator coverage | 2 | 2 | `test_registered_ops.py`, `test_ops.py`, `test_llama_ops.py`, per-op suites |
| Stream correctness | 3 | 2 | `test_stream.py` (19), `test_accelerator_stream.py` (7), `test/cpp/core/RBLNStreamTest.cpp` |
| Event correctness | 3 | 2 | `test_event.py`, `RBLNEventTest.cpp` |
| Storage operations | 3 | 2 | `test_non_zero_storage_offset.py`, `test_view_offset_ops.py`, `test_set_device_layout_like.py` |
| RNG reproducibility | 3 | 0 | No RNG test (generator is a stub) |
| Profiler output | 3 | 2 | `test_profiler.py` |
| Serialization round-trip | 3 | 0 | No test |

### 18.3 CI Integration

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 18.3.1 | CI pipeline runs device-generic test suite | 2 | 2 | `.github/workflows/ci.yaml` + `build.yaml`; `test_set_ci` pytest markers select the CI subset |
| 18.3.2 | Regression tests on each upstream update | 2 | 2 | `update-rebel-compiler-dependency.yaml`; `docs/THIRD_PARTY_UPDATE.md`; plus `nightly-torch.yaml` (#196) — a daily build against PyTorch **nightly**, which catches core breakage before the pin moves |
| 18.3.3 | Integration with `pytorch-integration-tests` framework | 3 | 0 | Not integrated |

---

## 19. Dtype Support Matrix — Level: **2**

Device residency, not merely acceptance: non-fp16/bf16 dtypes accept `device="rbln"` but occupy
zero device bytes and are served host-side (`c10/rbln/RBLNSupportedDtypes.h`,
`test/rbln/test_accelerator_contract.py`).

| Dtype | Priority | Points | Compute | Storage | AMP Target | CUDA Parity | Notes |
|-------|----------|--------|---------|---------|------------|-------------|-------|
| `float16` | 1 | 2 | `[x]` | `[x]` | `[ ]` | `[x]` | Primary dispatch dtype (`kDispatchDtypes`, `kSdpaDtypes`, `kCapabilityDtypes`) |
| `bfloat16` | 1 | 2 | `[x]` | `[x]` | `[ ]` | `[x]` | Second dispatch dtype |
| `float32` | 1 | 1 | `[ ]` | `[ ]` | — | `[ ]` | Accepted, host-backed, CPU fallback |
| `int8` | 2 | 1 | `[ ]` | `[ ]` | — | `[ ]` | Host-backed |
| `int32` | 2 | 1 | `[ ]` | `[ ]` | — | `[ ]` | Host-backed; int fast-path handlers exist (`fast_paths/int_elementwise.cpp`) |
| `int64` | 2 | 1 | `[ ]` | `[ ]` | — | `[ ]` | Host-backed; int64→int32 casts host-bounce |
| `bool` | 2 | 1 | `[ ]` | `[ ]` | — | `[ ]` | Host-backed; special-cased in the shim (`where.self_out` cond) |
| `float64` | 3 | 1 | `[ ]` | `[ ]` | — | `[ ]` | Host-backed |
| `int16` | 3 | 1 | `[ ]` | `[ ]` | — | `[ ]` | Host-backed |
| `uint8` | 3 | 1 | `[ ]` | `[ ]` | — | `[ ]` | Host-backed |
| `complex64` | 3 | 0 | `[ ]` | `[ ]` | — | `[ ]` | Unsupported (`view_as_real` explicitly falls back — conjugate unsupported) |
| `complex128` | 3 | 0 | `[ ]` | `[ ]` | — | `[ ]` | Unsupported |
| `float8_e4m3fn` | 3 | 0 | `[ ]` | `[ ]` | — | `[ ]` | Not in the dtype catalog |
| `float8_e5m2` | 3 | 0 | `[ ]` | `[ ]` | — | `[ ]` | Not in the dtype catalog |

---

## 20. Numerical Accuracy — Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 20.1 | float32 ops match CUDA within 1e-5 atol | 1 | 1 | Reference is CPU fp32, not CUDA. fp32 is host-served so it is exact by construction; the real comparison target is fp16/bf16 |
| 20.2 | float16 ops match CUDA within 1e-3 atol | 1 | 1 | Materially better evidence, same verdict: `test_fp16_numerics.py` (#203) now bounds per-op fp16 deviation against an fp32 CPU reference for add/mul/sum/matmul/silu_mul, and model tolerances were re-derived. Still **no CUDA reference**, so the row's actual claim stays unverified |
| 20.3 | Loss convergence curve matches CUDA | 1 | 0 | No training |
| 20.4 | bfloat16 ops match CUDA within 1e-2 atol | 2 | 1 | Covered by the same dtype covering array against CPU fp32 |
| 20.5 | Reduction ops handle large tensors without overflow | 2 | 1 | `test_sdpa_decode_overflow.py` covers a specific SDPA overflow case with a shape-based fallback; no general large-reduction sweep |
| 20.6 | Matmul results are deterministic | 2 | 1 | Not explicitly tested |

The pragmatic gap here is the reference: everything is validated against **CPU fp32**, which
catches gross errors but cannot detect a systematic divergence from what other accelerators
produce. A CUDA (or at minimum XPU) cross-reference would be the single highest-value addition.

---

## 21. Ecosystem Compatibility — Level: **3**

| Library | Priority | Points | Version Tested | Notes |
|---------|----------|--------|----------------|-------|
| HuggingFace transformers | 1 | 2 | per `pyproject` dev deps | `test/models/test_transformers.py` — Llama-3.2-1B/3B, Qwen2.5-1.5B, EXAONE-3.5-2.4B |
| HuggingFace accelerate | 2 | 1 | — | Not directly tested; its `torch.autocast(enabled=False)` pattern is explicitly accommodated |
| DeepSpeed | 2 | 0 | — | Not tested |
| torchvision | 2 | 0 | — | Not tested; no conv/pooling kernels |
| PEFT (LoRA, QLoRA) | 2 | 0 | — | Not tested |
| Flash Attention | 2 | 1 | — | Own `flash_attention_naive_{prefill,decode}` custom ops; not the `flash-attn` library |
| triton | 2 | N/A | — | RBLN does not target a Triton codegen path |
| Megatron-LM | 3 | 0 | — | Not tested |
| torchaudio | 3 | 0 | — | Not tested |
| torchtune | 3 | 0 | — | Not tested |
| bitsandbytes | 3 | 0 | — | Not tested |

Beyond the template: **vLLM** (`test/models/test_vllm_llm.py`, Qwen3-0.6B, with a worker patch)
and **optimum** (`test/models/test_optimum_llm.py`, EXAONE-3.5-7.8B) are both integration-tested —
these are the backend's actual serving targets and arguably matter more than several rows above.

---

## 22. Registration API Quick Reference

### PrivateUse1 Path — as implemented

| What | API / Macro | Status | Location |
|------|-------------|--------|----------|
| Backend name | `torch.utils.rename_privateuse1_backend("rbln")` | ✅ | `torch_rbln/__init__.py:53` |
| Device module | `torch._register_device_module("rbln", mod)` | ✅ | `torch_rbln/__init__.py:58` |
| Method generation | `generate_methods_for_privateuse1_backend("rbln")` | ❌ | Not called |
| Hooks | `RegisterPrivateUse1HooksInterface(hooks_ptr)` | ✅ | `c10/rbln/impl/RBLNGuardImpl.cpp:18` |
| Guard | `C10_REGISTER_GUARD_IMPL(PrivateUse1, RBLNGuardImpl)` | ✅ | `c10/rbln/impl/RBLNGuardImpl.cpp:14` |
| Allocator | `REGISTER_ALLOCATOR(c10::kPrivateUse1, &allocator)` | ✅ | `c10/rbln/RBLNAllocator.cpp` (end) |
| Generator | `REGISTER_GENERATOR_PRIVATEUSE1(GenClass)` | ➖ | Superseded by the `getNewGenerator` hook; `getDefaultGenerator` still missing |
| Kernels | `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` | ✅ | `aten/src/ATen/native/RBLNRegisterOps.cpp:134` + codegen from vendored `native_functions.yaml` |
| Global fallback | `TORCH_LIBRARY_IMPL(_, PrivateUse1, m).fallback(...)` | ✅ | `RBLNRegisterOps.cpp:118` |
| Autograd | `TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m).fallback(...)` | ✅ | `RBLNRegisterOps.cpp:129` |
| SDPA dispatch | `REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, …)` | ✅ | `RBLNRegisterOps.cpp:108` |
| AMP | `TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m)` | ❌ | Deliberately absent (§11) |
| Serialization | `TensorBackendMetaRegistry(PrivateUse1, …)` | ❌ | Not registered |
| Profiler | `REGISTER_PRIVATEUSE1_PROFILER(StubClass)` | ❌ | Not registered; kineto `IActivityProfiler` used instead |
| ProcessGroup | `torch.distributed.Backend.register_backend(…)` | ✅ | `torch_rbln/__init__.py:251` (`"rbln-ccl"`) |
| torch.compile | `register_interface_for_device("rbln", DevInterface)` | ❌ | Not called; `backend="rbln"` Dynamo backend used instead |
| Autoload | `entry_points = {"torch.backends": [...]}` | ✅ | `pyproject.toml:26` |

---

## Appendix: APIs Discovered Beyond the Checklist

Public / semi-public surface this backend exposes that the template does not enumerate. Most of
these are the raw material for the upstream candidates above.

| API | Location | What it does |
|-----|----------|--------------|
| `torch.rbln.explain(with_stack=False)` | `torch_rbln/profiler.py` | Hidden-overhead explainer region (see §24.2) |
| `torch.rbln.explain_steady(fn, warmup=…, as_diff=…)` | `torch_rbln/profiler.py` | Auto-places cold/warm regions to separate one-time from recurring overhead |
| `RBLNExplain.report/verdict/dump/help/diff` | `torch_rbln/profiler.py` | Human report, CI-gating dict, raw signal dict, per-signal prose, A/B diff |
| `torch.rbln.device_summary()` | `torch_rbln/device/device.py:368` | ASCII logical↔physical NPU topology table |
| `torch.rbln.physical_device_count()` | `torch_rbln/device/device.py:65` | Physical NPU count regardless of RSD grouping |
| `torch.rbln.is_dummy_device()` | `torch_rbln/device/device.py:90` | Whether host-backed, no-NPU mode is active |
| `torch.rbln.is_initialized()` | `torch_rbln/device/device.py:99` | CUDA-style lazy-init flag, shaped for `init_device_mesh` |
| `torch.rbln.offload()` | `torch_rbln/memory.py:345` | Scoped, nesting-counted host-to-disk offload switch |
| `torch.rbln.release_offload_temp_storage()` | `torch_rbln/memory.py:377` | Reclaims per-process offload temp files on a shutdown path |
| `torch.rbln.set_device_layout_like(target, ref)` | `torch_rbln/memory.py:81` | Adopt another allocation's device layout, no copy |
| `torch.rbln.memory_stats()` and friends | `torch_rbln/memory.py` | Dotted-key allocator stats (`allocated.current`, `reserved.peak`, …) |
| `torch_rbln._C._get_device_topology()` | `torch_rbln/csrc/rbln/Module.cpp:107` | `DeviceTopology` — entries, physical IDs, aggregated flag, unused devices |
| `torch_rbln._C._mark_zeros(vaddr)` | `torch_rbln/csrc/rbln/Module.cpp:138` | Mark device vmem logically zero without host allocation |
| `torch_rbln._C._create_tensor_from_ptr(...)` | `torch_rbln/device/device_tensor_utils.py` | Wrap an existing device pointer as a tensor |
| `torch_rbln._C._dispatch_*` diag surface | `torch_rbln/csrc/rbln/DispatchShim.h` | Per-op fallback / recompile counts, fallback reasons, call-site trace, warm-path segment timers |
| `torch_rbln._C._warmcache_clear()` | `torch_rbln/csrc/rbln/WarmCache.h` | Drop the C++ warm-runtime cache |
| `python -m torch_rbln.diagnose` | `torch_rbln/_internal/env_diagnostic.py` | Environment diagnostics that run even when the native extension fails to import |
| `TORCH_RBLN_DISABLE_FALLBACK` | `c10/rbln/RBLNFallbackConfig.h` | Category-wise conversion of silent CPU fallback into a hard error |
| `RBLN_DEVICE_MAP` / `RBLN_NPUS_PER_DEVICE` | `c10/rbln/DeviceMappingManager.h` | Logical-device aggregation of physical NPUs |
| `REGISTER_RBLN_CPU_FAST_PATH(op, fn)` | `aten/src/ATen/native/rbln/RBLNCPUFastPaths.h` | Self-registering per-op host micro-kernel |
| `c10::rbln::V2VBatch` / `H2VBatch` / `V2HBatch` | `c10/rbln/RBLNV2VBatch.h`, `c10/rbln/RBLNHostBatch.h` | Batched / strided copy accumulators over a shared `detail/RBLNCopyBatchImpl.h` |
| `torch.ops.rbln_custom_ops.*` | `torch_rbln/_internal/register_custom_ops.py` | Paged attention (prefill/decode, causal) and naive flash attention |
| `torch.rbln.Stream` / `Event` | `torch_rbln/device/streams.py` | **New (#154)** — `torch.cuda`-parity stream & event classes over a 32-slot per-device pool |
| `torch.rbln.current_stream()` / `default_stream()` / `set_stream()` / `stream()` | `torch_rbln/device/streams.py:112-170` | **New (#154)** — stream selection and `StreamContext` |
| `torch.rbln.bind_device_memory(tensor)` | `torch_rbln/memory.py:260` | **New (#213)** — materialize a lazy device allocation for out-of-band consumers (see §24.4) |
| `torch.rbln.huge_host_empty(nbytes)` | `torch_rbln/memory.py:293` | **New (#213)** — 2 MiB-aligned, prefaulted host slab for staging-free DMA (see §24.4) |
| `torch_rbln._internal.abi_check` | `torch_rbln/_internal/abi_check.py` | **New (#202)** — verifies the rebel ABI contract at import time |
| `torch_rbln._internal.rbln_runtime_lib` | `torch_rbln/_internal/rbln_runtime_lib.py` | **New (#198/#218)** — locates `librbln.so` from the rebel-compiler record (replaces the TVM-based `tvm_libinfo`) |

---

## Sources

- [Accelerator Integration Guide](https://docs.pytorch.org/docs/stable/accelerator/index.html)
- [OpenReg Reference Implementation](https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg)
- [Tracking Issue #158917](https://github.com/pytorch/pytorch/issues/158917)
- [PrivateUse1 Tutorial](https://docs.pytorch.org/tutorials/advanced/privateuseone.html)
- [PyTorch Multi-Device Blog Post](https://pytorch.org/blog/pt-multidevice-integration/)
- [ProcessGroup Extension Tutorial](https://docs.pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html)
- [Running and Writing Tests Wiki](https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests)
- [Accelerator Integration Working Group](https://github.com/pytorch-fdn/accelerator-integration-wg)
