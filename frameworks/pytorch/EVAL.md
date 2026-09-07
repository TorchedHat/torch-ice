## Part 1: PyTorch Readiness Evaluation

### Templates

Read `frameworks/pytorch/checklist.md`.
Copy it to `torch-air-report/torch_readiness_report_<backend>.md` as your working copy.

### Pre-Phase: Checklist Refinement via TorchTalk

Before every evaluation, sync the PyTorch checklist template with the current
PyTorch source using TorchTalk MCP tools. This ensures the checklist reflects
the latest APIs, hooks, and registration points.

**Ground Rules (internal skill rules -- do NOT include in generated reports):**
1. **Preserve structure**: Section numbering, table format, column layout, scoring model, level assignments, and markdown structure must NOT change during refinement. Only row-level content (add/update/remove items within existing tables) may change.
2. **Summary + level table at top**: Every generated document must start with the Executive Summary and Section Scores table (sorted by level) immediately after the header metadata. Metadata must include torch-air version and model.
3. **No section reordering**: Sections stay in their original document order. Only the summary table sorts by level. Template tables are pre-sorted by priority (1 first, then 2, then 3) -- preserve this order in generated reports.
4. **Log changes**: After refinement, output a short changelog (added N items, updated M items, removed K items) so the user can review before evaluation proceeds.
5. **No meta-content in output**: These ground rules, skill instructions, agent instructions, procedural steps (e.g., "The agent should...", "Check if...", "Search for...", "Once detected, the agent should..."), scoring model explanations, calculation formulas, and internal process details must never appear in the final generated report. When copying templates, strip all instructional prose, the Scoring Model section, and the Calculation section -- keep only headings, fillable tables, and filled-in content. The report is a standalone evaluation document for the reader, not a how-to guide for the evaluator.

**Step 1: Ensure PyTorch source is available**
- PyTorch source acquisition is handled in Phase 0 Step 0c. The resolved
  source path is stored in `PYTORCH_SOURCE`.
- If `PYTORCH_SOURCE` is not set (both TorchTalk and clone failed), skip
  refinement and proceed with the existing template as-is.

**Step 2: Validate existing checklist items**

Use a **hybrid strategy**: TorchTalk for Python-side APIs, modules, tests, and
dispatch bindings. Direct `grep` on PyTorch source for C++ macros and
registration utilities (TorchTalk `search` only indexes dispatch-registered
bindings, not C++ infrastructure like macros/registration functions).

| Checklist Section | Tool | Query | What to Check |
|-------------------|------|-------|---------------|
| Sec 1: Device Registration | `grep` | `rename_privateuse1_backend`, `generate_methods_for_privateuse1_backend`, `_register_device_module` in `torch/` and `c10/` | Function signatures, parameters unchanged |
| Sec 2: Accelerator Hooks | `grep` | `AcceleratorHooksInterface`, `RegisterPrivateUse1HooksInterface` in `c10/` | Hook method names, new mandatory hooks |
| Sec 2: Accelerator Hooks | TorchTalk `graph` | mode=calls on hooks interface class (when graph ready) | New extensibility points |
| Sec 3: Device Guard | `grep` | `C10_REGISTER_GUARD_IMPL`, `DeviceGuardImplInterface` in `c10/` | Guard interface methods |
| Sec 4: Autoload | `grep` | `torch.backends` entry point handling in `torch/` | Entry point group name, autoload mechanism |
| Sec 5: Memory | `grep` | `Allocator` subclass registration, `getPinnedMemoryAllocator` in `c10/` | Allocator interface changes |
| Sec 7: RNG | `grep` | `REGISTER_GENERATOR_PRIVATEUSE1` in `aten/` | Generator registration API |
| Sec 8: Operators | TorchTalk `tests` | mode=find query=`OpInfo` | New required ops, deprecated ops |
| Sec 8: Operators | `grep` | `TORCH_LIBRARY_IMPL` with `PrivateUse1` in reference backends | Registration patterns |
| Sec 9: Python Frontend | TorchTalk `modules` | trace `torch.accelerator` focus=full | New methods on torch.accelerator module |
| Sec 10: Autograd | `grep` | `AutogradPrivateUse1` in `c10/` and `aten/` | Autograd dispatch key registration |
| Sec 11: AMP | `grep` | `AutocastPrivateUse1` in `aten/` and `torch/amp/` | Autocast API changes |
| Sec 12: torch.compile | `grep` | `register_interface_for_device`, `DeviceInterface` in `torch/_inductor/` | Compile interface methods |
| Sec 12: torch.compile | TorchTalk `modules` | trace `DeviceInterface` focus=full | Interface method list |
| Sec 13: Serialization | `grep` | `TensorBackendMetaRegistry` in `torch/` | Serialization hook API |
| Sec 14: Distributed | TorchTalk `modules` | trace `ProcessGroup` focus=full | ProcessGroup interface changes |
| Sec 14: Distributed | `grep` | `register_backend` in `torch/distributed/` | Backend registration API |
| Sec 15: Profiler | `grep` | `REGISTER_PRIVATEUSE1_PROFILER`, `ProfilerStubs` in `torch/` | Profiler registration API |
| Sec 18: Testing | TorchTalk `tests` | mode=find query=`privateuse1` focus=files | Test file inventory |

All `grep` commands run against `PYTORCH_SOURCE` (set in Phase 0 Step 0c).
Use `grep -rn --include="*.py" --include="*.h" --include="*.cpp"` for broad
coverage. TorchTalk queries are only executed when TorchTalk is available;
otherwise those rows are skipped and grep provides the coverage.

For each query:
- If API still exists with same signature: no change needed
- If API renamed/moved: update checklist item text and notes
- If API removed: mark item as deprecated, add replacement if found

**Step 3: Discover new APIs not in checklist**

Run broader discovery queries using both tools:

TorchTalk queries:
1. `modules` mode=trace name=`torch.accelerator` focus=full -- find new Python methods
2. `modules` mode=list name=`accelerator` -- find new accelerator-related modules
3. `tests` mode=find query=`privateuse1` -- find new test infrastructure
4. `graph` mode=calls on hook/interface classes (when graph ready) -- find new extensibility points

grep queries on PyTorch source:
5. `grep -rn "privateuse1\|PrivateUse1" torch/ c10/ aten/ --include="*.py" --include="*.h" --include="*.cpp"` -- find all PU1 references
6. `grep -rn "def .*accelerator" torch/ --include="*.py"` -- find new accelerator APIs
7. `grep -rn "register.*private\|REGISTER.*PRIVATE" c10/ aten/ --include="*.h" --include="*.cpp"` -- find new registration macros

For each discovered API not already in the checklist:
- Determine which section it belongs to
- Assign appropriate priority (1=critical, 2=important, 3=nice-to-have)
- Add as new row to the template

**Step 4: Update the template**

Write changes to the template file at
`frameworks/pytorch/checklist.md`.
Log all changes (additions, modifications, removals) so the user can review.

---

### Phase 0: Source Discovery & Integration Path Detection

#### Step 0a: Backend Source Discovery

The user may not have the repo locally. The agent should:

1. **Check locally first**: `~/Dev/`, `~/projects/`, `pip show`, `python -c "import torch_<name>"`
2. **Search GitHub** if not local: `gh search repos "torch <name>"`, try `torch-<name>`, `torch_<name>`, `pytorch-<name>`
3. **Clone if needed**: `git clone --depth 1 <url> /tmp/<name>`
4. **Ask the user** only as a last resort
5. **Determine backend version**: Use `gh release list --repo <owner>/<repo> --limit 1`, `git describe --tags`, or `pip show <package>` to find the latest release version. Record it in the report header and the Source Discovery table.

#### Step 0b: PyTorch Version Resolution

Determine which PyTorch upstream version to evaluate the backend against.
The backend source must be available (from Step 0a) before this step.
Let `BACKEND_SOURCE` = the backend path from Step 0a (e.g., `/tmp/torch_npu`).

1. **User-specified version**: If `--pytorch-version <ver>` was provided,
   set `PYTORCH_VERSION=<ver>`. This overrides all auto-detection — the
   backend will be evaluated against this PyTorch version regardless of
   what it targets.
   Validate the version exists:
   ```
   git ls-remote --tags https://github.com/pytorch/pytorch.git refs/tags/v<ver>
   ```
   If validation fails, **stop and error** — do not fall through to
   auto-detection. Display a message listing available versions:
   ```
   git ls-remote --tags https://github.com/pytorch/pytorch.git "v*" | grep -oP 'v\d+\.\d+\.\d+$' | sort -V | tail -20
   ```
   Example: `Error: PyTorch version 2.20.0 does not exist. Recent releases: 2.3.0, 2.4.0, 2.5.0, ...`

2. **Detect from backend source**: Extract the PyTorch version the backend
   targets from `BACKEND_SOURCE`:
   ```
   grep -rn "torch[>=!~]=\|torch==" $BACKEND_SOURCE/setup.py $BACKEND_SOURCE/pyproject.toml $BACKEND_SOURCE/setup.cfg $BACKEND_SOURCE/requirements*.txt 2>/dev/null
   ```
   Parse the version from the match:
   - `torch==2.4.0` → `PYTORCH_VERSION=2.4.0` (exact pin, most reliable)
   - `torch>=2.4.0` → `PYTORCH_VERSION=2.4.0` (lower bound)
   - `torch>=2.4,<2.6` → `PYTORCH_VERSION=2.4.0` (use the lower bound)

   Also check: `pip show <backend_package> | grep Requires` if installed,
   or backend release tags that mirror PyTorch versions (e.g., backend
   `v2.1.0` → `PYTORCH_VERSION=2.1.0`).

   Use the most specific version found (exact pin > lower bound > tag match).

3. **Fallback to latest stable**: If none of the above yields a version,
   use the latest stable PyTorch release:
   ```
   pip index versions torch | head -1
   ```
   Or if pip is not available:
   ```
   git ls-remote --tags https://github.com/pytorch/pytorch.git "v*" | grep -oP 'v\d+\.\d+\.\d+$' | sort -V | tail -1
   ```
   Set `PYTORCH_VERSION` to the result.

Once resolved:
- Record `PYTORCH_VERSION` in the report header metadata (`PyTorch base`).
- Log to user: `Evaluating against PyTorch <PYTORCH_VERSION> (source:
  user-specified | backend-detected | latest stable)`.

#### Step 0c: PyTorch Source Acquisition

The evaluation needs PyTorch source at `PYTORCH_VERSION` (from Step 0b)
to refine the checklist and validate APIs.

1. **TorchTalk**: Call `get_status()`.
   - If TorchTalk is available, read the PyTorch source path and indexed
     git commit/version from the response.
   - Compare the indexed version against `PYTORCH_VERSION`.
   - If they match: set `PYTORCH_SOURCE` to TorchTalk's source path.
     Use TorchTalk MCP tools + grep. This is the best path — full
     cross-language analysis.
   - If they don't match: log a warning (`TorchTalk indexed at PyTorch
     <indexed_ver>, but evaluating against <PYTORCH_VERSION> — falling
     back to cloned source for version-specific checks`). TorchTalk tools
     can still be used for structural queries (call graphs, module tracing)
     that are stable across versions, but all version-sensitive grep
     queries (API signatures, registration macros) must run against the
     cloned source from step 2 below.

2. **Clone fallback**: If TorchTalk is not available, or if the indexed
   version doesn't match `PYTORCH_VERSION`:
   ```
   git clone --depth 1 --branch v$PYTORCH_VERSION https://github.com/pytorch/pytorch.git /tmp/pytorch-$PYTORCH_VERSION
   ```
   If the clone fails (tag not found), try normalizing the version:
   - Try `v$PYTORCH_VERSION.0` (e.g., `v2.4` → `v2.4.0`)
   - If both fail, list matching tags and suggest the closest:
     ```
     git ls-remote --tags https://github.com/pytorch/pytorch.git "v${PYTORCH_VERSION}*"
     ```
     Error with: `Tag v$PYTORCH_VERSION not found. Similar tags: v2.4.0, v2.4.1, ...`

   Set `PYTORCH_SOURCE=/tmp/pytorch-$PYTORCH_VERSION`.
   TorchTalk-only queries (marked `TorchTalk` in the Pre-Phase table)
   are skipped — grep covers the C++ macros, registration APIs, and
   Python-side patterns needed for checklist refinement.

`PYTORCH_SOURCE` now points to PyTorch source at the correct version for
use in all subsequent phases (Pre-Phase grep, Phase 1 probing, Phase 2.5
upstream comparison).

Then detect **PU1 vs Fork**:
- `grep -r "rename_privateuse1_backend\|register_privateuse1_backend"` -> PU1
- Check `c10/core/DeviceType.h`, `DispatchKey.h` for custom entries -> Fork
- Mark items for the other path as `[N/A]`

### Phase 1: Systematic Probing (Sections 1-23)

For each section, probe via source analysis or runtime testing. For each row:
- Points: 2 = fully implemented, 1 = partially, 0 = not implemented, N/A = excluded
- Fill Points, Notes columns (Priority is pre-filled)

**Probe order (by level, level 1 = most critical):**

**Level 1 (probe first):**
- Section 1: Device Registration & Management
- Section 8: Operator Registration (minimal kernels, extended ops, model validation, fallbacks, custom ops)
- Section 10: Autograd
- Section 3: Device Guard
- Section 2: Accelerator Hooks [PU1]
- Section 5: Memory & Allocator

**Level 2:**
- Section 13: Serialization & Model Portability
- Section 9: Python Frontend
- Section 11: AMP
- Section 12: torch.compile
- Section 14: Distributed
- Section 19: Dtype Support
- Section 20: Numerical Accuracy
- Section 18: Testing & CI

**Level 3:**
- Section 6: Streams & Events
- Section 7: RNG & Generator
- Section 4: Autoload [PU1]
- Section 16: DataLoader
- Section 15: Profiler
- Section 22: Ecosystem Compatibility
- Section 17: Future Modules

### Phase 2: Dynamic API Discovery

1. Introspect `torch.<name>` module -- flag public methods not in checklist
2. Query dispatcher for registered ops not tested in Section 8
3. Check hook overrides beyond Section 2

### Phase 2.5: Upstream Candidate Discovery

This is a non-scored advisory section. Goal: find features the backend
built that are generic enough to benefit all PU1/accelerator backends if
upstreamed to PyTorch core.

**Placement**: In the generated report, upstream candidates MUST appear
immediately after the Section Scores table and Readiness percentage
(inside the "Readiness Score & Summary" block), NOT at the end of the
document. This matches the template structure in `pytorch_checklist.md`.

#### Step 1: Find candidates (systematic diff)

The backend's source falls into two buckets:
- **Required PU1 plumbing** -- what Section 23 (Registration API Quick
  Reference) lists. Every PU1 backend must do these. Not upstream candidates.
- **Everything else** -- infrastructure the backend built beyond the required
  registration points. These are the candidates.

To find "everything else":

1. **Inventory the backend's public surface** -- list all Python modules,
   C++ source files, and exported symbols that are NOT direct implementations
   of a Section 1-22 checklist item. Focus on:
   - Python modules beyond the device module (profiler, memory, diagnostics, patches)
   - C++ files beyond the standard hooks/guard/allocator/generator
   - Monkey patches applied to `torch.*` or `torch._dynamo.*`
   - Custom context managers, decorators, or utilities

2. **Grep for infrastructure patterns** -- scan for code that wraps, extends,
   or works around PyTorch core APIs:
   ```
   grep -rn "monkey_patch\|patch_torch\|_original_" <backend>/ --include="*.py"
   grep -rn "fallback\|cpu_fallback\|fast_path" <backend>/ --include="*.py" --include="*.cpp"
   grep -rn "class.*Wrapper\|class.*Shim\|class.*Bridge" <backend>/ --include="*.py"
   grep -rn "explain\|diagnose\|hidden.*overhead" <backend>/ --include="*.py"
   grep -rn "reentrancy\|reentrant\|nested.*compile" <backend>/ --include="*.py"
   grep -rn "topology\|device_summary\|physical_device" <backend>/ --include="*.py"
   grep -rn "offload\|layout_like\|empty_cache" <backend>/ --include="*.py"
   ```

3. **Compare against other backends** -- if available (torch_npu, torch_xla,
   intel_extension_for_pytorch), check whether they independently built similar
   infrastructure. Two+ backends solving the same problem independently is a
   strong upstream signal.

#### Step 2: Classify each candidate

Apply these concrete tests in order:

**Generic** (ready to upstream):
- Contains zero references to backend-specific types, APIs, or hardware names
- Could be copy-pasted into `torch/accelerator/` and work for any PU1 backend
- Examples: device-index normalization, reentrancy guard, region-based profiler
  framework, atexit shutdown handler

**Needs Abstraction** (concept generic, implementation coupled):
- Solves a problem every backend faces, BUT implementation references
  backend-specific internals (allocator, compiler, runtime API)
- Upstream path: define an interface/hook in core, let backends implement
- Test: "Could another backend use this if we replaced the backend-specific
  calls with hook methods?" If yes, Needs Abstraction.
- Examples: `empty_cache()` with backend-specific cache types, explain profiler
  with backend-specific signal names, view-aware dispatch with backend-specific
  bad-pattern list

**Hardware-Specific** (not upstreamable):
- Solves a problem unique to this hardware's architecture, ISA, or runtime
- Other backends would never need this
- Examples: device-specific memory alignment workarounds, hardware topology
  that only this NPU uses, compiler-specific IR limitations

#### Step 3: Write upstream notes

For each Generic or Needs Abstraction feature, write:
- What the PyTorch core API would look like (function signature, module path)
- What the backend would implement (hook method, registration call)
- Whether PyTorch core already has a partial equivalent (grep pytorch source)

#### Categories to organize findings

Place discoveries into whichever of these 7 categories fits. Leave categories
empty (no rows) if nothing found.

| Category | What to look for |
|----------|-----------------|
| 24.1 CPU Fallback Infrastructure | Generic fallback dispatcher, diagnostic counters, per-op tracking, fast-path registry |
| 24.2 Profiler / Explain Framework | Hidden-overhead detection, region profiling, cause/remedy mapping, diff analysis |
| 24.3 View-Aware Dispatch | View recipe detection, shape simulation, fallback-to-contiguous policies |
| 24.4 Memory Management Patterns | `empty_cache()` with plugin points, offload context managers, layout affinity |
| 24.5 Device Topology & Utilities | Topology visualization, logical-to-physical mapping, capability queries |
| 24.6 Compile Integration Patterns | Reentrancy guards, event isolation, recompile limit handling, compile-only mode |
| 24.7 New Hooks / API Extensions | Methods beyond `AcceleratorHooksInterface`, new `torch.accelerator.*` APIs |

### Phase 3: Scoring (Summary at top of document)

```
Row weight: w_i = 1 / priority_i
Section Pct = sum(score_i * w_i) / sum(max_i * w_i) * 100  (excluding N/A rows, max_i=2)
weight_r = 1 / level
Readiness = (sum(Section Pct * weight_r) / sum(weight_r)) * 100
```

Fill the score table (sorted by level) at the top of the document, the overall readiness percentage, and the executive summary.

### Phase 4: Cleanup

If PyTorch source was cloned during Step 0c (i.e., `PYTORCH_SOURCE`
points to `/tmp/pytorch-$PYTORCH_VERSION`), log a message to the user:

```
Note: PyTorch source was cloned to /tmp/pytorch-<version> for this evaluation.
To free disk space, you may remove it later: rm -rf /tmp/pytorch-<version>
```

Do NOT auto-delete the cloned source — the user decides when to clean up.

---

## Part 2: Security Readiness (optional)

Security evaluation is a **dimension** of the PyTorch framework, not a separate
framework. Run only when `--security` or `--all` is specified.

Read `frameworks/pytorch/security/EVAL.md` and follow all phases.
Use `frameworks/pytorch/security/checklist.md` as the template.
Write output to `torch-air-report/torch_security_readiness_report_<backend>.md`.
