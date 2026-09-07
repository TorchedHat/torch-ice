# Torch Accelerator Integration Readiness (AIR)

A checklist-based evaluation tool that measures how well a hardware accelerator integrates with PyTorch — and whether its security posture is suitable for production deployment. It probes source code for device registration, operator coverage, memory management, distributed training, profiling, and more, then produces scored readiness reports. An optional security dimension covers multi-tenant isolation, memory encryption, transit security, data scrubbing, firmware, and the PyTorch integration surface.

## Workflow Process Outline:
- Skill Execution & PR Submission: The accelerator backend executes the skill and submits the comprehensive readiness report as a Pull Request (PR) to PyTorch-AIR repo.
- Engineering Review: A partner engineering discussion is held to review the overall readiness report and identify target features.
- Task Assignment & Collaboration: Tasks are assigned based on priority, workload capacity, complexity, and upstream engagement, with both partners collaborating as authors and co-authors on the respective PRs.
- Issue Creation: Red Hat files the identified issues within the PyTorch GitHub repository.
- PR Development: The assignees from the previous task assignment phase raise the corresponding PRs with co-authors.

Downstream Code Optimization: Following the successful merging of PRs, the partner initiates a downstream repository cleanup to eliminate redundant logic, ensuring the codebase remains streamlined and maintainable.

## What It Does

Given a backend name, source path, or GitHub URL, AIR:

1. Locates the backend source code (local, pip, or GitHub)
2. Detects the integration path (PrivateUse1 or Fork)
3. Probes 21 functional sections covering the full PyTorch integration surface
4. Optionally evaluates 6 security domains (37 items) for production deployment safety
5. Scores each item and computes a weighted readiness percentage
6. Identifies features the backend built that could be upstreamed to PyTorch core
7. Writes standalone markdown reports

## Installation

AIR is distributed as a Claude Code plugin through its own marketplace. Install it
from within Claude Code:

1. **Add the marketplace** (GitHub `owner/repo` shorthand):

   ```
   /plugin marketplace add TorchedHat/torch-air
   ```

2. **Install the plugin** (`plugin-name@marketplace-name`):

   ```
   /plugin install torch-air@torch-air
   ```

   Claude Code then prompts for an install scope: **user** (all projects),
   **project** (shared via `.claude/settings.json`), or **local** (this repo only).

Alternatively, run `/plugin` to open the interactive plugin manager and install
`torch-air` from the **Discover** tab.

### Install via settings.json

For team or project setups, declare the marketplace and plugin in
`.claude/settings.json` instead of running the commands:

```json
{
  "extraKnownMarketplaces": {
    "torch-air": {
      "source": { "source": "github", "repo": "TorchedHat/torch-air" }
    }
  },
  "enabledPlugins": {
    "torch-air@torch-air": true
  }
}
```

## Usage

Once installed, invoke the skill (namespaced by the plugin):

```
/torch-air:torch-accelerator-readiness <backend>              # functional only (default)
/torch-air:torch-accelerator-readiness <backend> --security   # security only
/torch-air:torch-accelerator-readiness <backend> --all        # both
```

Examples:

```
/torch-air:torch-accelerator-readiness <accelerator-name>
/torch-air:torch-accelerator-readiness <accelerator-name> --security
/torch-air:torch-accelerator-readiness <accelerator-name> --all
/torch-air:torch-accelerator-readiness /path/to/backend/source
/torch-air:torch-accelerator-readiness https://github.com/org/torch-backend --all
```

## Report Structure

### Integration Readiness Report

Each integration report contains:

- **Metadata table** — backend name, backend version, integration path, dispatch key, PyTorch version, torch-air version, model
- **Executive summary** — overall readiness, notable insights, strengths, gaps
- **Section scores** — per-section breakdown sorted by level
- **Upstream candidates** — features generic enough to benefit all backends
- **21 scored sections** — each row filled with points and evidence
- **Registration quick reference** — summary of all PU1/Fork registration points
- **Sources** — links to PyTorch docs, tutorials, and references

### Security Readiness Report

Each security report contains:

- **Metadata table** — accelerator name, backend package, version, deployment model, PyTorch version, torch-air version, model
- **Executive summary** — overall security readiness percentage, strengths, gaps
- **Domain scores** — per-domain breakdown sorted by level with weighted percentages
- **37 scored items** across 6 security domains with points and evidence
- **Gap analysis** — CRITICAL / HIGH / MEDIUM / LOW prioritized findings
- **Recommendations** — actionable next steps and vendor engagement topics
- **Evidence sources** — all referenced URLs and documents

## What It Evaluates

### Functional Integration (21 sections)

21 sections grouped into 3 levels. Level 1 carries the most weight in the overall readiness score.

### Level 1

| Section | What It Checks |
|---------|---------------|
| Device Registration & Management | Backend name registration, device module binding, device count, current device, multi-device indexing |
| Operator Registration | Minimal kernel set, extended op coverage, model validation, CPU fallbacks, custom ops |
| Autograd | AutogradPrivateUse1 dispatch key, backward pass, gradient accumulation, custom autograd functions |
| Device Guard | DeviceGuardImpl subclass, device/stream save-restore on scope exit |
| Accelerator Hooks [PU1] | AcceleratorHooksInterface methods — generator, context, pinned memory, device-from-pointer |
| Memory & Allocator | Device allocator, pinned memory, memory tracking APIs, OOM handling |

### Level 2

| Section | What It Checks |
|---------|---------------|
| Serialization & Model Portability | Save/load round-trip, cross-device deserialization, TensorBackendMeta hooks |
| Python Frontend & Device-Agnostic APIs | `torch.accelerator` module methods, device-agnostic tensor creation |
| AMP | Autocast registration, GradScaler support, dtype policies |
| torch.compile / Inductor | DeviceInterface registration, backend compiler, dynamic shapes, graph breaks |
| Distributed Training | ProcessGroup backend, collective ops (allreduce, broadcast, allgather), multi-node |
| Dtype Support Matrix | FP32, FP16, BF16, FP8, INT8, complex dtype coverage for compute and storage |
| Numerical Accuracy | Reference comparisons, tolerance settings, known numerics issues |
| Testing & Validation | Test infrastructure, OpInfo coverage, CI integration, device-agnostic test patterns |

### Level 3

| Section | What It Checks |
|---------|---------------|
| Streams & Events | Stream creation, synchronization, event recording, async transfers |
| RNG & Generator | Custom Generator subclass, manual seed, fork safety |
| Autoload [PU1] | `torch.backends` entry point, auto-import on `torch.device("<name>")` |
| DataLoader Integration | `pin_memory` support, worker-side device transfer |
| Profiler | Profiler stubs registration, trace export, kineto integration |
| Ecosystem Compatibility | Compatibility with torchvision, torchaudio, HuggingFace, and other libraries |
| Additional PyTorch APIs | Sparse tensors, quantization, nested tensors, and other API surfaces |

### Security (6 domains, 37 items)

Security is an evaluation dimension of the PyTorch framework, nested under `frameworks/pytorch/security/`.

| Domain | ID Prefix | Level | What It Covers |
|--------|-----------|-------|----------------|
| Multi-Tenant Isolation | SEC-MT | 1 | HW partitioning, SMMU, compute/memory/fault/temporal isolation |
| Device Memory Encryption | SEC-ME | 1 | At-rest encryption, HW Root of Trust, key management |
| Data Scrubbing | SEC-DS | 1 | Memory zeroing, KV cache clearing, verification |
| Host-Device Transit Security | SEC-HT | 2 | PCIe encryption, buffer protection, attestation |
| Firmware & Driver | SEC-FD | 2 | Secure boot, SAST, CVE process, DoS resistance |
| PyTorch Integration Surface | SEC-PI | 3 | Leak detection, input validation, CRCR integration |

## Scoring

Both functional and security evaluations use the same numeric scoring model.

### Row Scoring

Each checklist row is scored in the Points column:

| Points | Meaning |
|--------|---------|
| 2 | Fully implemented |
| 1 | Partially implemented |
| 0 | Not implemented |
| N/A | Not applicable (excluded) |

Every row has a max score of **2** and a **priority** (1 = critical, 2 = important, 3 = nice-to-have). The priority determines the row's weight: `weight = 1 / priority` (so P1 = 1.0, P2 = 0.5, P3 = 0.333). Priorities are fixed in the template and consistent across all evaluations.

### Section / Domain Score

The **max points** for a section/domain is the weighted sum assuming every non-N/A row scores a perfect 2:

```
max_pts = sum(2 * w_i)   for each non-N/A row
```

The **section/domain percentage** is:

```
section_pct = sum(score_i * w_i) / max_pts * 100
```

where `score_i` is the row's score (0, 1, or 2), `w_i = 1 / priority_i`, and N/A rows are excluded. Sections where every item is N/A are excluded entirely.

### Overall Readiness

Sections/domains are grouped into 3 weighted tiers:

| Tier | Weight |
|------|--------|
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |

```
weight = 1 / tier
Readiness (%) = (sum(section_pct × weight) / sum(weight)) × 100
```

Tier 1 covers foundational concerns (device registration, operators and memory for functional; isolation, encryption, and scrubbing for security). The weighting reflects that foundational readiness matters more than ecosystem polish.


## Output

Reports are written to `torch-air-report/`:

```
torch-air-report/torch_readiness_report_<backend>.md      # functional integration
torch-air-report/torch_security_readiness_report_<backend>.md   # security posture
```

## Repository Structure

```
torch-air/
├── VERSION                           # torch-air release version (recorded in reports)
├── SKILL.md                          # Orchestrator skill (functional + security flags)
|
├── frameworks/
│   └── pytorch/
│       ├── EVAL.md                   # PyTorch evaluation phases (Part 1: functional, Part 2: security)
│       ├── checklist.md              # Functional readiness checklist backends
│       ├── research_template_private.md  # Narrative research template for private backends
│       └── security/
│           ├── EVAL.md               # Security evaluation phases and scoring rules
│           └── checklist.md          # Security readiness checklist template (37 items)
├── crcr/
│   └── crcr-l1-onboarding.md        # CRCR Level 1 onboarding guide
└── README.md
```

Adding a new framework: create `frameworks/<name>/` with `EVAL.md` and `checklist.md`, then add the framework to the dispatch table in `SKILL.md`. Security dimensions for a framework live under `frameworks/<name>/security/`.
