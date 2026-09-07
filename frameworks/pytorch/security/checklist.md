# PyTorch Accelerator Security Readiness Checklist

> Evaluates the security posture of Out-of-Tree (OOT) hardware accelerator backends
> integrating with PyTorch via PrivateUse1.

| Field | Value |
|-------|-------|
| **Accelerator** | _[FILL: accelerator name]_ |
| **Backend package** | _[FILL: e.g. torch-spyre, torch_npu]_ |
| **Version evaluated** | _[FILL: release version or commit]_ |
| **Evaluation date** | _[FILL: date]_ |
| **Evaluator** | _[FILL: human or agent]_ |
| **torch-air version** | _[FILL: read from VERSION in torch-air repo, or git describe --tags --always]_ |
| **PyTorch version** | _[FILL: PyTorch version supported/tested by the backend — from setup.py, pyproject.toml, requirements, README compatibility table, or CI config]_ |
| **Model** | _[FILL: AI model name used to generate this report]_ |
| **Source** | _[FILL: repo URL]_ |
| **Deployment model** | _[FILL: single-tenant / multi-tenant / cloud]_ |

---

## Readiness Score & Summary

### Executive Summary

_[FILL: Write a concise summary covering:
- **Overall Security Readiness**: X%
- **Deployment context**: single-tenant vs shared infrastructure
- **Key strengths**: top security measures evidenced
- **Critical gaps**: highest-priority security gaps
- **Vendor engagement needed**: yes/no and what topics]_

### Scoring Model

Every row has a max score of **2** and a **priority** (1-3):
- **Priority 1** = Critical (blocks safe deployment) — weight 1.000
- **Priority 2** = Important (expected for production) — weight 0.500
- **Priority 3** = Nice-to-have (defense-in-depth) — weight 0.333

Row weight: `w = 1 / priority`

Each domain has a **level** (1-3). Level 1 domains are weighted highest.

**Points column**: 2 = fully evidenced, 1 = partially evidenced, 0 = absent/undocumented, N/A = excluded

**Domain score**: `percentage = sum(score_i * w_i) / sum(max_i * w_i) * 100` (max_i = 2 for non-N/A rows)

**Overall score**:
```
weight_r = 1 / level
Security Readiness (%) = (sum(domain_pct * weight_r) / sum(weight_r)) * 100
```

### Domain Scores

| Domain | Max Pts | Earned | Percentage |
|--------|---------|--------|------------|
| **Level 1** | | | |
| Multi-Tenant Isolation (SEC-MT) | | | |
| Device Memory Encryption (SEC-ME) | | | |
| Data Scrubbing (SEC-DS) | | | |
| **Level 2** | | | |
| Host-Device Transit Security (SEC-HT) | | | |
| Firmware & Driver (SEC-FD) | | | |
| **Level 3** | | | |
| PyTorch Integration Surface (SEC-PI) | | | |

### Calculation

```
Row weight: w_i = 1 / priority_i
Domain % = sum(score_i * w_i) / sum(max_i * w_i) * 100  (excluding N/A rows, max_i = 2)
weight_r = 1 / level
Security Readiness (%) = (sum(domain_pct * weight_r) / sum(weight_r)) * 100
```

---

## SEC-MT: Multi-Tenant Isolation

**Threat:** Cross-tenant data leakage, compute interference, side-channel attacks.

| ID | Check | Priority | Points | Notes |
|----|-------|----------|--------|-------|
| SEC-MT-01 | Hardware-level partitioning support | 1 | | |
| SEC-MT-02 | Memory isolation between tenants (SMMU/IOMMU) | 1 | | |
| SEC-MT-03 | Compute isolation (dedicated execution units) | 2 | | |
| SEC-MT-04 | Temporal isolation (memory scrubbing on tenant switch) | 1 | | |
| SEC-MT-05 | Fault isolation (crash containment) | 2 | | |
| SEC-MT-06 | Resource allocation granularity | 3 | | |

---

## SEC-ME: Device Memory Encryption

**Threat:** Physical access attacks, host compromise, model IP theft, regulatory non-compliance.

| ID | Check | Priority | Points | Notes |
|----|-------|----------|--------|-------|
| SEC-ME-01 | At-rest encryption of device memory | 1 | | |
| SEC-ME-02 | Hardware Root of Trust present | 1 | | |
| SEC-ME-03 | Key management (on-device vs host-provided) | 1 | | |
| SEC-ME-04 | Encrypted regions documented (full/partial) | 2 | | |
| SEC-ME-05 | Runtime data-in-use protection | 2 | | |

---

## SEC-DS: Data Scrubbing

**Threat:** Memory residue between sessions leaking KV cache, model fragments, or activations.

| ID | Check | Priority | Points | Notes |
|----|-------|----------|--------|-------|
| SEC-DS-01 | Memory zeroing between workload transitions | 1 | | |
| SEC-DS-02 | KV cache clearing between sessions (LLM) | 1 | | |
| SEC-DS-03 | Intermediate result cleanup | 2 | | |
| SEC-DS-04 | Scrubbing verification mechanism | 2 | | |
| SEC-DS-05 | Data minimization in transfer (encryption preferred) | 3 | | |

---

## SEC-HT: Host-Device Transit Security

**Threat:** PCIe bus snooping, DMA attacks, host buffer exposure, inter-device interception.

| ID | Check | Priority | Points | Notes |
|----|-------|----------|--------|-------|
| SEC-HT-01 | PCIe link encryption (AES-GCM or equivalent) | 1 | | |
| SEC-HT-02 | Host buffer protection (IOMMU/SMMU) | 1 | | |
| SEC-HT-03 | PII-aware data handling in transfer path | 2 | | |
| SEC-HT-04 | Device buffer access control | 1 | | |
| SEC-HT-05 | Inter-device communication encryption | 2 | | |
| SEC-HT-06 | Channel attestation (SPDM or equivalent) | 2 | | |

---

## SEC-FD: Firmware & Driver

**Threat:** Firmware compromise, driver vulnerabilities, kernel-level host compromise.

| ID | Check | Priority | Points | Notes |
|----|-------|----------|--------|-------|
| SEC-FD-01 | Secure boot (signed firmware verification) | 1 | | |
| SEC-FD-02 | Authenticated firmware updates | 1 | | |
| SEC-FD-03 | Anti-rollback protection | 2 | | |
| SEC-FD-04 | Memory safety (no CWE-119/122/190/401/680) | 1 | | |
| SEC-FD-05 | Vulnerability disclosure process | 1 | | |
| SEC-FD-06 | DoS resistance (resource limits) | 2 | | |
| SEC-FD-07 | Documentation-code consistency | 3 | | |
| SEC-FD-08 | Network surface minimization | 2 | | |

---

## SEC-PI: PyTorch Integration Surface

**Threat:** Memory corruption at the PyTorch/device trust boundary, information leakage, supply chain compromise.

| ID | Check | Priority | Points | Notes |
|----|-------|----------|--------|-------|
| SEC-PI-01 | OOT test coverage for security-relevant paths | 2 | | |
| SEC-PI-02 | Memory leak detection in CI | 1 | | |
| SEC-PI-03 | SAST on bridge code (C++/Python) | 1 | | |
| SEC-PI-04 | Security issue attribution (device vs PyTorch) | 2 | | |
| SEC-PI-05 | Input validation at dispatch boundary | 1 | | |
| SEC-PI-06 | Error handling without information leakage | 2 | | |
| SEC-PI-07 | CRCR security test integration | 3 | | |

---

## Gap Analysis (Prioritized)

### CRITICAL

_[FILL: gaps that block production deployment in shared infrastructure]_

### HIGH

_[FILL: gaps requiring vendor engagement]_

### MEDIUM

_[FILL: defense-in-depth gaps]_

### LOW

_[FILL: informational findings]_

---

## Recommendations

_[FILL: actionable next steps, prioritized]_

---

## Evidence Sources

_[FILL: all URLs, documents, and code paths referenced]_
