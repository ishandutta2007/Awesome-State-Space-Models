# The Linear Memory Advantage (O(1) Inference Cache)

## Overview
SSMs offer a constant-size latent state memory footprint during inference, solving the Key-Value (KV) cache inflation issue that limits Transformer serving scalability.

## Architecture Diagram
```mermaid
graph TD
    subgraph Transformer Cache
        T_KV[KV Cache grows linearly with sequence length: O(N)]
    end
    subgraph SSM Cache
        SSM_State[Latent state size remains constant: O(1) per token]
    end
```

## Technical Details
### KV Cache vs. Constant Latent State
In traditional Transformers, the key-value pairs of all historical tokens must be cached to compute self-attention. This memory footprint grows linearly with sequence length:
$$\text{Memory}_{\text{KV}} \propto \text{Batch Size} \times \text{Sequence Length} \times \text{Hidden Dimension}$$
This limits maximum context length and batch size on standard GPUs.

In contrast, SSMs unroll as recurrences during generation. The entire history is compressed into a fixed-size latent state $h_t$:
$$\text{Memory}_{\text{SSM}} \propto \text{Batch Size} \times \text{State Dimension}$$
The memory footprint remains completely constant ($O(1)$) regardless of whether generating the 10th or 1,000,000th token.

## References
- Gu, A., Goel, K., & Ré, C. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces." *arXiv preprint arXiv:2111.00396*.

---
[← Back to README](../README.md)
