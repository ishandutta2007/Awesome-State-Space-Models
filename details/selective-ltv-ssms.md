# Selective / Linear Time-Varying (LTV) SSMs

## Overview
LTV SSMs make their state transition matrices dependent on the incoming input tokens. This selection mechanism allows the model to filter information dynamically, matching the capability of Transformers.

## Architecture Diagram
```mermaid
graph TD
    Input[Token: x_t] --> MatrixGen[Dynamic Parameters Generator]
    MatrixGen -->|A_t, B_t, C_t| StateUpdate[h_t = A_t h_{t-1} + B_t x_t]
    StateUpdate --> Output[y_t = C_t h_t]
```

## Technical Details
### The Selective Mechanism
Unlike LTI models, Selective SSMs compute parameters dynamically at each time step:
$$B_t = s_B(x_t), \quad C_t = s_C(x_t), \quad \Delta_t = \text{softplus}(w_\Delta + s_\Delta(x_t))$$
This allows the model to perform content-aware reasoning:
- **Remembering:** Keep the state updated if the input token is highly informative.
- **Forgetting:** Scale $\Delta_t$ to zero or near-zero to ignore noise or irrelevant tokens.

### Computation
Because the parameters are time-dependent, we cannot use FFT for parallelization. Instead, we rely on a custom parallel associative scan implemented in GPU CUDA kernels, which achieves the same computational complexity order ($O(N)$) as standard recurrent models.

## References
- Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv preprint arXiv:2312.00752*.

---
[← Back to README](../README.md)
