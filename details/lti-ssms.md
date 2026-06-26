# Linear Time-Invariant (LTI) SSMs

## Overview
LTI State Space Models maintain constant transition matrices throughout a sequence. This LTI property allows them to use both convolutional representations for training and recurrent representations for inference.

## Architecture Diagram
```mermaid
graph LR
    Input[x_t] -->|Fixed A, B, C| State[h_t = A_bar h_{t-1} + B_bar x_t]
    State --> Output[y_t = C_bar h_t]
```

## Technical Details
### Mathematical Characteristics
An SSM is Linear Time-Invariant if its state matrices do not change across sequence time-steps:
$$A(t) = A, \quad B(t) = B, \quad C(t) = C$$
This allows the model to be unrolled in two distinct mathematical ways:
1. **Recurrent Representation (Inference):**
   $$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
   $$y_t = C h_t$$
2. **Convolutional Representation (Training):**
   $$y = K * x$$
   where $K$ is the SSM convolution kernel: $K_s = C\bar{A}^s\bar{B}$.

### Pros & Cons
- **Pros:** Parallelized training via FFT, extremely fast recurrent inference.
- **Cons:** Inability to perform content-based routing (context-dependent selection), making it struggle with association, copying, and logical routing tasks.

## References
- Gu, A., Goel, K., & Ré, C. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces." *arXiv preprint arXiv:2111.00396*.
- Gupta, A., Mehta, A., & Jonathan, S. (2022). "Diagonal State Spaces are as Effective as Structured State Spaces." *NeurIPS*.

---
[← Back to README](../README.md)
