# The Continuous Control Era (Classical Roots)

## Overview
Continuous State Space Models (SSMs) originated in classical control theory in the mid-20th century. Mathematically, they represent physical systems through systems of continuous differential equations, mapping an input signal x(t) to an output y(t) through a latent state h(t).

## Architecture Diagram
```mermaid
graph TD
    Input[Input Signal: x(t)] -->|B| State[State Equation: dh(t)/dt = Ah(t) + Bx(t)]
    State -->|C| Output[Output Signal: y(t) = Ch(t) + Dx(t)]
    State -->|A| State
```

## Technical Details
### Mathematical Formulation
The continuous-time linear time-invariant (LTI) state space model is formulated as:
$$\dot{h}(t) = A h(t) + B x(t)$$
$$y(t) = C h(t) + D x(t)$$
where:
- $h(t) \in \mathbb{R}^N$ is the latent state.
- $x(t) \in \mathbb{R}$ is the input.
- $y(t) \in \mathbb{R}$ is the output.
- $A, B, C, D$ are system transition matrices.

### Limitations in Deep Learning
Classical continuous-time SSMs struggled in deep learning contexts due to:
1. **Vanishing/Exploding Gradients:** Integrating differential equations over long intervals led to numerical instability during backpropagation.
2. **Computational Inefficiency:** Sequential integration hindered parallel training on modern hardware like GPUs.
3. **Discretization Challenges:** Adapting continuous models to discrete-time sequences (like text tokens) required robust mathematical discretization schemes.

## References
- Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering*.
- Sontag, E.D. (1998). *Mathematical Control Theory: Deterministic Finite Dimensional Systems*.

---
[← Back to README](../README.md)
