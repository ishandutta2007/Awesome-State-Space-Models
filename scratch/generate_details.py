import os

# Change working directory to the parent of the scratch folder (repository root)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(repo_root)

# Define the 13 topics with details
topics = {
    "continuous-control-era": {
        "title": "The Continuous Control Era (Classical Roots)",
        "description": "Continuous State Space Models (SSMs) originated in classical control theory in the mid-20th century. Mathematically, they represent physical systems through systems of continuous differential equations, mapping an input signal x(t) to an output y(t) through a latent state h(t).",
        "diagram": """graph TD
    Input[Input Signal: x(t)] -->|B| State[State Equation: dh(t)/dt = Ah(t) + Bx(t)]
    State -->|C| Output[Output Signal: y(t) = Ch(t) + Dx(t)]
    State -->|A| State""",
        "details": """### Mathematical Formulation
The continuous-time linear time-invariant (LTI) state space model is formulated as:
$$\\dot{h}(t) = A h(t) + B x(t)$$
$$y(t) = C h(t) + D x(t)$$
where:
- $h(t) \\in \\mathbb{R}^N$ is the latent state.
- $x(t) \\in \\mathbb{R}$ is the input.
- $y(t) \\in \\mathbb{R}$ is the output.
- $A, B, C, D$ are system transition matrices.

### Limitations in Deep Learning
Classical continuous-time SSMs struggled in deep learning contexts due to:
1. **Vanishing/Exploding Gradients:** Integrating differential equations over long intervals led to numerical instability during backpropagation.
2. **Computational Inefficiency:** Sequential integration hindered parallel training on modern hardware like GPUs.
3. **Discretization Challenges:** Adapting continuous models to discrete-time sequences (like text tokens) required robust mathematical discretization schemes.""",
        "references": "- Kalman, R.E. (1960). \"A New Approach to Linear Filtering and Prediction Problems.\" *Journal of Basic Engineering*.\n- Sontag, E.D. (1998). *Mathematical Control Theory: Deterministic Finite Dimensional Systems*."
    },
    "structured-lti-era": {
        "title": "The Structured Linear Time-Invariant Era (S4)",
        "description": "The Structured State Space (S4) model introduced structured transition matrices (specifically using HiPPO matrix initialization) to successfully model long-range dependencies in deep learning, enabling parallelized training like CNNs and fast inference like RNNs.",
        "diagram": """graph LR
    Discretization["Continuous SSM & Delta"] -->|Bilinear / ZOH| Discrete["Discrete SSM (A_bar, B_bar)"]
    Discrete -->|Parallel Training| CNN["Global Convolution: O(N log N)"]
    Discrete -->|Fast Serving| RNN["Recurrent Inference: O(1) step"]""",
        "details": """### HiPPO Matrix Initialization
The S4 model addresses the vanishing/exploding gradient problem by using the **High-order Polynomial Projection Operator (HiPPO)** framework. The transition matrix $A \\in \\mathbb{R}^{N \\times N}$ is initialized to a structured matrix that projects the history of the input signal onto Legendre polynomials:
$$A_{nk} = -\\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \\text{if } n > k \\\\ n+1 & \\text{if } n = k \\\\ 0 & \\text{if } n < k \\end{cases}$$
This allows the state $h(t)$ to mathematically memorize historical inputs over long horizons.

### Linear Time-Invariance (LTI)
Because the matrices $A$, $B$, and $C$ are independent of time $t$ and input $x(t)$, the recurrence relation can be unrolled globally as a single non-local convolution:
$$\\bar{K} = (C\\bar{B}, C\\bar{A}\\bar{B}, \\dots, C\\bar{A}^{L-1}\\bar{B})$$
This allows the entire sequence to be computed in parallel during training using Fast Fourier Transforms (FFT).""",
        "references": "- Gu, A., Goel, K., & Ré, C. (2021). \"Efficiently Modeling Long Sequences with Structured State Spaces.\" *arXiv preprint arXiv:2111.00396*.\n- Gu, A., Dao, T., Ermon, S., Atzmon, Y., & Ré, C. (2020). \"HiPPO: Recurrent Memory with Optimal Polynomial Projections.\" *NeurIPS*."
    },
    "selective-hardware-aware-era": {
        "title": "The Selective & Hardware-Aware Era (Mamba)",
        "description": "Mamba introduced input-dependent selective scan mechanisms to enable SSMs to filter relevant information dynamically, combined with hardware-aware memory management to bypass the training latency of non-LTI systems.",
        "diagram": """graph TD
    Input[Incoming Token x_t] -->|Computes| Params[Dynamic B_t, C_t, Delta_t]
    Params -->|SRAM Fused Kernel| Scan[Selective Scan Loop]
    Scan -->|Evicts/Updates| Latent[Latent State h_t]""",
        "details": """### Selection Mechanism
Mamba removes the Linear Time-Invariant constraint by making the transition parameters functions of the input:
$$B_t = s_B(x_t), \\quad C_t = s_C(x_t), \\quad \\Delta_t = \\text{softplus}(w_\\Delta + s_\\Delta(x_t))$$
This turns the LTI system into a Linear Time-Varying (LTV) system. The model can choose what to memorize and what to discard, resolving the fine-grained context-switching and copying deficits of LTI models.

### Hardware-Aware Optimization
Because LTV systems cannot be unrolled as convolutions, sequential scans would normally run slowly on GPUs. Mamba resolves this by:
1. **Memory Hierarchy Exploitation:** Keeping the state transitions within GPU **SRAM** rather than reading/writing to slower **HBM** (High Bandwidth Memory).
2. **Kernel Fusion:** Fusing the discretization and recurrent scan into a single custom CUDA kernel.""",
        "references": "- Gu, A., & Dao, T. (2023). \"Mamba: Linear-Time Sequence Modeling with Selective State Spaces.\" *arXiv preprint arXiv:2312.00752*."
    },
    "lti-ssms": {
        "title": "Linear Time-Invariant (LTI) SSMs",
        "description": "LTI State Space Models maintain constant transition matrices throughout a sequence. This LTI property allows them to use both convolutional representations for training and recurrent representations for inference.",
        "diagram": """graph LR
    Input[x_t] -->|Fixed A, B, C| State[h_t = A_bar h_{t-1} + B_bar x_t]
    State --> Output[y_t = C_bar h_t]""",
        "details": """### Mathematical Characteristics
An SSM is Linear Time-Invariant if its state matrices do not change across sequence time-steps:
$$A(t) = A, \\quad B(t) = B, \\quad C(t) = C$$
This allows the model to be unrolled in two distinct mathematical ways:
1. **Recurrent Representation (Inference):**
   $$h_t = \\bar{A}h_{t-1} + \\bar{B}x_t$$
   $$y_t = C h_t$$
2. **Convolutional Representation (Training):**
   $$y = K * x$$
   where $K$ is the SSM convolution kernel: $K_s = C\\bar{A}^s\\bar{B}$.

### Pros & Cons
- **Pros:** Parallelized training via FFT, extremely fast recurrent inference.
- **Cons:** Inability to perform content-based routing (context-dependent selection), making it struggle with association, copying, and logical routing tasks.""",
        "references": "- Gu, A., Goel, K., & Ré, C. (2021). \"Efficiently Modeling Long Sequences with Structured State Spaces.\" *arXiv preprint arXiv:2111.00396*.\n- Gupta, A., Mehta, A., & Jonathan, S. (2022). \"Diagonal State Spaces are as Effective as Structured State Spaces.\" *NeurIPS*."
    },
    "selective-ltv-ssms": {
        "title": "Selective / Linear Time-Varying (LTV) SSMs",
        "description": "LTV SSMs make their state transition matrices dependent on the incoming input tokens. This selection mechanism allows the model to filter information dynamically, matching the capability of Transformers.",
        "diagram": """graph TD
    Input[Token: x_t] --> MatrixGen[Dynamic Parameters Generator]
    MatrixGen -->|A_t, B_t, C_t| StateUpdate[h_t = A_t h_{t-1} + B_t x_t]
    StateUpdate --> Output[y_t = C_t h_t]""",
        "details": """### The Selective Mechanism
Unlike LTI models, Selective SSMs compute parameters dynamically at each time step:
$$B_t = s_B(x_t), \\quad C_t = s_C(x_t), \\quad \\Delta_t = \\text{softplus}(w_\\Delta + s_\\Delta(x_t))$$
This allows the model to perform content-aware reasoning:
- **Remembering:** Keep the state updated if the input token is highly informative.
- **Forgetting:** Scale $\\Delta_t$ to zero or near-zero to ignore noise or irrelevant tokens.

### Computation
Because the parameters are time-dependent, we cannot use FFT for parallelization. Instead, we rely on a custom parallel associative scan implemented in GPU CUDA kernels, which achieves the same computational complexity order ($O(N)$) as standard recurrent models.""",
        "references": "- Gu, A., & Dao, T. (2023). \"Mamba: Linear-Time Sequence Modeling with Selective State Spaces.\" *arXiv preprint arXiv:2312.00752*."
    },
    "structured-state-space-duals": {
        "title": "Structured State Space Duals (SSD / Mamba-2)",
        "description": "Structured State Space Duals (SSD) establishes a mathematical bridge between SSMs and structured attention mechanisms, enabling the use of Tensor Core matrix multiplication to double training speeds.",
        "diagram": """graph LR
    SSM[State Space Models] <-->|Mathematical Duality| Attention[Structured Attention]
    Attention -->|Hardware Path| TensorCore["GPU Tensor Cores (MatMul)"]
    TensorCore --> Speedup["2x Training Throughput"]""",
        "details": """### The Dual Representation
Mamba-2 proves that a class of selective SSMs can be represented as a form of attention with a structured mask. Specifically, the state update:
$$h_i = A_i h_{i-1} + B_i x_i$$
$$y_i = C_i h_i$$
can be written as a matrix multiplication:
$$Y = (C \\cdot H) \\cdot B^T \\odot M$$
where $M$ is a structured mask matrix.

### Hardware Benefits
Standard GPU architectures are highly optimized for Matrix Multiplication (GEMM) via Tensor Cores. By formulating the state space scan as a structured matrix multiplication, Mamba-2 bypasses compiler memory-bound limitations, yielding over $2\\times$ improvement in training throughput compared to Mamba-1.""",
        "references": "- Dao, T., & Gu, A. (2024). \"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duals.\" *arXiv preprint arXiv:2405.21060*."
    },
    "ssm-attention-hybrids": {
        "title": "SSM-Attention Hybrids",
        "description": "SSM-Attention Hybrids interleave linear State Space Model layers with self-attention layers to combine the infinite-context compression of SSMs with the exact retrieval capabilities of attention.",
        "diagram": """graph LR
    Input --> SSM1[SSM Layer] --> Attn[Attention Layer] --> SSM2[SSM Layer] --> Output""",
        "details": """### Core Architecture
While pure SSMs scale linearly, they can struggle with exact retrieval over extremely long context windows. Hybrid models resolve this by interleaving blocks:
- **SSM Layers (e.g., Mamba):** Handle sequence modeling over long distances with $O(N)$ efficiency.
- **Attention Layers:** Perform localized or sparse full-context cross-retrieval to prevent information degradation.

### Examples
- **H3 (2022):** The first model to interleave state spaces with attention.
- **Jamba (2024):** Integrates Mamba and Attention blocks along with Mixture-of-Experts.
- **RecurrentGemma (2024):** A commercial-grade hybrid model from Google utilizing recurrent linear blocks and attention.""",
        "references": "- Fu, D.Y., Sabharwal, A., Soltani, B., & Ré, C. (2022). \"Hungry Hungry Hippos: Towards Language Modeling with State Space Models.\" *arXiv preprint arXiv:2212.14052*.\n- AI21 Labs (2024). \"Jamba: A Hybrid Transformer-Mamba Language Model.\" *arXiv preprint arXiv:2403.19887*."
    },
    "ssm-moe-hybrids": {
        "title": "SSM-MoE Hybrids (Mixture-of-Experts SSMs)",
        "description": "SSM-MoE Hybrids combine state space models with sparse Mixture-of-Experts (MoE) routing, creating massive model parameter scaling with low compute overhead.",
        "diagram": """graph TD
    Input[Incoming Token] --> SSM[SSM Layer]
    SSM --> Router[MoE Router]
    Router -->|Expert 1| E1[Feed-Forward Expert 1]
    Router -->|Expert 2| E2[Feed-Forward Expert 2]
    E1 & E2 --> Combine[Output Combination]""",
        "details": """### Architectural Integration
Mixture-of-Experts (MoE) routes inputs to a subset of neural network 'experts' dynamically. Combining MoE with SSMs:
1. Replaces the dense Feed-Forward Network (FFN) blocks of the SSM layer with an MoE block.
2. Maintains linear context scaling while scaling parameters to hundreds of billions.

### Key Benefits
- **Scale:** High capacity models with sparse activation.
- **Throughput:** Keeps FLOPs-per-token low, ensuring fast execution during both training and inference.
- **Memory Footprint:** Extremely small KV cache equivalent compared to standard Transformer MoEs.""",
        "references": "- Anthony, Q., Tokpanov, Y., Glorioso, P., & Millidge, B. (2024). \"BlackMamba: Mixture of Experts for State-Space Models.\" *arXiv preprint arXiv:2402.01771*."
    },
    "linear-memory-advantage": {
        "title": "The Linear Memory Advantage (O(1) Inference Cache)",
        "description": "SSMs offer a constant-size latent state memory footprint during inference, solving the Key-Value (KV) cache inflation issue that limits Transformer serving scalability.",
        "diagram": """graph TD
    subgraph Transformer Cache
        T_KV[KV Cache grows linearly with sequence length: O(N)]
    end
    subgraph SSM Cache
        SSM_State[Latent state size remains constant: O(1) per token]
    end""",
        "details": """### KV Cache vs. Constant Latent State
In traditional Transformers, the key-value pairs of all historical tokens must be cached to compute self-attention. This memory footprint grows linearly with sequence length:
$$\\text{Memory}_{\\text{KV}} \\propto \\text{Batch Size} \\times \\text{Sequence Length} \\times \\text{Hidden Dimension}$$
This limits maximum context length and batch size on standard GPUs.

In contrast, SSMs unroll as recurrences during generation. The entire history is compressed into a fixed-size latent state $h_t$:
$$\\text{Memory}_{\\text{SSM}} \\propto \\text{Batch Size} \\times \\text{State Dimension}$$
The memory footprint remains completely constant ($O(1)$) regardless of whether generating the 10th or 1,000,000th token.""",
        "references": "- Gu, A., Goel, K., & Ré, C. (2021). \"Efficiently Modeling Long Sequences with Structured State Spaces.\" *arXiv preprint arXiv:2111.00396*."
    },
    "needle-in-a-haystack-deficit": {
        "title": "The \"Needle in a Haystack\" Retrieval Deficit",
        "description": "Due to compressing sequence history into a fixed-size hidden state, pure SSMs can struggle to retrieve specific, long-tail facts buried deep inside massive contexts.",
        "diagram": """graph LR
    Input[Massive Text / Haystack] -->|Compression| Latent[Fixed-Size Vector State]
    Latent -->|Lossy Retrieval| Output[Missed specific fact / Needle]""",
        "details": """### The Compression Bottleneck
The mathematical strength of SSMs—compressing history into a fixed-size latent state to achieve constant memory footprint—is also their primary limitation. 
- **Lossy Compression:** Compressing a million-token sequence into a fixed-dimensional vector space is mathematically lossy.
- **Retrieval Deficit:** Standard Transformers keep all raw historical tokens in memory (KV Cache), letting them perform precise retrieval. SSMs must rely on the latent representation, which can overwrite sparse or isolated facts (\"needles\") when new information flows in.

### Mitigation Strategies
1. **Interleaved Attention Layers:** Inserting sparse self-attention layers to recover exact token indexing.
2. **Complex Updates (Mamba-3):** Enhancing state-space representations to track high-entropy state changes better.""",
        "references": "- Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). \"Lost in the Middle: How Language Models Use Long Contexts.\" *arXiv preprint arXiv:2307.03172*.\n- De, S., et al. (2024). \"Griffin: Mixing Gated Linear Recurrent Units with Local Attention for Efficient Language Models.\" *arXiv preprint arXiv:2402.19427*."
    },
    "ultra-long-context-document-legal-auditing": {
        "title": "Ultra-Long Context Document & Legal Auditing",
        "description": "State Space Models scale context windows to millions of tokens, enabling the ingestion of entire legal databases and codebases for auditing without memory crashes.",
        "diagram": """graph LR
    Docs[Millions of Legal/Code Tokens] -->|Linear Scaling| SSM[SSM Forward Pass]
    SSM --> Analysis[Entity Linking, Clause Extraction, Anomaly Detection]""",
        "details": """### Real-World Relevance
In fields such as legal auditing, compliance, and systems engineering, documents can span hundreds of thousands or millions of tokens.
- **Transformer Limitation:** Processing a 1,000,000 token document requires Terabytes of GPU memory just for the attention matrix.
- **SSM Capability:** SSMs scale linearly ($O(N)$) during training and maintain a constant memory profile during inference, permitting full-document processing on single, commodity GPU nodes.

### Applications
- **Legal Compliance:** Ingesting multiple regulatory acts to flag anomalies or contradictory clauses.
- **Codebase Auditing:** Evaluating complex software architectures for structural flaws or security holes in a single context window.""",
        "references": "- Gu, A., & Dao, T. (2023). \"Mamba: Linear-Time Sequence Modeling with Selective State Spaces.\" *arXiv preprint arXiv:2312.00752*."
    },
    "high-resolution-1d-biomedical-genomic-mapping": {
        "title": "High-Resolution 1D Biomedical Genomic Mapping",
        "description": "Biomedical sequences like DNA span millions of base-pairs. Models like Caduceus leverage bidirectional SSM layouts to analyze genomic structures and long-range sequence mutations efficiently.",
        "diagram": """graph LR
    DNA[DNA Strand: A, C, G, T] -->|Bidirectional SSM| Latent[Sequence Embeddings]
    Latent --> Analysis[Variant Effect Prediction / Gene Identification]""",
        "details": """### The Genomic Sequence Challenge
Genomic sequencing datasets consist of DNA/RNA bases ($A, C, G, T$) stretching over billions of steps.
1. **Long-Range Dependencies:** Mutations in non-coding regions can affect gene expression millions of base-pairs away.
2. **Quadratic Scaling Failure:** Attention models are unable to scale to context windows of this size.

### The SSM Solution
Genomic models like **Caduceus** leverage bidirectional State Space Layers:
- **Bidirectionality:** Since DNA is double-stranded and has no natural 'left-to-right' direction, bidirectional state tracking is critical.
- **Linear Efficiency:** Captures multi-megabase contexts to model DNA structural dependencies without computational bottlenecks.""",
        "references": "- Schiff, L., et al. (2024). \"Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling.\" *arXiv preprint arXiv:2403.03234*."
    },
    "continuous-streaming-time-series-audio-analytics": {
        "title": "Continuous Streaming Time-Series & Audio Analytics",
        "description": "Because SSMs are defined using continuous differential equations, they excel at modeling continuous physical signals such as audio, seismic telemetry, and medical sensor streams.",
        "diagram": """graph TD
    Continuous[Continuous Signal] -->|Discretization Delta| Discrete[SSM Layer]
    Discrete --> Prediction[Classification / Sound Generation]""",
        "details": """### Continuous-to-Discrete Representation
Physical sensors generate high-frequency continuous data. Because SSMs are rooted in continuous-time differential equations:
1. **Signal Discretization:** The discretization step size $\\Delta$ acts as a sampling rate filter.
2. **Multi-Rate Modeling:** The model can adjust to different telemetry sampling rates by simply changing $\\Delta$ during evaluation, without retraining.

### Key Domains
- **Raw Audio Processing:** SaShiMi uses structured SSMs to generate raw waveforms, capturing long-term audio coherence.
- **Biomedical Sensor Telemetry:** Tracking continuous ECG or EEG signals in ICU patient monitoring pipelines.""",
        "references": "- Goel, K., Gu, A., Donahue, C., & Ré, C. (2022). \"It's Raw! Audio Generation with State-Space Models.\" *ICML*.\n- Alcaraz, J. C. L., & Strijbos, M. (2022). \"State Space Models for Multivariate Time Series Forecasting.\" *arXiv preprint arXiv:2207.01211*."
    }
}

# Create details directory if it doesn't exist
os.makedirs("details", exist_ok=True)

# Generate markdown files
for filename, content in topics.items():
    filepath = os.path.join("details", f"{filename}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"""# {content['title']}

## Overview
{content['description']}

## Architecture Diagram
```mermaid
{content['diagram']}
```

## Technical Details
{content['details']}

## References
{content['references']}

---
[← Back to README](../README.md)
""")
    print(f"Generated {filepath}")
