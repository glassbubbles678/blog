# BitNet: Scaling 1-bit Transformers for Large Language Models

[Paper](https://arxiv.org/pdf/2310.11453.pdf)

This paper proposes BitNet, a 1-bit Transformer architecture for LLMs that aims to scale efficiently in terms of both memory and computation.

## Introduction

**Issues with LLMs**

- They are expensive to host due to high inference cost and energy consumption.
- The memory bandwidth required for accessing and processing model parameters becomes a bottleneck.
- When deployed on distributed systems or multi-device platforms, the inter-device communication overhead impacts the inference latency and energy consumption.

**Model quantization**

- Model quantization is a solution to reduce memory footprint and computational cost while maintaining competitive performance.

1. Post-training model quantization
- These methods do not require changes to the training pipeline or retraining the model, so they are simple and easy to apply.
- However, the model is not optimized for the quantized representation during training which can lead to a loss in accuracy.
2. Quantization-aware training
- The model is trained to account for reduced precision from the start, typically resulting in better accuracy.
- It also allows continued training and finetuning, which is essential for LLMs.
- However, it becomes increasingly difficult for the model to converge as the precision decreases.
- Further, it is unknown whether quantization-aware training follows the scaling laws of neural language models.

**1-bit LLMs**

- BitNet explores quantization-aware training for 1-bit LLMs.
- It employs low-precision binary weights and activations, and maintains high-precision for optimizer states and gradients during training.
- The implementation only requires replacing linear projections (i.e. `nn.Linear` in PyTorch) in the Transformer.

## BitNet

- BitNet uses `BitLinear` which employs binarized model weights instead of conventional matrix multiplication.
- The remaining components are left in high-precision (8-bit) for the following reasons:
    - Residual connections and layer normalizations contribute negligible computation costs to LLMs.
    - The computation cost of QKV transformation is much smaller than the parametric projection as the model grows larger.
    - The precision is preserved for input/output embeddings since the language models have to use high-precision probabilities to perform sampling.

### BitLinear

**Quantization of weights**

- The weights are binarized to +1/-1.
- The weights are centralized to be zero-mean before binarization to increase capacity within a limited numerical range.
- A scaling factor of $\beta$ is used after binarization to reduce the $\ell_2$ error between the real-valued and binarized weights.

$$\widetilde{W} = \text{Sign}(W - \alpha)$$

$$
\text{Sign}(W_{ij}) = 
\begin{cases}
+1, & \text{if } W_{ij} > 0 \\
-1, & \text{if } W_{ij} \leq 0 \\
\end{cases}
$$

$$\alpha = \frac{1}{nm} \sum_{ij} W_{ij}$$

**Quantization of activations**

- The activations are quantized to $b$-bit precision (8-bit in this work).
- The abysmax quantization technique is used which scales activations into the range $[-Q_b,Q_b]$ ($Q_b = 2^{b-1}$) by multiplying with $Q_b$ and dividing by the absolute maximum of the input matrix.

$$\tilde{x} = \text{Quant}(x) = \text{Clip}(x \times \frac{Q_b}{\gamma}, -Q_b+\epsilon, Q_b-\epsilon)$$

$$\text{Clip}(x,a,b) = \max(a, \min(b, x))$$

$$\gamma = \lVert x \rVert _\infty$$

- $\epsilon$ is a small floating point-number that prevents overflow while clipping.

- For activations before non-linear functions such as ReLU, they are scaled into the range $[0,Q_b]$ by subtracting the minimum of the inputs so that all values are non-negative.

$$\tilde{x} = \text{Quant}(x) = \text{Clip}((x - \eta) \times \frac{Q_b}{\gamma}, -Q_b+\epsilon, Q_b-\epsilon)$$

$$\eta = \min_{ij} x_{ij}$$

**Matrix multiplication**

- Using the above quantizations, the matrix multiplication can be written as:

$$y = \widetilde{W} \tilde{x}$$

- Assuming that within each of $W$ and $x$, the elements are mutually independent and share the same distribution, and $W$ and $x$ are independent, the variance of the output $y$ is estimated as:

$$\text{Var}(y) = n \text{Var}(\tilde{w}\tilde{x}) = n E[\tilde{w}^2] E[\tilde{x}^2] = n \beta^2 E[\tilde{x}^2] \approx E[\tilde{x}^2]$$

- For full-precision computation, the variance of $y$ is at the scale of 1 with the standard initialization methods (Kaiming, Xavier, etc.), which greatly benefits training stability.
- To preserve the variance after quantization, a LayerNorm function is introduced before activation quantization.
- The variance of the output is then:

$$\text{Var}(y) \approx E[\text{LN}(\tilde{x}^2)] = 1$$

- This is the same magnitude as the full-precision counterpart.
- The implementation is [SubLN](https://arxiv.org/pdf/2210.06423.pdf).

**Overall BitLinear Formulation**

$$y = \widetilde{W} \tilde{x} = \widetilde{W} \text{Quant}(\text{Ln}(x)) \times \frac{\beta\gamma}{Q_b}$$

$$\text{LN}(x) = \frac{x - E[x]}{\sqrt{\text{Var}(x) + \epsilon}}$$

$$\beta = \frac{1}{nm} \lVert W \rVert _1$$

After the `SubLN` operation, the activations are quantized with the abysmax function. 
The matrix multiplication is performed between 1-bit weights and quantized activations. 
The output activations are then rescaled using $\{\beta, \gamma\}$ to dequantize them to the original precision.

**Group Quantization and Group Normalization for Model Parallelism**

- Model parallelism to scale up LLMs involves partitioning matrix multiplication across multiple devices.
- This requires the tensors to be independent along the partition dimension.
- However, the parameters $\alpha, \beta, \gamma$ and $\eta$ are calculated from whole tensors, breaking this prerequsite.
- One solution is to introduce an $\textit{all-reduce}$ operation for each parameter.
- However, although the communication for each parameter is small, the amount of synchronization grows as the model becomes deeper and the forward pass slows down significantly.
- This problem also exists in `SubLN` where the mean and variance need to be estimated across the partition dimension.
- The solution used is the divide the weights and activations into groups and independently estimate each group's parameters.
- The parameters can be computed locally without any communication.

**Group Quantization**

- For a weight matrix $W \in \mathcal{R}^{n \times m}$, it is divided into $G$ groups along the partition dimension.
- Each group has a size of $\frac{n}{G} \times m$.
- The parameters for each group are independently estimated as follows:

$$\alpha_g = \frac{n}{G} \times m \sum{ij} W_{ij}^{(g)}, \beta_g = \frac{n}{G} \times m \lVert W^{(g)} \rVert _1$$

- Similarly, for activations, the input matrix $x \in \mathcal{R}^{n \times m}$ can be divided into $G$ groups with parameters calculated as follows:

$$\gamma_g = \lVert x^{(g)} \rVert _\infty, \eta_g = \min_{ij} x^{(g)}_{ij}$$

- For layer normalization, group normalization is applied to compute the mean and variance for each group independently:

$$\text{LN}(x^{(g)}) = \frac{x^{(g)} - E[x^{(g)}]}{\sqrt{\text{Var}(x^{(g)}) + \epsilon}}$$

### Model Training

- Straight-through estimator: The straight-through estimator is used to approximate gradients during backpropagation. 
It bypasses non-differentiable functions such as Sign and Clip during the backward pass.

- Mixed-precision training: Low-precision binary weights and activations and high-precision optimizer states and gradients are used.
A high-precision latent weight is maintained for learnable parameters to accumulate parameter updates.
The latent weights are binarized on the fly during the forward pass and never used during inference.

- Large learning rate: A small update on latent weights may make no difference in the 1-bit weights.
To address this, the learning rate is increased to accelerate optimization. 
Unlike the FP16 Transformer that diverges if a large learning rate is used at the beginning of training, BitNet benefits from a large learning rate.

### Computation Efficiency

1. Arithmetic operations energy

- In vanilla Transformers, matrix multiplication with dimensions $m \times n$ and $n \times p$ has energy consumption as follows:

$$E\_{add} = m \times (n-1) \times p \times \hat{E}\_{add}$$

$$E\_{mul} = m \times n \times p \times \hat{E}\_{mul}$$

$\hat{E}\_{add}$ and $\hat{E}\_{mul}$ are the energy consumption for addition and multiplication operations.

- In BitNet, the energy consumption of matrix multiplication is domination by the addition operations since the weights are 1-bit.
- The multiplication operations are only applied to scale the outputs with the scalars $\beta$ and $\frac{\gamma}{Q_b}$, so the energy consumption for multiplication is:

$$E\_{mul} = (m \times n \times + m \times p) \times \hat{E}\_{mul}$$

## Comparison with FP16 Transformers

### Inference Optimal Scaling Law

- For the vanilla Transformer architecture, the loss scales as the power law with the amount of computation used for training.
- Plotting the scaling curve against the parameter count, the loss scaling of BitNet is found to be similar to the FP16 transformer.
- However, the power law does not properly model the relationship between loss and actual compute.
- While previous work estimated compute by calculating the FLOPs, this does not apply to 1-bit models whose computation is dominated by integer operations.
- Moreover, it measures training computation rather than inference.
- The inference optimal scaling law predicts loss against energy consumption.
- The focus is on inference energy cost since it scales with usage of the model whereas training cost is incurred only once.
- BitNet is found to have higher scaling efficiency.

## Comparison with Post-Training Quantization

- The zero-shot scores of BitNet are comparable to 8-bit models, while the inference cost is much lower.