# The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

[Paper](https://arxiv.org/pdf/2402.17764.pdf)

BitNet b1.58 is a 1-bit LLM variant in which every single parameter is ternary $\{-1, 0, 1\}$.
It matches the full-precision Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance,
while being significantly more cost-effective in terms of latency, memory, throughput and energy consumption.

## 1-bit LLMs

- The increasing size of LLMs has posed challenges for deployment and has raised concerns about environmental and economic impact due to high energy consumption.
- One approach is to use post-training quantization, however this is sub-optimal.
- The major computation cost comes from floating-point addition and multiplication operations.
    - BitNet is a 1-bit model architecture that only involves integer addition during matrix multiplication, which saves orders of energy costs for LLMs.
- In addition to computation, the process of transferring model parameters from DRAM to the memory of an on-chip accelerator can be expensive during inference.
    - 1-bit LLMs have much lower memory footprint, reducing the cost and time of loading weights, leading to faster inference.
- BitNet b1.58 is a 1-bit LLM variant in which every single parameter is ternary $\{-1, 0, 1\}$.
- BitNet b1.58 has all the benefits of BitNet with additional advantages:
    - It has stronger modeling capability due to its explicit support for feature filtering made possible by the inclusion of 0 in the model weights.
    - BitNet b1.58 can match full-precision baselines in terms of both perplexity and end-task performance, starting from a 3B size, when using the same configuration.

## BitNet b1.58

![Bitnet b1.58](https://github.com/glassbubbles678/blog/blob/main/images/bitnet-b1.58.png)

### Quantization Function

**Quantization of weights**

- The absmean quantization function is used to constrain the weights to $\{-1, 0, 1\}$.
- It first scales the weight matrix by its average absolute value and then rounds each value to the nearest integer among $\{-1, 0, 1\}$.

$$\widetilde{W} = \text{RoundClip}(\frac{W}{\gamma + \epsilon}, -1, 1)$$

$$\text{RoundClip}(x,a,b) = \max(a,\min(b,\text{round}(x)))$$

$$\gamma = \frac{1}{nm} \sum_{ij} \mid W_{ij} \mid$$

**Quantization of activations**

- The quantization function for activations follows the same implementation as BitNet.
- The only difference is that the activations are not scaled to $[0, Q_b]$ before the non-linear functions.
- Instead, all activations are scaled to $[-Q_b, Q_b]$ to get rid of the zero-point quantization.

### LLaMa Components

- BitNet b1.58 adopts LLaMa-alike components including RMSNorm, SwiGLU and rotary embeddings.

## Results

- 3.9B BitNet b1.58 has better perplexity than LLaMa LLM 3B while being 2.4 times faster and using 3.32 times less GPU memory.
- 3.9B BitNet b1.58 outperforms LLaMa LLM 3B on end-tasks with lower memory and latency cost. 
- This indicates BitNet b1.58 is a Pareto improvement over state-of-the-art LLM models.

### Memory and Latency

- In terms of memory and latency, the speed-up increases as the model size scales.
- This is because the time cost for `nn.Linear` grows with the model size.
- The embedding remains full-precision and its memory proportion is smaller for larger models.

### Energy

- The majority of BitNet b1.58 is INT8 addition whereas LLaMa LLM consists of FP16 addition and multiplication.
- Consequently, BitNet b1.58 saves 71.4 times arithmetic operations energy consumption for matrix multiplication on 7nm chips.
- As the model size scales, BitNet b1.58 becomes increasingly more efficient in terms of energy consumption compared to the FP16 LLaMa LLM baseline.
- This is because the percentage `nn.Linear` grows with model size, while the cost from other components is smaller for larger models.

### Throughput

- BitNet b1.58 can support upto 11 times the batch size of LLaMa LLM, resulting in an 8.9 times higher throughput.

### Scaling Law

- BitNet b1.58 is enabling a new scaling law with respect to model performance and inference cost.
- In terms of latency, memory usage and energy consumption:
    - 13 B BitNet b1.58 is more efficient than 3B FP16 LLM
    - 30 B BitNet b1.58 is more efficient than 7B FP16 LLM
    - 70 B BitNet b1.58 is more efficient than 13B FP16 LLM
    
### Training with 2T tokens

- To test the scalability of BitNet b1.58 in terms of number of tokens, a BitNet b1.58 model was trained with 2T tokens.
- It achieved superior performance on end tasks compared to StableLM 3B, indicating strong generalization capabilities.

## Discussion and Future Work

### 1-bit Mixture-of-Expert (MoE) LLMs

- Mixture-of-Experts is a cost-effective approach for LLMs.
- However, while it significantly reduces the FLOPs, it has high memory consumption and inter-chip communication overhead.
- 1.58b LLMs reduce the memory footprint which reduces the number of devices required to deploy MoE models.
- It significantly reduces the overhead of transferring activations across networks.
- Ultimately, there would be no overhead if models could be placed on a single chip.

### Native Support of Long Sequence in LLMs

- Handling long sequences is difficult due to the memory consumption introduced by KV caches.
- BitNet b1.58 reduces activations from 16 to 8 bits, allowing context length to be doubled given the same resources. This is a significant step towards native support for long sequences.

### LLMs on Edge and Mobile

- Edge and mobile devices are limited by their memory and computational power which can restrict the performance and scale of LLMs.
- The reduced memory and energy consumption of 1.58 bit LLMs allows them to be deployed on these devices.
- Moreover, 1.58 bit LLMs are more friendly to CPU devices.

### New Hardware for 1-bit LLMs

- New hardware and systems specifically optimized for 1-bit LLMs can be designed.
