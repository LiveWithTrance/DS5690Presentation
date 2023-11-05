# Baichuan 2: Open Large-scale Language Models 

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, JunTao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, Zhiying Wu. arXiv. https://doi.org/10.48550/arXiv.2309.10305


## Presenter:Changzhou Li
- 

## Overview
Large language models (LLMs) have demonstrated remarkable performance on a variety of natural language tasks based on just a few examples of natural language instructions, reducing the need for extensive feature engineering. However, most powerful LLMs are closed-source or limited in their capability for languages other than English. In this technical report, we present Baichuan 2, a series of large-scale multilingual language models containing 7 billion and 13 billion parameters, trained from scratch, on 2.6 trillion tokens. Baichuan 2 matches or outperforms other open-source models of similar size on public benchmarks like MMLU, CMMLU, GSM8K, and HumanEval. Furthermore, Baichuan 2 excels in vertical domains such as medicine and law. We will release all pre-training model checkpoints to benefit the research community in better understanding the training dynamics of Baichuan 2.
### Context
In recent years, the field of natural language processing has been revolutionized by the advent of large language models (LLMs). These models have shown exceptional ability in understanding and generating human-like text, leading to significant improvements in a variety of language tasks. However, the majority of these advanced models are proprietary, closed-source, and often exhibit limitations when dealing with languages other than English. 
### Problem Addressed
The paper introduces "Baichuan 2," which tackles the problem of accessibility and language inclusivity in large-scale language models. The authors aim to provide an open-source alternative to the existing LLMs that not only performs competitively on standard benchmarks but also offers extensive multilingual support and excels in specialized domains such as medicine and law.

### Approach:
The authors' approach involves training Baichuan 2 from scratch, using a massive dataset of 2.6 trillion tokens. The model comes in two sizes, with 7 billion and 13 billion parameters, positioning it among the larger models available in the open-source domain. The training process is designed to be transparent, with checkpoints to be released publicly, allowing the research community to study and understand the model's training dynamics.
### Training Data
2.6 Trillion tokens. Data sourcing: During data acquisition, our
objective is to pursue comprehensive data
scalability and representativeness. We gather data
from diverse sources including general internet
webpages, books, research papers, codebases,
and more to build an extensive world knowledge
system. 

The composition of the training corpus is
shown in Figure 1.


### Beichuan2 Performance

#### Academic Benchmarks


**************************

## Architecture Overview

The model architecture of Baichuan 2 is based on the prevailing Transformer (Vaswani et al., 2017). with following modifications.

### Tokenizer

### Positional Embeddings
#### Rotary Embedding (RoPE)

One of the fundamental advancements in LLaMA2 is the adoption of Rotary Position Embedding (RoPE) in place of traditional absolute positional encoding. What sets RoPE apart is its ability to seamlessly integrate explicit relative position dependencies into the self-attention mechanism of the model. This dynamic approach offers several key advantages:
- Flexibility in Sequence Length: Traditional position embeddings often require defining a maximum sequence length, limiting their adaptability. RoPE, on the other hand, is incredibly flexible. It can generate position embeddings on-the-fly for sequences of any length.
- Decaying Inter-Token Dependency: RoPE is smart about modeling the relationship between tokens. As tokens become more distant from each other in a sequence, RoPE naturally reduces their inter-token dependencies. This gradual decay aligns more closely with how humans understand language, where the importance of earlier words tends to diminish.
- Enhanced Self-Attention: RoPE equips the linear self-attention mechanisms with relative position encoding, a feature not present in traditional absolute positional encoding. This enhancement allows for more precise utilization of token embeddings.

#### ALiBi

#### RMSNorm (Root Mean Square Layer Normalization)

Beichuan2 adopts Root Mean Square Layer Normalization (RMSNorm), to enhance the transformer architecture by replacing the existing Layer Normalization (LayerNorm). LayerNorm has been beneficial for improving training stability and model convergence, as it re-centers and re-scales input and weight matrix values. However, this improvement comes at the cost of computational overhead, which slows down the network.

$$ \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2} $$

$$ y = \frac{x}{\text{RMS}(x)} \times \gamma + \beta $$

- γ is the scale parameter.
- β is the shift parameter.

RMSNorm, on the other hand, retains the re-scaling invariance property while simplifying the computation. It regulates the combined inputs to a neuron using the root mean square (RMS), providing implicit learning rate adaptation. This makes RMSNorm computationally more efficient than LayerNorm.

$$ \mu = \frac{1}{N} \sum_{i=1}^{N} x_i $$

$$ \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 $$

$$ y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta $$

- x is the input vector.
- N is the dimensionality of x.
- μ is the mean of the input.
- $σ^2$ is the variance of the input.
- ϵ is a small constant added for numerical stability.
- γ is the scale parameter.
- β is the shift parameter.

Extensive experiments across various tasks and network architectures show that RMSNorm performs as effectively as LayerNorm while reducing computation time by 7% to 64%.

This custom script first standardizes the input x, by dividing it by its root mean square, thereby making it invariant to scaling changes. The learned weight parameter self.weight is applied to each element in the standardized tensor. This operation adjusts the magnitude of the values based on the learned scaling factor.

#### KV (Key-Value) Caching

Key-Value (KV) caching is a technique used to accelerate the inference process in machine learning models, particularly in autoregressive models like GPT and Llama. In these models, generating tokens one by one is a common practice, but it can be computationally expensive because it repeats certain calculations at each step. To address this, KV caching comes into play. It involves caching the previous Keys and Values, so we don’t need to recalculate them for each new token. This significantly reduces the size of matrices used in calculations, making matrix multiplications faster. The only trade-off is that KV caching requires more GPU memory (or CPU memory if a GPU isn’t used) to store these Key and Value states.


#### SwiGLU (Swiss Function + Gated Linear Unit)

SwiGLU, as utilized in LLaMA2 models, is an activation function designed to enhance the performance of the position-wise feed-forward network (FFN) layers in the Transformer architecture.

The definition of SwiGLU is given by the following mathematical expression:

$$ \text{SwiGLU}\left(x, W, V, b, c, \beta\right) = \text{Swish}\_{\beta}\left(xW + b\right) \otimes \left(xV + c\right) $$

Here, x is the input to the neuron, W and V are weight matrices, b and c are bias vectors, and β is a constant. The ⊗ symbol denotes element-wise multiplication, while the Swish function is defined as:

$$ \text{Swish}\_{\beta}\left(x\right) = x \cdot \sigma\left(\beta x\right) $$

where σ is the sigmoid function. The purpose of the Swish function is to introduce non-linearity into the activation function while still allowing for efficient computation.

During the forward pass, the input tensor x is subjected to multi layer of linear transformations. The SwiGLU activation function, applied after first transformation, enhances the expressive power of the model. The final transformation maps the tensor back to its original dimensions. This unique combination of SwiGLU activation and multiple FeedForward layer enhances the performance of the model.

#### memory efficient attention 
For the attention layer of Baichuan 2, we
adopt the memory efficient attention (Rabe and
Staats, 2021) implemented by xFormers2
. By
leveraging xFormers’ optimized attention with
biasing capabilities, we can efficiently incorporate
ALiBi’s bias-based positional encoding while
reducing memory overhead. This provides
performance and efficiency benefits for Baichuan
2’s large-scale training




**************************
## Pseudal-code

### Algorithm: Token Embedding

**Input:** $\( v \in V \cong [N_v] \)$, a token ID.  
**Output:** $\( e \in \mathbb{R}^{d_e} \)$, the vector representation of the token.  
**Parameters:** $\( W_e \in \mathbb{R}^{d_e \times N_v} \)$, the token embedding matrix.

1. return $e = W_e[:, v]$

### Algorithm: Rotary Positional Embedding
**Input:** $\( x_q, x_k \) ∈ \( \mathbb{R}^{d} \)$, query and key tensors.  
**Output:** $\( x_q', x_k' \) ∈ \( \mathbb{R}^{d} \)$, tensors with rotary embeddings applied.   
**Parameters:**    
  dim $∈ \( \mathbb{N} \)$, dimension of the frequency tensor.   
  end $∈ \( \mathbb{N} \)$, end index for precomputing frequencies.   
  $\theta$ $∈ \( \mathbb{R} \)$, scaling factor for frequency computation, default to 10000.0.   
  $freqs_{cis}$ $∈ \( \mathbb{C}^{dim \times end} \)$, precomputed frequency tensor with complex exponentials.   

1. Compute frequency scale: $\( \text{freqs} = \frac{1.0}{(\theta ^ {(\text{range}(0, \text{dim}, 2)[: (\text{dim} // 2)] / \text{dim})}) }\)$   
2. Initialize time indices: $\( t = range(\text{end}) \)$   
3. Compute outer product of t and freqs: $freqs_{mat}$ = $\text{outer}(t, \text{freqs}) \)$      
4. Convert $freqs_{mat}$ to polar coordinates: $freqs_{cis}$ = polar(ones_like($freqs_{mat}$), $freqs_{mat}$)     
5. Convert $\( x_q \)$ and $\( x_k \)$ into complex matrices   
6. Reshape $freqs_{cis}$ for broadcasting compatibility with $\( x_q \)$ and $\( x_k \)$   
7. Apply rotary embeddings to $\( x_q \)$ and $\( x_k \)$ using complex multiplication   
8. Convert the results back to real values: $\( x_q' \), \( x_k' \)$   
9. Return $\( x_q' \)$ and $\( x_k' \)$   

### Algorithm: Basic Single-query Attention
**Input:**  
$\( e \in \mathbb{R}^{d_{in}} \)$ - vector representation of the current token. <br>
$\( e_t \in \mathbb{R}^{d_{in}} \)$ - vector representations of context tokens $\( t \in [T] \).$ <br>

**Output:**  
$\( \mathbf{v} \in \mathbb{R}^{d_{out}} \)$ - vector representation of the token and context combined. <br>

**Parameters:**  
$\( W_q, W_k \in \mathbb{R}^{d_{attn} \times d_{in}} \)$, $\( b_q, b_k \in \mathbb{R}^{d_{attn}} \)$ - the query and key linear projections. <br>
$\( W_v \in \mathbb{R}^{d_{out} \times d_{in}} \)$, $\( b_v \in \mathbb{R}^{d_{out}} \)$ - the value linear projection. <br>

1.  $q \leftarrow W_q e + b_q$ <br>
2.  $k_t \leftarrow W_k e_t + b_k$ <br>
3.  $v_t \leftarrow W_v e_t + b_v$ <br>
4.  $\alpha_t \leftarrow \frac{\exp(q^T k_t / \sqrt{d_{attn}})}{\sum_u \exp(q^T k_u / \sqrt{d_{attn}})}$ <br>
5.  $\mathbf{\tilde{v}} = \sum_{t=1} \alpha_t \times v_t $

### Algorithm: Attention
**Input:**  
$\( X \in \mathbb{R}^{d_x \times \ell_x} \)$, vector representations of primary sequence. <br>
$\( Z \in \mathbb{R}^{d_z \times \ell_z} \)$, vector representations of context sequence. <br>

**Output:**  
$\( \tilde{v} \in \mathbb{R}^{d_{out} \times \ell_x} \)$, updated representations of tokens in X, folding in information from tokens in Z. <br>

**Parameters:** $\( W_{qkv} \)$ consisting of: <br>
$\( W_q \in \mathbb{R}^{d_{attn} \times d_x} \)$, $\( b_q \in \mathbb{R}^{d_{attn}} \)$ <br>
$\( W_k \in \mathbb{R}^{d_{attn} \times d_z} \)$, $\( b_k \in \mathbb{R}^{d_{attn}} \)$ <br>
$\( W_v \in \mathbb{R}^{d_{out} \times d_z} \)$, $\( b_v \in \mathbb{R}^{d_{out}} \)$ <br>

**Hyperparameters:**
𝐻, number of attention heads <br>
$\( \text{Mask} \in \{0,1\}^{\ell_z \times \ell_x} \)$ <br>

1. $\( q \leftarrow W_q X + b_q^T \)$  [[Query $\( \in \mathbb{R}^{d_{attn} \times \ell_x} \)]]$ <br>
2. $\( k \leftarrow W_k Z + b_k^T \)$  [[Key $\( \in \mathbb{R}^{d_{attn} \times \ell_z} \)]]$ <br>
3. $\( v \leftarrow W_v Z + b_v^T \)$  [[Value $\( \in \mathbb{R}^{d_{out} \times \ell_z} \)]]$ <br>
4. $\( S \leftarrow KTQ \)$  [[Score $\( \in \mathbb{R}^{\ell_z \times \ell_x} \)]]$ <br>
5. For each $\( t_z, t_x \)$, if $\( \text{Mask}[t_z, t_x] \)$, then $\( S[t_z, t_x] \leftarrow -\infty \)$ <br>
6. $\( \tilde{v} = V \cdot \text{softmax}(S/\sqrt{d_{attn}}) \)$ <br>

### Algorithm: Multi-head Attention
**Input:**  
$\( X \in \mathbb{R}^{d_x \times \ell_x} \)$, vector representations of primary sequence. <br>
$\( Z \in \mathbb{R}^{d_z \times \ell_z} \)$, vector representations of context sequence. <br>

**Output:**  
$\( \tilde{V} \in \mathbb{R}^{d_{out} \times \ell_x} \)$, updated representations of tokens in X, folding in information from tokens in Z. <br>

**Hyperparameters:** 
H, number of attention heads <br>
$\( \text{Mask} \in \{0,1\}^{\ell_z \times \ell_x} \)$ <br>

**Parameters:** $W$ consisting of: <br>

For $\( h \in [H] \)$, $\( W^h_{qkv} \)$ consisting of: <br>
$\( W^h_q \in \mathbb{R}^{d_{attn} \times d_x} \)$, $\( b^h_q \in \mathbb{R}^{d_{attn}} \)$ <br>
$\( W^h_k \in \mathbb{R}^{d_{attn} \times d_z} \)$, $\( b^h_k \in \mathbb{R}^{d_{attn}} \)$ <br>
$\( W^h_v \in \mathbb{R}^{d_{mid} \times d_z} \)$, $\( b^h_v \in \mathbb{R}^{d_{mid}} \)$ <br>
$\( W_o \in \mathbb{R}^{d_{out} \times H \times d_{mid}} \)$, $\( b_o \in \mathbb{R}^{d_{out}} \)$ <br>

1. For $\( h \in [H] \)$: <br>
2. $\( y^h \leftarrow \text{Attention}(X, Z|W^h_{qkv}, \text{Mask}) \)$ <br>
3. $\( Y \leftarrow [y^1, y^2, ..., y^H] \)$ <br>
4. Return $\( \tilde{V} = W_o Y + b_o^T \)$

### Algorithm: Grouped-Query Attention
$\tilde{V}$ ← $GroupedQueryAttention(X, Z|W, Mask)$  

**Input:** $X ∈ R^{d_{\text{x}} \times l_{\text{x}}}$, $Z ∈ R^{d_{\text{z}}×l_{\text{z}}}$, vector representations of primary and context sequence.   
**Output:** $\tilde{V} ∈ R^{d_{\text{out}}×l_{\text{x}}}$, updated representations of tokens in X, folding in information from tokens in Z.   

**Hyperparameters:** 
H, number of local attention heads   
$N_kv$, number of key-value pairs   
RepetitionFactor, repetitions for local heads ($N_rep$)   
$\( \text{Mask} \in \{0,1\}^{\ell_z \times \ell_x} \)$, attention mask   

**Parameters:** W consisting of:  
For $h ∈ [H], W^h$ consisting of:    
    $W^h_q ∈ R^{d_{att}×d_x}$,   
    $W^h_k ∈ R^{d_{att}×d_z}$,  
    $W^h_v ∈ R^{d_{att}×d_z}$   
$Wo ∈ R^{d_{out}×H×d_{att}}$, Wo is the output linear transformation.

1. For $h ∈ [H]$:  
2. $Xq_h$ ← $LinearTransformation(X, W^h_q)$  
3. $Xk_h$ ← $LinearTransformation(Z, W^h_k)$  
4. $Xv_h$ ← $LinearTransformation(Z, W^h_v)$  
5. Cache keys and values: $cache_k$, $cache_v$ based on current position  
6. If $N_kv < H$:  
7. Repeat keys and values for local attention using RepetitionFactor  
8. Compute scores: $S_h$ ← $Xq_h • Xk_h^T / sqrt(datt)$  
9. Apply mask to scores: $S_h$ ← $S_h + Mask$  
10. Normalize scores: $S_h$ ← $Softmax(S_h)$  
11. Compute context: $C_h$ ← $S_h • Xv_h$  
12. $Y ← [C_1, C_2, ..., C_H]$  
13. return $\tilde{V} = WoY$  

### Algorithm: RMS Layer Normalization

**Input:** $x ∈ ℝ^d$, neural network activations.   
**Output:** $y ∈ ℝ^d$, normalized activations.   
**Parameters:** $γ, β ∈ ℝ^d$, element-wise scale and offset.   

1. $μ ← Σ_{i=1}^d x[i]/d$
2. $σ^2 ← Σ_{i=1}^d (x[i] - μ)^2/d$
3. $RMS ← sqrt(Σ_{i=1}^d x[i]^2/d)$
4. $y ← (x/RMS) * γ + β$
5. return $y$


### Algorithm: Unembedding
**Input:** $\( e \in \mathbb{R}^{d_e} \)$: a token encoding.   
**Output:** $\( p \in \Delta(V) \)$: a probability distribution over the vocabulary.   
**Parameters:** $\( W_u \in \mathbb{R}^{N \times d_e} \)$: the unembedding matrix.   

1. return p = $softmax(W_u e)$

**************************
### Algorithm: DTransformer

**Input:** `x`, a sequence of token IDs.

**Output:** $P \in (0,1)^{N \times \text{length}(x)}$, where the t-th column of `P` represents $\hat{P̂_θ}(x[t+1]|x[1:t])$.

**Hyperparameters:** $ℓ_{\text{max}}, L, H, d_e, d_{mlp} \in \mathbb{N}$

**Parameters:**
- $W_e \in \mathbb{R}^{d_e \times N}$, $W_p \in \mathbb{R}^{d_e \times ℓ_{\text{max}}}$: the token and rotary positional embedding matrices.
- For each layer `l`:
  - $W_l$, Group Query Attention parameters for layer `l`.
  - $\gamma^1, \beta^1, \gamma^2, \beta^2$: sets of RMS layer-norm parameters.
  - $w^l_{mlp1}, b^l_{mlp1}, w^l_{mlp2}, b^l_{mlp2}$: MLP parameters.
- $\gamma, \beta$: final RMSlayer-norm parameters.
- $W_u \in \mathbb{R}^{N \times d_e}$: the unembedding matrix.

**Algorithm:**
1. $ℓ \leftarrow \text{length}(x)$
2. For each `t` in `ℓ`: $e_t \leftarrow W_e \cdot x[t] + W_p[:,t]$
3. $X \leftarrow [e_1, e_2, ... e_ℓ]$
4. For each `l` from 1 to `L`:
   - For each `t` in `ℓ`:
     - $X{[:,t]} \leftarrow {RMSLayerNorm}(\tilde{X}{[:,t]} | \gamma_l{1}, \beta_l{1})$
     - $X \leftarrow X + \text{GroupedQueryAttention}(X, W_l, \text{Mask}[t, :] = [t \leq t'])$$
     - $X{[:,t]} \leftarrow {RMSLayerNorm}(\tilde{X}{[:,t]} | \gamma_l{2}, \beta_l{2})$
     - $X \leftarrow X + w^l_{mlp2} \cdot \text{SwiGLU}(w^l_{mlp1} \tilde{X} + b^l_{mlp1}1^T) + b^l_{mlp2}1^T$
5. For each `t` in `ℓ`: $X[:,t] \leftarrow {RMSLayerNorm}(X[:,t], \gamma, \beta)$
6. Return $P = \text{softmax}(W_u X)$

**************************

## Critical Analysis
### Limitations and Ethical Considerations

Like other large language models, Baichuan 2 also
faces ethical challenges. It’s prone to biases and
toxicity, especially given that much of its training
data originates from the internet. Despite our best
efforts to mitigate these issues using benchmarks
like Toxigen (Hartvigsen et al., 2022), the risks
cannot be eliminated, and toxicity tends to increase
with model size. Moreover, the knowledge of
Baichuan 2 models is static and can be outdated or
incorrect, posing challenges in fields that require
up-to-date information like medicine or law. While
optimized for Chinese and English for safety, the
model has limitations in other languages and may
not fully capture biases relevant to non-Chinese
cultures.
There’s also the potential for misuse, as the
model could be used to generate harmful or
misleading content. Although we try our best
efforts to balance safety and utility, some safety
measures may appear as over-cautions, affecting
the model’s usability for certain tasks. We
encourage users to make responsible and ethical
use of Baichuan 2 models. Meanwhile, we will
continue to optimize these issues and release
updated versions in the future.



## References

- Baichuan 2: Open Large-scale Language Models: https://arxiv.org/pdf/2309.10305
- Baichuan2-7B-Intermediate-Checkpoints: https://huggingface.co/baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints
- Mistral-7b Paper: https://arxiv.org/pdf/2310.06825.pdf
- RMS Normalization: https://arxiv.org/abs/1910.07467  
- Rotary Positional Embedding (RoPE): https://arxiv.org/abs/2104.09864  
- SwiGLU Activation Function: https://paperswithcode.com/method/swiglu
- Group Query Attention Paper: https://arxiv.org/pdf/2305.13245.pdf
- MMLU Dataset: https://paperswithcode.com/dataset/mmlu  
- MMLU Paper: https://arxiv.org/abs/2009.03300
- CMMLU Dataset: https://github.com/haonan-li/CMMLU/tree/master/data
- CMMLU Paper: https://arxiv.org/abs/2306.09212
- C-Eval Dadaset:https://cevalbenchmark.com
- C-Eval Paper: https://arxiv.org/abs/2305.08322
- Formal Algorithm of Transformers Paper: https://arxiv.org/pdf/2207.09238
- RLHF Explained: https://huyenchip.com/2023/05/02/rlhf.html  
- Transformers Explained: https://deepgram.com/learn/visualizing-and-explaining-transformer-models-from-the-ground-up  
- Activation Functions Explained: https://www.geeksforgeeks.org/activation-functions-neural-networks/  

## Video Overview

## Code Demo
- Annotated Beichuan2 source code: 
- Try Beichuan2: 
- Fine tune Beichuan2: 






