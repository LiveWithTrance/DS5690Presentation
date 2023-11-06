# Baichuan 2: Open Large-scale Language Models 

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, JunTao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, Zhiying Wu. arXiv. https://doi.org/10.48550/arXiv.2309.10305


## Presenter:Changzhou Li

## Overview

### Introduction

The landscape of Large Language Models (LLMs) in 2023 is characterized by rapid development and the emergence of numerous advanced models. There are new LLM coming out each month and today I will talk about a paper about a new LLM Beichuan 2, a series of large-scale multilingual language models containing 7 billion and 13 billion parameters.

### Problem Addressed in the paper

Most powerful Large language models (LLMs) are closed-source or limited in their capability for languages other than English. Baichuan 2, as a second generation of Beichuan model, aims to work well in multilingual tasks especially in Chinese. Also, in order to benefit the research community in better understand the training dynamics of LLM, the team will release all the pre-training model checkpoints.

### Beichuan2 Performance

#### Academic Benchmark
<img width="740" alt="benchmark" src="https://github.com/LiveWithTrance/DS5690Presentation/assets/111295481/db312d4f-a6d4-409a-a428-06ae830bf59c">

In the above, C-Eval, CMMLU and Gaokao are the Chinese benchmarks.

### Training Data
2.6 Trillion token. During data acquisition, our objective is to pursue comprehensive data scalability and representativeness. We gather data from diverse sources including general internet webpages, books, research papers, codebases, and more to build an extensive world knowledge system. 

The composition of the training data:

<img width="415" alt="Training data resource" src="https://github.com/LiveWithTrance/DS5690Presentation/assets/111295481/58f42a34-df73-489f-96ba-52c11cc6b929">



### Question 1: What is the usage of Byte-pair encoding (BPE)?
### Question 2: What is the advantage of Attention with Linear Biases (ALiBi)?


**************************

## Architecture Overview

The model architecture of Baichuan 2 is based on the prevailing Transformer (Vaswani et al., 2017) with the following modifications.

## Data Processing 

Beichuan2 focuses on data frequency and quality. Data frequency relies on clustering and deduplication. Beichuan2 bulit a large-scale deduplication and clustering system supporting both LSH-like(Locality-sensitive hashing) features and dense embedding features. This system can cluster
and deduplicate trillion-scale data within hours. Based on the clustering, individual documents, paragraphs, and sentences are deduplicated and
scored. Those scores are then used for data sampling in pre-training. 

The size of the training data at different stages of data processing:

<img width="882" alt="截屏2023-11-05 22 29 30" src="https://github.com/LiveWithTrance/DS5690Presentation/assets/111295481/015e2458-292f-4507-907a-a8d6accb9418">


## Tokenizer

A tokenizer needs to balance two critical factors: a high compression rate for efficient inference, and an appropriately sized vocabulary to ensure adequate training of each word embedding.

<img width="415" alt="截屏2023-11-05 15 58 18" src="https://github.com/LiveWithTrance/DS5690Presentation/assets/111295481/93ca01e6-5f9b-47bd-8339-06ed8f8b9e20">

### Byte-pair encoding (BPE)

Byte-Pair Encoding (BPE) is a data compression and subword tokenization algorithm.

BPE iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte. For instance, given the character sequence aaabdaaabac, the sub-sequence aa occurs three times and is the most frequent pair. BPE would replace aa with a new symbol, say Z, resulting in the sequence ZabdZabac​3​. This process continues iteratively, reducing the most common pairs of characters or bytes in the data, which in turn helps in compressing the data.

BPE ensures that the most common words are represented in the vocabulary as a single token while the rare words are broken down into two or more subword tokens and this is in agreement with what a subword-based tokenization algorithm does.

Usage: This approach enables LLM models to handle out-of-vocabulary (OOV) words and reduces the overall vocabulary size.

### Rotary Positional Embedding (RoPE)

RoPE is used for for Baichuan 2-7B.

Rotary Position Embedding (RoPE) is a concept used within transformer architectures to encode the absolute position of tokens in a sequence. Unlike traditional position embeddings that add a separate positional vector to each token, RoPE incorporates positional information directly into the attention mechanism of the transformer model.

Advantages:

Flexibility with Sequence Length: RoPE can handle any sequence length, making it adaptable for NLP models that process texts of varying lengths, unlike traditional position embeddings that are fixed to a specific sequence length​​.

Decaying Inter-Token Dependency: It reduces the influence of each token on others with increasing relative distances, which is beneficial for long sequences. This feature helps in reducing computational complexity while still preserving accurate predictions​1​.

Enhanced Self-Attention: RoPE is capable of equipping linear self-attention with relative position encoding. By considering the relative positions of tokens during self-attention, models can achieve more accurate predictions and a deeper understanding of the relationships between tokens​1​.

### Attention with Linear Biases (ALiBi)


Attention with Linear Biases(ALiBi) is used for Baichuan 2-13B, which is different from most of the open-source models using RoPE.

Attention with Linear Biases is introduced as an alternative to traditional positional encodings in Transformers, aiming to facilitate the handling of sequences at inference time which are longer than the ones encountered during training. Unlike position embeddings, ALiBi adds a constant bias to each attention score, simplifying computations and foregoing the learning of the scalar throughout training.

**Working Mechanism**:

- **Bias Addition**: In the attention sublayer of the Transformer model, when computing attention scores for each head, a constant bias is added to each score. This bias is head-specific and is set to a scalar known as \( m \), which remains constant and is not learned during training.
- **Modified Attention Score Calculation**: The attention score calculation involves the dot product of two vectors, \( q\) and \( k\), followed by the application of the softmax function in traditional attention mechanisms. However, ALiBi modifies this process by adding a bias term to the dot product before the softmax function is applied. The new formula for attention scores in ALiBi is:

$$
\text{Attention}(\mathbf{q}, \mathbf{k}) = \text{softmax}\left(\frac{\mathbf{q}\mathbf{k}^T + m}{\sqrt{d_k}}\right) \mathbf{v}
$$

Where:
- \(q \) and \( k \) are the query and key vectors, respectively.
- \( m \) is the constant bias added to the attention scores.
- \( d \) is the dimension of the key vectors.
- \( v \) is the value vector.

### Advantages of ALiBi
- **Simplifies Calculations**: ALiBi eliminates the need for position embeddings, simplifying the overall computation within the attention mechanism.
- **Maintains Accuracy**: Despite its simplicity, ALiBi has been shown to produce attention scores that are just as accurate as those using traditional position embeddings.
- **Eases Implementation**: Without position embeddings, ALiBi reduces the computational complexity, making it easier to implement and train Transformer models.
- **Handles Longer Sequences**: ALiBi is particularly effective for sequences at inference time that are longer than those encountered during training.


## Activations and Normalizations
### SwiGLU (Swish-Gated Linear Unit)

SwiGLU is designed to leverage the benefits of both the Swish and Gated Linear Unit (GLU) activation functions​​. Activation functions in neural networks, like SwiGLU, are crucial for introducing non-linearity, allowing networks to learn complex relationships between inputs and outputs​1.

The Swish function is defined as Swish(x) = x * sigmoid(beta * x), with 'beta' being a trainable parameter, and it has been shown to perform well in deep networks due to its smoothness​1​. On the other hand, GLU, proposed by Microsoft researchers, is expressed as GLU(x) = x * sigmoid(Wx + b), with 'W' and 'b' also being trainable parameters, and has shown effectiveness in natural language processing tasks.

However, this paper mentions that SwiGLU has a “bilinear” layer and contains three parameter matrices, differing from the vanilla Transformer’s feed-forward layer that has two matrices, so Beichuan2 reduce the hidden sizefrom 4 times the hidden size to 3/8 hidden size and rounded to the multiply of 128.

### Memory efficient attention 
For the attention layer of Baichuan 2, we adopt the memory efficient attention (Rabe andStaats, 2021) implemented by xFormers2
. By leveraging xFormers’ optimized attention with biasing capabilities, we can efficiently incorporate ALiBi’s bias-based positional encoding while
reducing memory overhead. This provides performance and efficiency benefits for Baichuan 2’s large-scale training.


### RMSNorm

## Optimization
### AdamW


**************************
## Pseudal-code

### Algorithm: Byte-Pair Encoding (BPE)
**Input:** 
- `training_text` ∈ \( \mathcal{S} \), a string containing the text used for training.
- `num_merges` ∈ \( \mathbb{N} \), the number of merge operations to perform.

**Output:** 
- `vocabulary` ∈ \( \mathcal{D} \), a dictionary representing the subword vocabulary with subwords as keys and their frequencies as values.

1. **Initialize Vocabulary**:
    - `vocabulary = get_unique_characters(training_text)`
2. **Perform Merge Operations**:
    - For `i` in `range(num_merges)`:
        a. `pair_frequencies = calculate_pair_frequencies(training_text)`
        b. `most_frequent_pair = find_most_frequent_pair(pair_frequencies)`
        c. `new_subword = merge_pair(most_frequent_pair)`
        d. `update_vocabulary(vocabulary, new_subword)`
        e. `training_text = replace_pair(training_text, most_frequent_pair, new_subword)`

### Algorithm: BPE Tokenization
**Input:** 
- `text` ∈ \( \mathcal{S} \), a string containing the text to be tokenized.
- `vocabulary` ∈ \( \mathcal{D} \), the subword vocabulary obtained from the training phase.

**Output:** 
- `tokens` ∈ \( \mathcal{L} \), a list of tokens representing the tokenized text.

1. **Initialize Token List**:
    - `tokens = []`
2. **Tokenize Text**:
    - While `text` is not empty:
        a. `longest_subword = find_longest_matching_subword(text, vocabulary)`
        b. `tokens.append(longest_subword)`
        c. `text = remove_subword(text, longest_subword)`

### Algorithm: BPE Detokenization
**Input:** 
- `tokens` ∈ \( \mathcal{L} \), a list of tokens representing the tokenized text.

**Output:** 
- `detokenized_text` ∈ \( \mathcal{S} \), a string representing the detokenized text.

1. **Concatenate Tokens**:
    - `detokenized_text = concatenate_tokens(tokens)`


### Algorithm: Rotary Positional Embedding (RoPE)
**Input:**
- \( x_q, x_k \) ∈ \( \mathbb{R}^{d} \), query and key tensors.
  
**Output:**
- \( x_q', x_k' \) ∈ \( \mathbb{R}^{d} \), tensors with rotary embeddings applied.

**Parameters:**
- \( \theta \) ∈ \( \mathbb{R} \), a non-zero constant for rotation matrix computation.
- \( m, n \) ∈ \( \mathbb{N} \), absolute positions of tokens.

1. **Define Rotation Matrix Function:**
   - \( \mathbf{R}_{\theta,t} = \begin{pmatrix} \cos t\theta & -\sin t\theta \\ \sin t\theta & \cos t\theta \end{pmatrix} \)

2. **Compute Rotation Matrices for Queries and Keys:**
   - \( \mathbf{R}_{\theta,m} = \mathbf{R}_{\theta,t=m} \)
   - \( \mathbf{R}_{\theta,n} = \mathbf{R}_{\theta,t=n} \)

3. **Apply Rotary Embeddings to Queries and Keys:**
   - \( f_q(\mathbf{x}_m, m) = \mathbf{R}_{\theta,m} \mathbf{W}_q \mathbf{x}_m = \mathbf{q}_m \)
   - \( f_k(\mathbf{x}_n, n) = \mathbf{R}_{\theta,n} \mathbf{W}_k \mathbf{x}_n = \mathbf{k}_n \)

4. **Compute Relative Rotation Matrix:**
   - \( \mathbf{R}_{\theta,n-m} = \mathbf{R}_{\theta,m}^\mathsf{T} \mathbf{R}_{\theta,n} \)

5. **Compute Inner Product:**
   - \( g(\mathbf{x}_m, \mathbf{x}_n, n-m) = \mathbf{x}_m^\mathsf{T} \mathbf{W}_q^\mathsf{T} \mathbf{R}_{\theta,n-m} \mathbf{W}_k \mathbf{x}_n = \mathbf{q}_m^\mathsf{T} \mathbf{k}_n \)

6. **Output Rotated Queries and Keys:**
   - \( x_q' = f_q(\mathbf{x}_m, m) \)
   - \( x_k' = f_k(\mathbf{x}_n, n) \)

7. **Return:**
   - Return \( x_q', x_k' \)
 

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

Like other large language models, Baichuan 2 also faces ethical challenges. It’s prone to biases andtoxicity, especially given that much of its training data originates from the internet. Despite our best efforts to mitigate these issues using benchmarks
like Toxigen (Hartvigsen et al., 2022), the risks cannot be eliminated, and toxicity tends to increase
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

## Answer: What is the usage of Byte-pair encoding (BPE)?
## Answer: Why Use ALiBi?
Traditional position embeddings can sometimes be problematic, especially when dealing with non-linear relationships in language or when handling sequences longer than those seen during training. ALiBi's head-specific constant bias simplifies the model without compromising on performance, making it an attractive choice for NLP tasks.


## Video Overview
https://youtu.be/ZrBtgtWXbb4?si=Wm1Fuy32fOJky3JZ

## Code Demo
- Try Beichuan2: 
- Fine tune Beichuan2: 

## References

- Baichuan 2: Open Large-scale Language Models: https://arxiv.org/pdf/2309.10305
- Baichuan2-7B-Intermediate-Checkpoints: https://huggingface.co/baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints
- Byte-Pair Encoding Paper: https://arxiv.org/pdf/2310.06825
- Sentence Piece: https://arxiv.org/pdf/1808.06226
- Rotary Positional Embedding (RoPE): https://arxiv.org/abs/2104.09864
- Attention with Linear Biases (ALiBi): https://arxiv.org/pdf/2108.12409
- X-formers: https://github.com/facebookresearch/xformers
- SwiGLU Activation Function: https://paperswithcode.com/method/swiglu

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









