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

<img width="740" alt="benchmark" src="https://github.com/LiveWithTrance/DS5690Presentation/assets/111295481/0497392f-bcb7-4063-84be-827c35bbfef4">

In the above, C-Eval, CMMLU and Gaokao are the Chinese benchmarks.

### Training Data
2.6 Trillion token. During data acquisition, our objective is to pursue comprehensive data scalability and representativeness. We gather data from diverse sources including general internet webpages, books, research papers, codebases, and more to build an extensive world knowledge system. 

The composition of the training data:

<img width="415" alt="Training data resource" src="https://github.com/LiveWithTrance/DS5690Presentation/assets/111295481/cbde6016-08ba-488c-b606-20dc7abd17a1">


## Data Processing 

Beichuan2 focuses on data frequency and quality. Data frequency relies on clustering and deduplication. Beichuan2 bulit a large-scale deduplication and clustering system supporting both LSH-like(Locality-sensitive hashing) features and dense embedding features. This system can cluster
and deduplicate trillion-scale data within hours. Based on the clustering, individual documents, paragraphs, and sentences are deduplicated and
scored. Those scores are then used for data sampling in pre-training. 

The size of the training data at different stages of data processing:

<img width="882" alt="截屏2023-11-05 22 29 30" src="https://github.com/LiveWithTrance/DS5690Presentation/assets/111295481/7681e1fb-5868-46a6-a5f9-92d47b843810">


### Question 1: What is the usage of Byte-pair encoding (BPE)?

### Question 2: What is the advantage of Attention with Linear Biases (ALiBi)?


**************************

## Architecture Overview

The model architecture of Baichuan 2 is based on the prevailing Transformer (Vaswani et al., 2017) with the following modifications.

## Tokenizer

A tokenizer needs to balance two critical factors: a high compression rate for efficient inference, and an appropriately sized vocabulary to ensure adequate training of each word embedding.

<img width="466" alt="tokenizer" src="https://github.com/LiveWithTrance/DS5690Presentation/assets/111295481/5f59e771-347f-4b37-9673-c00a13dd0710">


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

Decaying Inter-Token Dependency: It reduces the influence of each token on others with increasing relative distances, which is beneficial for long sequences. This feature helps in reducing computational complexity while still preserving accurate predictions​​.

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


### Root Mean Square Layer Normalization (RMSNorm)

RMSNorm modifies LayerNorm by removing the re-centering operation, aiming to provide a more efficient normalization technique without sacrificing performance. It has been shown to be effective across different tasks and models, and offers a more computationally efficient alternative to LayerNorm.

## Optimization

### AdamW

They use the AdamW optimizer with β1 set to 0.9 and β2 set to 0.95, weight decay of 0.1, and clip the grad norm to 0.5. The models are warmed up with 2,000 linear scaling steps to reach the max learning rate, followed by cosine decay.

### Mixed Precision Training

The models are trained using BFloat16 mixed precision, which has a better dynamic range than Float16, making it more robust for training large language models. Full precision is used for value-sensitive operations such as positional embeddings to avoid issues with low precision.

### NormHead 

They normalize the output embeddings (referred to as 'head') to stabilize training and improve model performance. NormHead helps to stabilize the norm of the embeddings, particularly for rare tokens, and emphasizes the cosine similarity of embeddings over L2 distance, which is beneficial for the linear classifier that computes logits by dot product.

### Max-z Loss

To address the issue of very large logits, which can cause problems during inference, they introduce a max-z loss to normalize the logits. This loss is inspired by NormSoftmax and the auxiliary z-loss from PaLM, and it helps to stabilize training and make inference more robust to hyper-parameter choices.

## Scaling Laws

### Neural Scaling Laws
They observe that the error decreases as a power function of training set size, model size, or both. Before training large models, they train smaller models to fit a scaling law for training larger models.

### Model Size Range
They launched a range of model sizes from 10M to 3B parameters, each trained for up to 1 trillion tokens, using consistent hyper-parameters and the same dataset sourced from Baichuan 2. They use these results to inform the training of larger models.


**************************
## Pseudal-code

### Algorithm: Byte-pair Encoding (BPE)

**Input:** 
- `sequence`: A sequence of characters or bytes.

**Hyperparameters:**
- `vocab_size`: The desired size of the vocabulary.

**Output:** 
- A sequence with common pairs of bytes replaced with a single, unused byte.

**Procedure:**
1. Initialize a vocabulary `V` with individual bytes from `sequence`.
2. While `len(V) < vocab_size`:
   1. Count the frequency of each adjacent byte pair in `sequence`.
   2. Identify the most frequent pair of bytes `pair`.
   3. Replace all occurrences of `pair` with a new, unused byte `Z`.
   4. Add `Z` to the vocabulary `V`.

### Algorithm: Rotary Positional Embedding (RoPE)

**Input:** 
- `sequence_embeddings`: A sequence of token embeddings.

**Parameters:**
- `max_seq_length`: The maximum sequence length.

**Output:** 
- The same sequence with Rotary Positional Embeddings applied.

**Procedure:**
1. For each position `i` in the sequence and each dimension `d` in the embedding:
   1. Compute a rotation matrix `R` based on `i` and `d`.
   2. Apply `R` to the embedding of the token at position `i`.

### Algorithm: Attention with Linear Biases (ALiBi)

**Input:** 
- `Q`: Query matrix.
- `K`: Key matrix.
- `V`: Value matrix.

**Parameters:**
- `m`: A constant bias (one per head).

**Hyperparameters:**
- `d_k`: Dimension of the key vectors.

**Output:** 
- The attention-weighted value matrix.

**Procedure:**
1. Compute the dot product of `Q` and `K^T`.
2. Add the bias `m` to each dot product.
3. Scale the result by `1/sqrt(d_k)`.
4. Apply softmax to obtain attention weights.
5. Multiply the attention weights by `V` to get the final output.


### Algorithm: SwiGLU (Swish-Gated Linear Unit)

**Input:** 
- `X`: An input tensor.

**Parameters:**
- `W1`, `b1`, `W2`, `b2`: Trainable parameters for the linear transformations.

**Output:** 
- An output tensor after the SwiGLU activation.

**Procedure:**
1. Compute the intermediate tensor `A` as `A = X * W1 + b1`.
2. Apply the sigmoid function to get `S = sigmoid(A)`.
3. Compute another intermediate tensor `B` as `B = X * W2 + b2`.
4. Apply the Swish function to `B`: `Swish(B) = B * S`.
5. The output is the element-wise product of `X` and `Swish(B)`.

### Algorithm: Memory Efficient Attention

**Input:** 
- `Q`: Query matrix.
- `K`: Key matrix.
- `V`: Value matrix.
- `attention_mask`: An optional mask to exclude certain positions.

**Output:** 
- The attention-weighted value matrix with reduced memory usage.

**Procedure:**
1. Compute attention scores using a memory-efficient attention mechanism.
2. Apply the `attention_mask` to the scores, if provided.
3. Normalize the scores using softmax.
4. Multiply the normalized scores by `V` to get the final output.

### Algorithm: Root Mean Square Layer Normalization (RMSNorm)

**Input:** 
- `X`: An input tensor.

**Parameters:**
- `gain`: A trainable scale parameter.
- `bias`: A trainable shift parameter.

**Output:** 
- The RMS-normalized tensor.

**Procedure:**
1. Compute the root mean square of `X` for each layer.
2. Divide `X` by the root mean square to normalize.
3. Scale and shift the normalized tensor using `gain` and `bias`.

### Algorithm: AdamW Optimization

**Input:** 
- `parameters`: The model parameters.
- `gradients`: The computed gradients for the parameters.
- `learning_rate`: The learning rate at the current step.

**Hyperparameters:**
- `beta1`: The exponential decay rate for the first moment estimates (typically 0.9).
- `beta2`: The exponential decay rate for the second moment estimates (typically 0.95).
- `weight_decay`: The weight decay coefficient.
- `grad_norm_clip`: The norm to clip gradients to.

**Output:** 
- Updated model parameters.

**Procedure:**
1. Initialize `step` to 0.
2. While training:
   1. Increment `step`.
   2. Compute biased first and second moment estimates of the gradients.
   3. Adjust gradients with weight decay.
   4. Compute bias-corrected first and second moment estimates.
   5. Update parameters with gradients using the AdamW update rule.
   6. Clip gradients to `grad_norm_clip`.
   7. Adjust `learning_rate` using warm-up and cosine decay.

 


**************************

## Critical Analysis

### Limitations and Ethical Considerations

the knowledge of Baichuan 2 models is static and can be outdated or incorrect, posing challenges in fields that requireup-to-date information like medicine or law. While optimized for Chinese and English for safety, themodel has limitations in other languages and maynot fully capture biases relevant to non-Chinese cultures.

There’s also the potential for misuse, as the model could be used to generate harmful or misleading content. Although the team had tried their best
efforts to balance safety and utility, some safety measures may appear as over-cautions, affecting the model’s usability for certain tasks. 

## Answer: What is the usage of Byte-pair encoding (BPE)?

This approach enables LLM models to handle out-of-vocabulary (OOV) words and reduces the overall vocabulary size.

## Answer: Why Use ALiBi?

Traditional position embeddings can sometimes be problematic, especially when dealing with non-linear relationships in language or when handling sequences longer than those seen during training. ALiBi's head-specific constant bias simplifies the model without compromising on performance, making it an attractive choice for NLP tasks.


## Video Overview
https://youtu.be/ZrBtgtWXbb4?si=Wm1Fuy32fOJky3JZ

## Code Demo

- Try Beichuan2: cli_demo.py
  
- Fine tune Beichuan2: fine-tune.py


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









