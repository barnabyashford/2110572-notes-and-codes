Page 1:
Transformer

Page 2:
2
Attention Is All You Need. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

Page 3:
The Transformer 
Only rely on the attention mechanism - No RNN!
-
More parallelizable -> Faster training time
-
Better at longer sequence
Attention Is All You Need. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
3

Page 4:
http://jalammar.github.io/illustrated-transformer/ 
=
4
N=6
encoder
decoder

Page 5:
http://jalammar.github.io/illustrated-transformer/ 
5
In each layer..
Masked

Page 6:
http://jalammar.github.io/illustrated-transformer/ 
Word Embeddings
6

Page 7:
Scaled Dot-Product Attention
There are many variations used in the Transformer:
1.
Self Attention
2.
Encoder-decoder Attention
3.
Masked Self Attention
4.
Multi-headed Self Attention
7

Page 8:
1) Self Attention
Q, K, and V are all from the same input.
8
and dk is a number.
Attention Is All You Need. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long 
Beach, CA, USA.

Page 9:
2) Encoder-Decoder Attention
Encoder-Decoder Attention
(or Cross Attention)
Self Attention
From encoder
From decoder
ALL from same input
9
From decoder
From encoder

Page 10:
3) Masked Self Attention
Prevent the model from seeing into the future to preserve the 
autoregressive property.
Used in decoder only (so the model can’t see the answer).
Masking
10
ALL from same input

Page 11:
Masked Self Attention
Prevent the model from seeing into the future to preserve the 
autoregressive property.
Used in decoder only (so the model can’t see the answer).
11
I like dogs <eos>     <bos> ฉัน ชอบ สุนัข

Page 12:
Masked Self Attention
-inf
-inf
0
1
12

Page 13:
4) Multi-head Self Attention
Multi-head attention allows the model to jointly attend to 
information from different representations.
Attention Is All You Need. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
13

Page 14:
http://jalammar.github.io/illustrated-transformer/ 
14

Page 15:
http://jalammar.github.io/illustrated-transformer/ 
15

Page 16:
http://jalammar.github.io/illustrated-transformer/ 
16

Page 17:
http://jalammar.github.io/illustrated-transformer/ 
17

Page 18:
Positional Encodings
18
The black cat fights the white cat.

Page 19:
Positional Encodings
Without RNN, the model cannot make use of the order of the input sequence, e.g. the first, second, or 
third token.
19
Attention Is All You Need. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
pos is the token position, i is the dimension
The black cat fights the white cat.

Page 20:
20
Attention Is All You Need. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

Page 21:
21
http://jalammar.github.io/illustrated-transformer/ 
i = 4

Page 22:
Types of positional encoding
22
Token 1
Token 2
Token 3
Token 4
Token 5
Token 6
Absolute
1
2
3
4
5
6
Relative
-2
-1
0
1
2
3
1. Absolute Positional encoding
-
1.1) Fixed encoding (Transformer)
-
E.g. sinusoidal forms
-
1.2) Learned encoding (GPT)
2. Relative Positional encoding (GPT NeoX)

Page 23:
Layer Normalization
23

Page 24:
24
http://jalammar.github.io/illustrated-transformer/ 

Page 25:
Batch norm. (each feature (channel)) vs Layer norm. (each word)
25

Page 26:
Done: 
- Let’s link encoder to decoder
- Then, generate final output
26

Page 27:
Final output
27
http://jalammar.github.io/illustrated-transformer/ 

Page 28:
Visualizing Attention (N encoders & N decoders)
https://github.com/jessevig/bertviz 
28

Page 29:
Large Language Models
29

Page 30:
Pretrained Language Models
All Transformer based
1.
Decoder-based model: GPT 
2.
Encoder-based model: BERT
3.
Encoder and Decoder: BART
30

Page 31:
Generative Pre-Training 
(OpenAI GPT) 
31

Page 32:
32
OpenAI GPT (Generative Pre-Training) [Radford, 2018]

Page 33:
Generative Pre-Training (OpenAI GPT) 
A model comprises of Transformer decoders only.
The framework consists of two stages:
1.
Unsupervised pre-training
2.
Supervised finetuning
Excels at text generation tasks
Try it out:
https://transformer.huggingface.co/
33

Page 34:
Generative Pre-Training (OpenAI GPT) 
1) Unsupervised pre-training
Given a large unlabelled corpus                                 , the model’s objective is to maximize the following 
likelihood (also called “Language Modelling”).
where k is the context length and the conditional probability P is modeled using a neural network with 
parameters 𝚯.
34

Page 35:
Generative Pre-Training (OpenAI GPT) 
1) Unsupervised pre-training
35

Page 36:
Generative Pre-Training (OpenAI GPT) 
1) Unsupervised pre-training
36

Page 37:
Generative Pre-Training (OpenAI GPT) 
2) Supervised finetuning
We assume a labeled dataset C, where each instance consists of a sequence of 
input tokens (x1, . . . , xm), along with a label y.
Total finetuning loss
37
classification
LM 
(unsupervised)
L1
L2

Page 38:
Generative Pre-Training (OpenAI GPT) 
2) Supervised finetuning
38

Page 39:
GPT2 & GPT3
Bigger Models + More data = Better Results
39
~100M
~175B
~2B
Model size

Page 40:
OpenAI API
40
https://platform.openai.com/docs/models/gpt-3-5
text
code

Page 41:
41

Page 42:
42
As of 2025-01-17

Page 43:
Bidirectional Encoder Representation 
from Transformers (BERT)
43
https://jalammar.github.io/illustrated-gpt2/ 

Page 44:
44
https://jalammar.github.io/illustrated-gpt2/ 
Year 2018: NLP “Pretrained” Models

Page 45:
BERT [Devlin, et al, 2018]:  
Bidirectional Encoder Representation from Transformers
45
Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, arXiv 2018, https://arxiv.org/pdf/1810.04805.pdf

Page 46:
BERT [Devlin, et al, 2018]:  
Bidirectional Encoder Representation from Transformers
46
Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, arXiv 2018, https://arxiv.org/pdf/1810.04805.pdf
Unsupervised

Page 47:
BERT (cont.): Overall idea
47
Phase1: Unsupervised Learning
-
Mask LM
-
Next sentence prediction
Phase2: Supervised learning
-
E.g. Finetune on QA, Text 
classiﬁcation, etc.

Page 48:
Phase1: Unsupervised Phase
48
Semi-supervised training, using 2 prediction tasks:
1.1) Mask language modeling
○
represents the word using both its left and right context, so called deeply bidirectional.
○
Mask out 15% of the words in the input, run the entire sequence through a deep bidirectional 
encoder, and then predict only the masked words.
1.2) Next sentence prediction (NSP)
○
to learn relationships between sentences.

Page 49:
Phase1: Unsupervised Phase 
(1.1. Masked LM)
49
Mask language modeling:
Source: https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

Page 50:
Phase1: Unsupervised Phase 
(1.2. Next Sentence Prediction)
50
Source:https://arxiv.org/pdf/1810.04805.pdf
CLS = Classiﬁcation, which is used for NSP (Next Sentence Prediction) during this phase
SEP = a special separator token (e.g. separating questions/answers).
NSP

Page 51:
Phase2: Supervised Phase
51
Supervised training (e.g. Email sentence classiﬁcation)
●
you mainly have to train the classiﬁer, with minimal changes happening to the pre-trained 
model during the training phase (ﬁne-tuning approach).

Page 52:
Phase2: Supervised Phase
E.g., Spam Email Classiﬁcation
52
●
BERT is basically a trained 
Transformer Encoder 1 stack.
●
The ﬁrst input token is supplied 
with a special (classiﬁcation 
embedding) [CLS] token for 
classiﬁcation task to represent the 
entire sentence.
●
The output is generated from only 
this special [CLS] token.
1 Vaswani et al., Attention Is All You Need, NIPS 2017, https://arxiv.org/pdf/1706.03762.pd

Page 53:
Phase2: Supervised Phase
E.g., other tasks
53
Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, arXiv 2018, https://arxiv.org/pdf/1810.04805.pdf

Page 54:
Pretrained BERT (English)
54
https://github.com/google-research/bert
New bert model

Page 55:
Pretrained BERT 
(Multilingual)
55
https://github.com/google-research/bert/blob/master/multilingual.md

Page 56:
BERT Family
56
https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/
Based on Aj.Ekapol’s slide

Page 57:
57
https://www.researchgate.net/figure/The-Pre-trained-language-
model-family_fig4_342684048 

Page 58:
Roberta (Robustly optimized BERT approach) 2019
A trick and tuning study
Dynamic masking > static
Next sentence prediction is not optimal
Larger batch + higher learning rate
RoBERTa: A Robustly Optimized BERT Pretraining Approach
Used in WangchanBERTa
Highly recommended to go watch https://www.youtube.com/watch?v=kXPMLX0vfYU 
for more info on WangchanBERTa
https://arxiv.org/abs/1907.11692
Based on Aj.Ekapol’s slide

Page 59:
DistilBERT 2019
Knowledge distillation to get smaller 
models
Reduce the # of transformer layers 
by half. Use tricks in Roberta.
Use KL-divergence between teacher 
and student model
“Cheaper training”
eight 16GB V100 GPUs for approximately 
three and a half days
https://github.com/huggingface/transformers https://medium.com/huggingface/distilbert-8cf3380435b5
Based on Aj.Ekapol’s slide

Page 60:
Millions of parameter
https://medium.com/huggingface/distilbert-8cf3380435b5
Based on Aj.Ekapol’s slide

Page 61:
ALBERT 2019
Want higher hidden units without growing the model. Factorized embedding matrix
Share attention layer parameters across layers. More stable training as a side 
effect.
Some experiments show dropout 
hurt performance
https://arxiv.org/abs/1909.11942
V = Vocab, H = Hidden, E = Lower Dimensional Embedding space
Based on Aj.Ekapol’s slide

Page 62:
BART [2019]
62

Page 63:
BART 2019
A model comprises of both Transformer encoders and 
decoders 
BART optimizes on denoising corrupted inputs
By using both encoders and decoders, BART combines 
the bidirectionality of BERT and autoregressive ability of 
GPT.
BERT
GPT
BART
Input noising scheme
63

Page 64:
Text-to-text Transfer Transformer (T5) 2019
T5 is also an encoder-decoder model
It pretrains on the span-corruption objective.
64

Page 65:
https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/
Based on Aj.Ekapol’s slide

Page 66:
Pre-trained transformer types (by training method)
Encoder only (autoencoder)
-
BERT, ALBERT, RoBERTa
-
Seq classification, token 
classification
-
Masked words and predict
Encoder-decoder (seq2seq)
-
MASS, BART, T5
-
Machine translation, text 
summary
-
Mask phrases
Decoder only (autoregressive)
-
GPT, CTRL
-
Text generation
-
Predict next word
https://huggingface.co/transformers/summary.html   https://arxiv.org/pdf/1905.02450.pdf
Based on Aj.Ekapol’s slide

Page 67:
When to use which?
Text Generation: GPT models
Sequence-to-sequence (e.g. text summarization, translation): BART, T5, or similar
Text classification/ Token classification: RoBERTa
67

Page 68:
Beyond NLP
68

Page 69:
Beyond NLP
1) Computer Vision
ViT uses only encoder.
69
Baselines are 
ResNet
AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

Page 70:
Beyond NLP
2) Text-Image (Multimodal)
Image search, i.e., CLIP
70
https://openai.com/blog/clip/
Both are transformers

Page 71:
Beyond NLP
3) Text-to-Image
Diffusion, e.g., 
-
DALL-E
-
DALL-E3
-
Stable Diffusion
-
Midjourney
71
https://openai.com/blog/dall-e/ 

