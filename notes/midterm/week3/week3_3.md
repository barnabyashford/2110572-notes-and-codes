Page 1:
The Transformer family

Page 2:
https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/

Page 3:
Pretrained models
●With word2vec (2013), you can pretrained models based on large data
●The type of pretraining gets more complicated overtime
■
Word level - Glove (2014)
■
Sentence level - ULMFit (2018)
Predicting words (Language modeling)
This means we can just use sentences 
with no supervised labels.
How to pre-train?

Page 4:
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
GPT, Llama
-
Text generation
-
Predict next word
https://huggingface.co/transformers/summary.html
https://arxiv.org/pdf/1905.02450.pdf

Page 5:
BERT - Bidirectional Encoder Representations from Transformers
https://arxiv.org/abs/1810.04805

Page 6:
BERT Pretraining
●Full transformer built on wordpiece tokens.
●Masked LM to learn bi-directional LM (has access to future words)
●Next sentence prediction to learn discourse

Page 7:
Masking
●
15% of the token in the training data is selected.
○
80% becomes [MASK] token
○
10% becomes a random token
○
10% left as is
This is a MASK that I draw with.
Prevents training/inferencing mismatch. 
(No [MASK] in regular sentences).

Page 8:
Next Sentence Prediction
●
To help the model learn discourse
●
Predict whether
○
two sentences from a paragraph
○
two sentences randomly selected
●
The [CLS] token is the position 
responsible for this prediction.
●
[CLS] token is often used as a summary
embedding for the sentence.

Page 9:
BERT use cases
●
Being an encoder BERT does classification tasks very well
○
Sequence prediction tasks: sentiment analysis, topic classification, spam detection, etc.
○
Token prediction tasks: Name entity prediction, Part of speech tagging, spelling correction
○
Natural language inference (NLI)
A: The boy is running through the grassy area
B: The boy is in his room.
Ans: Contradiction
B: The boy is outside.
Ans: Entailment
B: The boy is in a park.
Ans: Neutral

Page 10:
BERT use cases
Supervised!
Unsupervised!

Page 11:
Roberta (Robustly optimized BERT approach)
A trick and tuning study 
based on BERT
Dynamic masking > static
Next sentence prediction is 
not optimal
Larger batch + higher 
learning rate
RoBERTa: A Robustly Optimized BERT Pretraining Approach
Used in WangchanBERTa

Page 12:
Decoder only pre-training
●
GPT (Generative Pre-trained Transformer) pre-trains by predicting next word
●
Enforces causal attention (mask attention values on future tokens
●
Trained using teacher forcing. Inference in an auto-regressive manner – can 
generate a sentence!
BERT
GPT
Full attention matrix
Causal attention matrix
t2 can only attends to t2 
and t1
This is a pen
is a pen .
This MASK a pen
is
t1
t2
t3
t4

Page 13:
Encoder-Decoder
●
BART (Bidirectional and Auto-Regressive Transformers) combines BERT 
masking and GPT autoregressive characteristics
●
Mask spans of words into a single [MASK].
○
The model has to expand the mask into different amount of words in an autoregressive 
manner
●
Not used much after people figuring out that GPT-like models is good enough 
for generation while simpler
https://arxiv.org/abs/1910.13461

Page 14:
Transformers and scaling

Page 15:
Scaling law in language models
Scaling Laws for Neural Language Models https://arxiv.org/pdf/2001.08361.pdf
More data more params more 
compute leads to better models

Page 16:
Scaling law in language models
Scaling Laws for Neural Language Models https://arxiv.org/pdf/2001.08361.pdf
Training usually stops early for 
efficiency reasons

Page 17:
Scaling law in language models
LLaMA: Open and Efficient Foundation Language Models https://arxiv.org/abs/2302.13971
Longer training can be beneficial

Page 18:
Scaling law in language models
The Pile: An 800GB Dataset of Diverse Text for Language Modeling https://arxiv.org/pdf/2101.00027.pdf
Besides data size, data 
quality matters

Page 19:
Scaling law in language models
Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research 
https://arxiv.org/abs/2402.00159
Besides data size, data 
quality matters

Page 20:
WangchanBERTa: 
Pretraining Transformer-based Thai Language Models
Slides courtesy of Lalita Lowphansirikul and Charin Polpanumas

Page 21:
Hardware; 4 K80s vs 8 V100s vs 1,024 v100s
Better hardware means you have more room for iterations. We can iterate with smaller datasets 
but that sometimes defeat the purpose of training a LARGE language model.
Model
# of GPUS
Dataset Size
Effective Batch Size Steps
Days Spent
ThaiKeras BERT-th
4 K80s
0.5GB
32
1M
20
WangchanBERTa
8 V100s
78GB
4,092
500k
134
RoBERTa
1,024 V100s
160GB
8,000
500k
1
XLM-RoBERTa
500 V100s
2.5TB
8,192
1.5M
NA

Page 22:
Data Volume and Diversity
Model
# of GPUS
Dataset Size
Effective Batch Size Steps
Days Spent
ThaiKeras BERT-th
4 K80s
0.5GB
32
1M
20
WangchanBERTa
8 V100s
78GB
4,092
500k
134
RoBERTa
1,024 V100s
160GB
8,000
500k
1
XLM-RoBERTa
500 V100s
2.5TB
8,192
1.5M
NA
•
Thai Wikipedia is over 100x smaller than Thai texts used to train XLM-RoBERTa (71.7GB) 
and over 300x smaller than texts used to train the original RoBERTa.
•
Wikipedia also only include formal texts from encyclopedia.

Page 23:
Short Sequence Length
Text; Sequence length=128; WangchanBERTa SentencePiece tokenizer
<s> hamtaro หรือแฮมทาโร่แก๊งจิ๋วผจญภัยการ์ตูนญี่ปุ่นที่มีเหล่าแฮมสเตอร์เป็นตัวละครหลักเป็นผลงานของอาจารย์
kawai ritsuko เดิมทีแฮมทาโร่นั้นเป็นนิทานส าหรับเด็กมาก่อนถูกตีพิมพ์ที่ญี่ปุ่นในปีค.ศ. 1997 เพราะกองบรรณาธิการของ
นิตยสารการ์ตูนอยากได้การ์ตูนที่มีตัวเอกเป็นแฮมสเตอร์และอาจารย์ก็ก าลังเลี้ยงแฮมสเตอร์อยู่พอดีไม่แปลกใจเลย
ท าไมอาจารย์ถึงวาดการ์ตูนและถึงเล่าถึงกิจวัตรประจ าวันของเหล่าแฮมสเตอร์ได้ออกมาสมจริงและน่ารักสุดๆหนังสือ
นิทานhamtaro ได้รับความนิยมมากจนกลายมาเป็นทีวีอนิเมะในหน้าร้อนของปีค.ศ. 2000 เป็นที่นิยมทั้งเด็กทุกวัยไป
จนถึงวัยรุ่นความฮอตไม่หยุดอยู่ที่ญี่ปุ่นนะคะทีวีอนิเมะhamtaro ได้ออกอากาศในหลายประเทศรวมทั้งประเทศไทย
ด้วยแถมยังถูกสร้างเป็นภาพยนตร์และเกมอีกด้วยสินค้าที่ออกมาก็เยอะมาก</s>
Model
# of GPUS
Dataset Size
Effective Batch Size Sequence Length
ThaiKeras BERT-th
4 K80s
0.5GB
32
128
WangchanBERTa
8 V100s
78GB
4,092
416
RoBERTa
1,024 V100s
160GB
8,000
512
XLM-RoBERTa
500 V100s
2.5TB
8,192
512

Page 24:
Tokenization; Most Subword Tokenizers Are Domain Dependent
Even same SentencePiece tokenizers might get different results with different training set. Moreover, 
WordPiece tokenizer tokenizes too small subwords; we will see in later sections how this leads to a 
challenge in question answering task.
Text: ศิลปะไม่เป็นเจ้านายใครและไม่เป็นขี้ข้าใคร
WangchanBERTa (spm): ['<s>', '', 'ศิลปะ', 'ไม่เป็น', 'เจ้านาย', 'ใคร', 'และ', 'ไม่เป็น', 'ขี้ข้า', 'ใคร', '</s>']
WangchanBERTa-processed (spm): ['<s>', '', 'ศิลปะ', 'ไม่เป็น', 'เจ้านาย', 'ใคร', '<_>', 'และ', 'ไม่เป็น', 'ขี้ข้า', 'ใคร', '</s>']
XLM-RoBERTa (spm): ['<s>', '', 'ศิลปะ', 'ไม่เป็น', 'เจ้า', 'นาย', 'ใคร', 'และ', 'ไม่เป็น', 'ขี้', 'ข้า', 'ใคร', '</s>']
MBERT (WordPiece): ['[ C L S ]', 'ศ', '# #ิิล', '# # ป', '# # ะ', '# # ไ', '# # ม', '# #ิ่', '# # เปิ็น', '# # เ', '# # จ', '# #ิ้า', '# # นา',
'# # ย', '# # ใ', '# # คร', 'และ', '# # ไ', '# # ม', '# #ิ่', '# # เปิ็น', '# # ข', '# #ิี', '# #ิ้', '# # ข', '# #ิ้า', '# # ใ', '# # คร', '[ S E P ]']

Page 25:
Space Tokens as Important Boundaries
SentencePiece will create tokens where a space token is merged another non-space token.
Text: ศิลปะไม่เป็นเจ้านายใครและไม่เป็นขี้ข้าใคร
WangchanBERTa: ['<s>', '', 'ศิลปะ', 'ไม่เป็น', 'เจ้านาย', 'ใคร', 'และ', 'ไม่เป็น', 'ขี้ข้า', 'ใคร', '</s>']
WangchanBERTa-processed: ['<s>', '', 'ศิลปะ', 'ไม่เป็น', 'เจ้านาย', 'ใคร', '<_>', 'และ', 'ไม่เป็น', 'ขี้ข้า', 'ใคร', '</s>']
XLM-RoBERTa: ['<s>', '', 'ศิลปะ', 'ไม่เป็น', 'เจ้า', 'นาย', 'ใคร', 'และ', 'ไม่เป็น', 'ขี้', 'ข้า', 'ใคร', '</s>']
mBERT: ['[ C L S ]', 'ศ', '# #ิิล', '# # ป', '# # ะ', '# # ไ', '# # ม', '# #ิ่', '# # เปิ็น', '# # เ', '# # จ', '# #ิ้า', '# # นา', '# # ย', 
'# # ใ', '# # คร', 'และ', '# # ไ', '# # ม', '# #ิ่', '# # เปิ็น', '# # ข', '# #ิี', '# #ิ้', '# # ข', '# #ิ้า', '# # ใ', '# # คร', '[ S E P ]']

Page 26:
Token size
Eyeballing for vocab size of SentencePiece
วิชา|ที่|อาจารย์|อร|รถ|พล|สอน|คือ| |ศาสตร์|ที่|นํา|ทฤษฎี|ทาง|ภาษา|ศาสตร์|มา|รวม|กับ|เทคโนโลยี|ต่างๆ| |เป็น|
การศึกษา|ที่ใช้|การ|ผสม|ผ|สาน|ระหว่าง|วิทยา|การ|ค|อมพิวเตอร์|และ|ทฤษฎี|ทาง|ภาษา|ศาสตร์
5k
25k
32k
วิชา|ที่|อาจารย์|อรรถ|พล|สอน|คือ| |ศาสตร์|ที่นํา|ทฤษฎี|ทาง|ภาษา|ศาสตร์|มา|รวมกับ|เทคโนโลยี|ต่างๆ| |เป็น|
การศึกษา|ที่ใช้|การผสมผสาน|ระหว่าง|วิทยาการ|ค|อมพิวเตอร์|และ|ทฤษฎี|ทาง|ภาษา|ศาสตร์
วิชา|ที่|อาจารย์|อรรถพล|สอน|คือ| |ศาสตร์|ที่นํา|ทฤษฎี|ทาง|ภาษาศาสตร์|มา|รวมกับ|เทคโนโลยี|ต่างๆ| |เป็น|
การศึกษา|ที่ใช้|การผสมผสาน|ระหว่าง|วิทยาการ|ค|อมพิวเตอร์|และ|ทฤษฎี|ทาง|ภาษาศาสตร์

Page 27:
Assorted Thai Texts
Wisesight, Chaos Theory and Pantip to the rescue
Wisesight contributed 51.44GB of data from Twitter, 
Facebok, Pantip, Instagram, and YouTube in 2019.
Chaos Theory/Pantip contributed 22.35GB of Pantip
data from 2015-2019.
More information https://www.youtube.com/watch?v=kXPMLX0vfYU&ab_channel=EkapolC

Page 28:
Transformer problems
●
Embedding layer size.
●
Attention matrix is NxN.
○
Problems in terms of compute and memory
○
Cannot scale to long sequences (limited context size)
●
Tokens are fixed.

Page 29:
ALBERT 2019
Want higher hidden units without growing the model. Factorized embedding matrix
Share attention layer parameters across layers. More stable training as a side 
effect.
Some experiments show dropout 
hurt performance
https://arxiv.org/abs/1909.11942
V = Vocab, H = Hidden, E = Lower Dimensional Embedding space

Page 30:
Sparse attentions
●
Instead of computing the full attention matrix, people have found that many 
parts can be dropped
Big Bird: Transformers for Longer Sequences 2020 https://arxiv.org/abs/2007.14062

Page 31:
FlashAttention
●
Low-level code optimization by taking advantage of the memory heirachy of 
the GPU
●
Higher FLOP but lower memory transfer = W in speed (~3x)
●
Use less memory by using recompute rather than saving in memory (linear 
memory growth, ~20x savings at 4k sequence length)
https://arxiv.org/abs/2205.14135 2022
https://github.com/HazyResearch/flash-attention

Page 32:
FlashAttention

Page 33:
Flash attention 2
2023
https://arxiv.org/abs/2307.08691

Page 34:
Fixing limited context
●
If we use a pre-trained GPT model with token size of 1024, can we use it on a 
token size of 2048?
○
Not really
○
Positional embeddings were fixed at training (up to 1024)
○
Model only saw up to 1024 length, might not be able to generate something longer

Page 35:
XL-transformer
Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context 2019 https://arxiv.org/abs/1901.02860
Chunking limits the length 
of context
Let next chunk attention 
has access to previous 
chunk 
Used in XL-Net and others
This is quite similar to RNNs

Page 36:
RoPE (Rotary Position Embedding)
●
We want the transformer to generalize beyond the trained context length.
●
There are three cars. vs Three cars are there.    - Three and cars should have 
similar attention properties 
https://arxiv.org/abs/2104.09864 2021
3
4
1
2

Page 37:
RoPE
Sets of rotations for each 2D dimension
q k v representations can be written as
x – input embedding, p – position embedding
W – weight matrix for q, k, v
RoPE reformulates
Dot product of q and k yields a relative 
difference in position term (m-n)
RoPE is used in Llama 2 and 3, and Qwen 2

Page 38:
LongRoPE, 2024
●
Even with RoPE extending a transformer
beyond the trained context size requires
fine-tuning on longer text
●
Downscale RoPE position non-uniformly
https://arxiv.org/abs/2402.13753

Page 39:
Group Query Attention (GQA), 2023
●
Group multiple K and V from multiple head into groups
●
Smaller memory footprint, better GPU utilization (KV-cache)
●
Better modeling splitting in multi-GPU settings.
KV-cache https://medium.com/@joaolages/kv-caching-explained-276520203249
https://arxiv.org/abs/2305.13245

Page 40:
Pre-LN vs Post-LN transformers
●
The original transformer puts Layer norm 
between the residual blocks (Post-LN)
●
Pre-LN puts the layer norm inside.
○
Helps make the training more stable
○
Help eliminates the warm-up step in network 
training.
On Layer Normalization in the Transformer Architecture 2020 
https://arxiv.org/abs/2002.04745

Page 41:
RMS Layer Norm (Root Mean Square Layer Norm)
●
Like Layer Norm, but does not rescale (just normalizes). More compute 
efficient than layer norm.
●
Used in LLMs to stabilize training and improve convergence.
○
“Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN 
Transformers” discusses how these are equivalent.
https://arxiv.org/abs/1910.07467 Root Mean Square Layer Normalization 2019
RMS Layer norm
Layer norm

Page 42:
QWEN-2.5 Tech Report, Dec 2024
●
Let’s see how much we understand QWEN-2.5 architecture by this point
https://arxiv.org/abs/2412.15115

Page 43:
DeepSeek V3, 27 Dec 2024
https://arxiv.org/abs/2412.19437

Page 44:
On tokenization
●
Tokens remains a challenge
●
Bad tokens leads to
○
OOVs
○
Longer than expected context length
GPT4o ~1500
unique tokens for Thai

Page 45:
Case study: PhayaThaiBERT, 2023
●
Adds English loanword tokens to WangchanBERTa (Sentencepiece Unigram)
○
Vocab size increases from 25,005 to 249,262 (randomly initialized new vocabs)
○
Model params from 106M parameters to 278M parameters.
https://arxiv.org/abs/2311.12475

Page 46:
Beyond Transformer?
●
Some people say we are hitting a wall in transformer architecture.
●
There are researches on newer architecture choices. Most are tackling the 
length/memory issue. 
●
S4, MAMBA, Jamba – State-space model inspired. Has RNN flavors.
●
Titans
https://www.ai21.com/blog/announcing-jamba
https://arxiv.org/abs/2501.00663

Page 47:
Summary
●
3 main architectural choices: Encoder only, encoder-decoder, decoder only.
●
Several techniques trying to improve the attention layer bottleneck: hardware, 
algorithmic, systems.
●
Tokenization still plays a big role.

