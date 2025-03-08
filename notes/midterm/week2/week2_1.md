# Language Modeling

A language model assigns probability to a sentence, or predict next word.

## Probability

We intuitively calculate probability of a sentence using the following concept:

- Conditional Probability

```math
P(B \mid A) = \frac{P(A,B)}{P(A)}
```

```math
P(A,B) = P(B \mid A) \times P(A)
```

- Chain Rule

```math
P(A,B,C,D) = P(A) \times P(A \mid B) \times P(C \mid A,B) \times P(C \mid A,B,C)
```

Full estimation has its limitation:
- There will always be new sentences

## N-grams

A way to language modeling based on **Markov Assumption**: probability of some future unit (next word) without looking too far into the past.

**unigram, bigrams, trigrams or n-
grams**

```math
P(w_{i} \mid w_{i-1}) = \frac{c(w_{i-1},w_{i})}{c(w_{i-1})}
```

We do everything in log space  $( \ln (P(S)) )$ to

- Avoid underflow (numbers too small)
- Also, adding is faster than multiplying

```math
\ln (P(A,B,C,D)) = \ln (P(A)) + \ln (P(B \mid A)) + \ln (P(C \mid A,B)) + \ln (P(D \mid A,B,C))
```

## Evaluation

**Extrinsic Evaluation**:

- Measure the performance of a downstream task (e.g. spelling correction, machine 
translation, etc.)
- Cons: Time-consuming

**Intrinsic Evaluation**:

- Evaluate the performance of a language model on a hold-out dataset (test set): **Perplexity**!
- Cons: An intrinsic improvement does not guarantee an improvement of a 
downstream task, but perplexity often correlates with such improvements

> Improvement in perplexity should be confirmed by an evaluation of a real task

### Perplexity

Perplexity is a quick evaluation metric for language models. I is the inverse probability of the test set, normalized by the number of words:

```math
PP(W) = \sqrt[N]{\prod_{i = 1}^{N} \frac{1}{P(w_{i} \mid w_1 \dots w_{i-1})}}
```

- Logarithmic Version:

```math
b^{-\frac{1}{N} \sum_{i = 1}^{N} \log_{b} (P(w_{i} \mid w_1 \dots w_{i-1}))}
```

- As we work on log likelihood

```math
e^{-\frac{1}{N} \sum_{i = 1}^{N} \ln (P(w_{i} \mid w_1 \dots w_{i-1}))}
```

```math
\text{Lower perplexity is better!}
```

## Smoothing

Problems with dealing with probability.

### Zeros

combinations that appear in the test set, but not in training set (each word still in the vocab list).

### Unknown words (UNK)

Words we have never seen before in training set and not in vocab list (OOV (out of vocabulary)). Here are some ways to deal with this problem:

#### Assign it as a probability of normal word

1. Create a set of vocabulary with minimum frequency threshold
  - This is fixed in advance.
  - Or from top n frequency
  - Or words that have frequency more than 1,2,..,v

2. Convert any words in training and testing that is not in this predefined set
  - to ‘UNK’ token.
  - Simply, deal with UNK word as a normal word

```math
P(UNK) = \frac{wc(UNK_{freq = 1})}{wc(total)}
```

#### Or just define probability of UNK word with constant value

For example:

```math
P(UNK) = \frac{1}{total \  vocb}
```

> **However, this still cannot solve the zero issue.**

### Add-one estimation 

Add-one estimation (or **Laplace smoothing**)
- We add one to all the n-grams counts
- For bigram, where V is the number of unique words in the corpus:

```math
P(S) = \frac{c(w_{i}, w_{i = 1}) + 1}{c(w_{i = 1}) + V}
```

**Pros**

- Easiest to implement

**Cons**

- Usually perform poorly compared to other techniques
- The probabilities change a lot if there are too many zeros n-grams
  - useful in domains where the number of zeros isn’t so huge

### Backoff

**Key Idea**: 

- Use less context for contexts you don’t know about
- Use only the best available n-grams if you have good evidence
- otherwise backoff!

**Example**:

```math
\text{Tri-gram} \rightarrow \text{Bi-grams} \rightarrow \text{Unigram}
```

### Interpolation

Key Idea: mix unigram, bigram, trigram

```math
\hat{P}(w_{n} \mid w_{n-2} w_{n-1}) = \lambda_{3} P(w_{n} \mid w_{n-2} w_{n-1}) + \lambda_{2} P(w_{n} \mid w_{n-1}) \lambda_{1} P(w_{n}) + \lambda_{0} P(UNK)
```

> $\lambda$ is chosen from testing on validation data set, and the summation of $\lambda_{i}$ is 1 $(\sum \lambda_{i} = 1)$

■
Interpolation is like merging several models

Page 37:
+ Smoothing#3: Interpolation (cont.)
39
I
want
to
eat
chinese
food
lunch
spend
Total
2533
927
2417
746
158
1093
341
278
8493
0.2982
0.1091
0.2846
0.0878
0.0186
0.1287
0.0402
0.0327
1.0000
i
want
to 
eat
chinese
food
lunch
spend
i
0.002
0.33
0
0.0036
0
0
0
0.00079
want
0.0022
0
0.66
0.0011
0.0065
0.0065
0.0054
0.0011
to
0.00083
0
0.0017
0.28
0.00083
0
0.0025
0.087
eat
0
0
0.0027
0
0.021
0.0027
0.056
0
chinese
0.0063
0
0
0
0
0.52
0.0063
0
food
0.014
0
0.014
0
0.00092
0.0037
0
0
lunch
0.0059
0
0
0
0
0.0029
0
0
spend
0.0036
0
0.0036
0
0
0
0
0
■
Interpolation for Bigram
■
Where C is a constant (often = 1/vocabulary) in corpus and vocabulary size = 1,446
“eat spend”

Page 38:
+ Absolute discounting: save some probability 
mass for the zeros
■
Suppose we want to subtract a little from a count of 4 
to save probability mass for the zeros?
■
How much to subtract?
■
Church and Gale (1991)
■
AP newswire dataset 
■
22 million words in training set
■
next 22 million words in validation set
■
On average, a bigram that occurred 4 times in the 
first 22 million words (training) occurred 3.23 times
in the next 22 million words (validation)
■
So the discrepancy between train & validate of “only 
this word” is 4 - 3.23  = 0.77
■
The average discrepancy of all words is about 0.75! 
(called discount, d)
40
Bigram count in 
training 
Bigram count in 
validation set
0
0.0000270
1
0.448
2
1.25 (~ -0.75)
3
2.24 (~ -0.75)
4
3.23 (~ -0.75)
5
4.21 (~ -0.75)
6
5.23 (~ -0.75)
7
6.21 (~ -0.75)
8
7.21 (~ -0.75)
9
8.26 (~ -0.75)

Page 39:
+ Absolute discounting: save some probability 
mass for the zeros (cont.)
■
Absolute discounting formalizes this intuition by 
subtracting a fixed (absolute) discount d 
(d=0.75) from each count and give to zero counts.
■
BUT should we just use the regular unigram? 
■
Solution: Kneser–Ney Smoothing
41
Bigram count in 
training
Bigram count in 
validation set
0
0.0000270
1
0.448
2
1.25
3
2.24
4
3.23
5
4.21
6
5.23
7
6.21
8
7.21
9
8.26
a
b
c
a
10
0
0
b
c
a
b
c
a
9/10
?
?
b
c
P(b) = 0.1, P(c) = 0.3
P(b|a) = 0 + xP(b)
P(c|a) = 0 + xP(c)
xP(b) + xP(c) = 0.1

Page 40:
+ Smoothing#4: Kneser–Ney Smoothing 
■Kneser–Ney Smoothing 
■Similar to interpolation, but better estimation for probabilities of lower-order grams (like 
unigram)
■Ex: I can’t see without my reading ___ .
■The blank word should be glasses, but if we only consider unigram, a word like 
Francisco has higher probability
■But, Francisco always follows San (San Francisco).
■We should use continuation probability instead (i.e. how likely a word is a continuation of 
any word)
42

Page 41:
+ Smoothing#4: Kneser–Ney Smoothing (cont.)
43
■
Kneser–Ney Smoothing
■
How many word types precede w?
■
|{wi : c(wi,w)>0}|
■
Normalized by total number of word bigram types (all possible combinations)
■
If our corpus contains these bigrams
■
{ San Francisco, San Francisco, San Francisco, Sun glasses, Reading glasses, Colored 
glasses }
■
Pcont(Francisco) = (1/4) = 0.25 
■
Pcont(glasses) = (3/4) = 0.75 
■
Now, a word like “Francisco” will have low Pcontinuation

Page 42:
+ Smoothing#4: Kneser–Ney Smoothing (cont.)
44
■
Kneser–Ney Smoothing
■
In case of bigram,
■
Where
■
d is a constant number, often set to 0.75
a number of word type that 
can precede wi-1
the normalized 
discount

Page 43:
+ Example: a bigram Kneser-ney 
Imagine we have the following training corpus:
<s> I am Sam </s>
<s> Sam I am </s>
<s> I am Sam </s>
<s> I like green eggs </s>
Train a bigram Kneser-ney model using the corpus above
46

Page 44:
+ Example: a bigram Kneser-ney (cont.)
Create a unigram counting table
47
training corpus:
<s> I am Sam </s>
<s> Sam I am </s>
<s> I am Sam </s>
<s> I like green eggs </s>

Page 45:
+ Example: a bigram Kneser-ney (cont.)
Create a bigram counting table
48
training corpus:
<s> I am Sam </s>
<s> Sam I am </s>
<s> I am Sam </s>
<s> I like green eggs </s>

Page 46:
+ Example: a bigram Kneser-ney (cont.)
Compute the log-likelihood of the sentence “<s> am Sam </s>”
Pkn2(am|<s>)=(max(0-0.75,0)/4)+(0.75*2/4)*(1/11) =0.03409  
Pkn2(Sam|am)=(max(2-0.75,0)/3)+(0.75*2/3)*(2/11) =0.5076  
Pkn2(</s>|Sam)=(max(2-0.75,0)/3)+(0.75*2/3)*(3/11)=0.5530
LL = ln(0.03409) + ln(0.5076) + ln(0.5530) = -4.6492 
49
training corpus:
<s> I am Sam </s>
<s> Sam I am </s>
<s> I am Sam </s>
<s> I like green eggs </s>

Page 47:
+ Example: a bigram Kneser-ney (cont.)
Compute the perplexity of the sentence “<s>  am Sam  </s>”
Perplexity = exp( -LL/n ) = exp( -(-4.6492) / 3 ) = 4.7
50

Page 48:
+ Smoothing Summary
■Summary
■1) Add-1 smoothing:
■OK for text categorization, not for language modeling
■For very large N-grams like the Web:
■2) Backoff
■The most commonly used method:
■3) Interpolation
■The best method
■4) Kneser–Ney smoothing
51

Page 49:
+ Reference/Suggested Reading:
Jurafsky, Dan, and James H. Martin. Speech and language processing. Chapter 3,  
https://web.stanford.edu/~jurafsky/slp3/3.pdf
52

Page 50:
+
Neural Language Model
53

Page 51:
+ Neural Language Model
■Traditional Language Model
■Performance improves with keeping around higher n-grams counts and doing smoothing 
and so-called backoff (e.g. if 4-gram not found, try 3-gram, etc)
■However,
■It needs a lot of memory to store all those n-grams
■It lacks long-term dependency
■"Jane walked into the room. John walked in too. It was late in the day, and everyone 
was walking home after a long day at work. Jane said hi to ___
54
i
want
to 
eat
chinese
food
lunch
spend
i
0.002
0.33
0
0.0036
0
0
0
0.00079
want
0.0022
0
0.66
0.0011
0.0065
0.0065
0.0054
0.0011
to
0.00083
0
0.0017
0.28
0.00083
0
0.0025
0.087
eat
0
0
0.0027
0
0.021
0.0027
0.056
0
chinese
0.0063
0
0
0
0
0.52
0.0063
0
food
0.014
0
0.014
0
0.00092
0.0037
0
0
lunch
0.0059
0
0
0
0
0.0029
0
0
spend
0.0036
0
0.0036
0
0
0
0
0

Page 52:
+ Neural Language Model (cont.)
■Recurrent Neural Network (RNN)
■Consider all previous word in the corpus
■In language modeling,
■Input (x) is current word in vector form
■Output (y) is the next word
■Usually, RNN’s performance is better than traditional language models
55

Page 53:
+ Neural Language Model (cont.)
■Recurrent Neural Network (RNN)
■A simple language model
56
Embedding layer
RNN
w1
RNN
RNN
w2
w3
Dense Layer
Softmax layer
w2
w3
w4
TARGET
Input
I eat Chinese food
i
want
to 
eat
chinese
food
lunch
spend
i
5
827
0
9
0
0
0
2
want
2
0
608
1
6
6
5
1
to
2
0
4
686
2
0
6
211
eat
0
0
2
0
16
2
42
0
chinese
1
0
0
0
0
82
1
0
food
15
0
15
0
1
4
0
0
lunch
2
0
0
0
0
1
0
0
spend
1
0
1
0
0
0
0
0

Page 54:
+ Neural Language Model (cont.)
■Recurrent Neural Network (RNN)
■A simple language model
57
Embedding layer
RNN
I
RNN
RNN
eat
Chinese
Dense Layer
Softmax layer
eat
Chinese
food
TARGET
Input
I eat Chinese food
i
want
to 
eat
chinese
food
lunch
spend
i
5
827
0
9
0
0
0
2
want
2
0
608
1
6
6
5
1
to
2
0
4
686
2
0
6
211
eat
0
0
2
0
16
2
42
0
chinese
1
0
0
0
0
82
1
0
food
15
0
15
0
1
4
0
0
lunch
2
0
0
0
0
1
0
0
spend
1
0
1
0
0
0
0
0

Page 55:
+ Neural Language Model (cont.)
58
Softmax (all classes V)
For each training example, 
Whole training data (T)

Page 56:
+ Neural Language Model (cont.)
■RNN suffers from vanishing gradient
■Use a RNN that has memory unit such as
■Long Short Term Memory (LSTM)
■Gate Recurrent Unit (GRU)
■Bidirectional RNN?
■Bidirectional RNN cannot apply here since 
we predict the next word and cannot use 
future information (violating assumption).
■However, special types of special networks 
(Transformer: BERT) can be applied without 
violating assumptions.
59
https://paperswithcode.com/method/bilstm#

Page 57:
+ Neural Language Model (cont.)
■Conclusion
■Neural Language Model vs. N-grams Model
■A competitive n-grams model needs huge amount of memory, larger 
than RNN
■Neural Language Model usually perform better than n-grams model 
because 
■it considers long-term dependency information
■It subtly processes word semantics via word embeddings
■However, n-grams are still quite useful and often are incorporated into 
neural language models as features or for beamsearch pruning. 
(ngrams →NN)
60
<s>,w1
w1,w2
w2,w3

Page 58:
+ Neural Language Model (cont.)
■[Y. Bengio, R. Ducharme, P. Vincent, and C. Janvin. 2003. A neural probabilistic language 
model. JMLR, 3:1137–1155] 
■This model only use Multilayer Perceptron and Word embedding, not even RNN
61

Page 59:
+ Neural Language Model (cont.)
■[Sundermeyer, Martin, Hermann Ney, and Ralf 
Schlüter. "From feedforward to recurrent LSTM neural 
networks for language modeling." IEEE Transactions 
on Audio, Speech, and Language Processing 23.3 
(2015): 517-529.]
■LSTM can be use with traditional techniques via 
interpolation to improve the result
62
N-Gram
MLP
RNN

Page 60:
+ Language Model SOTA (2019; outdated)
https://github.com/sebastianruder/NLP-progress/blob/master/english/language_modeling.md
63
●
Encoder Model: XLNet, BERT
●
Decoder Model: GPT, GPT-2, GPT-3
●
Encoder-Decoder Models: T5, BART

Page 61:
+
64
https://blog.tensorflow.org/2020/05/how-hugging-face-achieved-2x-performance-boost-question-answering.html
Outdated

Page 62:
+
65
https://medium.com/airesearch-in-th/wangchanberta-
%E0%B9%82%E0%B8%A1%E0%B9%80%E0%B8%94%E0%B8%A5%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%A1%E0%B8%A7%E0%B8%A5%E0%B8%9C%E0%B8%A5%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B2%E0%B9%84%E0%B
8%97%E0%B8%A2%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B9%83%E0%B8%AB%E0%B8%8D%E0%B9%88%E0%B9%81%E0%B8%A5%E0%B8%B0%E0%B8%81%E0%B9%89%E0%B8%B2%E0%B8%A7%E0%B8%AB%E0%B8%99%E0%B9%89%E
0%B8%B2%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%AA%E0%B8%B8%E0%B8%94%E0%B9%83%E0%B8%99%E0%B8%82%E0%B8%93%E0%B8%B0%E0%B8%99%E0%B8%B5%E0%B9%89-d920c27cd433

Page 63:
+ PhayaThaiBERT
66
https://arxiv.org/pdf/2311.12475

Page 64:
+
67
https://blogs.cfainstitute.org/investor/2023
/05/26/chatgpt-and-large-language-
models-six-evolutionary-steps/

Page 65:
+
68
https://www.reddit.com/r/Infographi
cs/comments/1c81n2o/the_size_of
_llms_apr_2024/?rdt=36268

Page 66:
+
69
https://epoch.ai/blog/tracking-large-scale-ai-models

Page 67:
+ Thai LLMs
70
~17B 
tokens
14B and 72B models also available
https://arxiv.org/pdf/2412.13702
https://huggingface.co/openthaigpt/openthaigpt1.5-7b-instruct

Page 68:
+ Large Language Models (LLMs)
As of Jan-2025, there are 163,240 LLMs in HF!
71
GPT-4(rumoured to be 1.8 trillion)
163,240 LLMs and counting!
https://medium.com/@vipra_singh/building-llm-applications-large-language-models-part-6-ea8bd982bdee
https://huggingface.co/models?pipeline_tag=text-generation&sort=trending

Page 69:
+ Multimodal LLM
72
https://medium.com/@cout.shubham/exploring-multimodal-large-language-models-a-step-forward-in-ai-626918c6a3ec

Page 70:
+ Reasoning Models
GPT-o1 and o3 (Dec 2024)
QwQ (qwen) - open-source!
73
https://lunary.ai/blog/open-ai-o1-reasoning-models
https://www.thealgorithmicbridge.com/p/openai-o3-model-is-a-message-from
o3 →~$3000/task
Credit Aj.Piyalitt Ittichaiwong's post 
on 2 Jan 2025

Page 71:
+ Conclusion
74
■Introduction
■N-grams
■Evaluation and Perplexity
■Smoothing
■Neural Language Model

Page 72:
+
Thank you J
75

