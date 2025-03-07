Page 1:
+
Word Representations
Slides adapted from Assoc. Prof.Peerapon
Vateekul, Ph.D.
Department of Computer Engineering, 
Faculty of Engineering, Chulalongkorn University
1

Page 2:
+ Outline
■1) How to represent words?
■Symbolic vs distributional word 
representations
■2) Distributional: Sparse vector 
representations (discrete representation)
■Term-document matrix
■TF-IDF
■3) Distributional: Dense vector representations
■Word2Vec
■CBOW
■Word2Vec training methods
■Pre-trained vector representations
■Adaptation
■Compositionality
2

Page 3:
+
Part 1: How to represent words?
Symbolic vs distributional word representations
3

Page 4:
+ How to represent words? 
Symbolic vs Distributional representations 
■Representing words in computer is one of the most important tasks in NLP.
■Words = Input for our models
4
1) symbolic
2) distributional
Knowledge 
graph
One-hot model

Page 5:
+ How to represent words? 
Symbolic vs Distributional representations (cont.)
■Symbolic Representations
■A lexical database, such as WordNet that has hypernyms and synonyms 
■Drawback:
■Requires human labor to create and update
■Missing new words, nuances
■Hard to compute accurate word similarity 
5
1
Knowledge graph

Page 6:
+ How to represent words? 
Symbolic vs Distributional representations (cont.)
■Symbolic Representations
■Earlier work in NLP, the vast majority of (rule-based and statistical) NLP models 
considered words as discrete atomic symbols. 
■E.g. One-hot model
เสือ= [ 0 1 0 0 0 … 0 0 ]
สัตว์กินเนื้อ= [ 0 0 0 0 1 … 0 0 ]
* Each point in the vector represents each vocab.
■Drawback:
■Cannot capture similarity between words
6
1

Page 7:
+ How to represent words? 
Symbolic vs Distributional representations (cont.)
■Distributed representations (aka distributional methods)
■“You shall know a word by the company it keeps” (J. R. Firth 1957)
■The meaning of a word is computed from the distribution of words around it.
■Can encode similarity between words!
7
2

Page 8:
+ How to represent words? 
Symbolic vs Distributional representations (cont.)
■Most modern NLP models use distributional word representations to represent words
■Word meaning as a vector
■In this class, we will examine the development of distributed word representations 
before the rise of deep learning, then we will introduce you to word representation 
techniques used in deep learning models.
8
1) Sparse Representation
2) Dense Representation

Page 9:
+
Part 2: Distributional: Sparse vector 
Term-document matrix
Co-occurrence matrix
TF-IDF
9

Page 10:
+ Sparse vector representations
in Distributional Representations
10
1. Term-Frequency (Raw Frequency)
2. Co-occurrence matrix
3. PPMI
4. TF-IDF

Page 11:
+ Sparse vector representations: 
term-document matrix 
11
Reference: Jurafsky, Dan, and James H. Martin. Speech and language processing.  3rd edition draft, 
https://web.stanford.edu/~jurafsky/slp3/, August  2017 
■Each row represents a word in the vocabulary and term-document matrix
■Each column represents a document.
document
vocabulary
1

Page 12:
+ Sparse vector representations: 
term-document matrix (cont.)
12
Reference: Jurafsky, Dan, and James H. Martin. Speech and language processing.  3rd edition draft, 
https://web.stanford.edu/~jurafsky/slp3/, August  2017 
■Application: Document Information Retrieval
■Two documents that are similar tend to have similar words/vectors (document similarity)

Page 13:
+ Sparse vector representations: 
term-document matrix (cont.)
13
Reference: Jurafsky, Dan, and James H. Martin. Speech and language processing.  3rd edition draft, 
https://web.stanford.edu/~jurafsky/slp3/, August  2017 
■Documents as vectors
■Two documents are similar if their vectors are similar (document similarity)
document
vocabulary

Page 14:
+ Sparse vector representations: 
term-document matrix (cont.)
14
Reference: Jurafsky, Dan, and James H. Martin. Speech and language processing.  3rd edition draft, 
https://web.stanford.edu/~jurafsky/slp3/, August  2017 
■Words as vectors
○
Two words are similar if their vectors are similar (word similarity)
document
vocabulary

Page 15:
+ Sparse vector representations: 
co-occurrence matrix (1)
15
Reference: Jurafsky, Dan, and James H. Martin. Speech and language processing.  3rd edition draft, 
https://web.stanford.edu/~jurafsky/slp3/, August  2017 
■Word-word or word-context matrix
■Instead of entire documents, use smaller contexts
■Two words are similar if their vectors are similar
vocabulary
2
window = 4

Page 16:
+ Sparse vector representations: 
co-occurrence matrix (2)
16
Reference: Jurafsky, Dan, and James H. Martin. Speech and language processing.  3rd edition draft, 
https://web.stanford.edu/~jurafsky/slp3/, August  2017 
■Two similar words tend to have similar vectors (word similarity)

Page 17:
+ Sparse vector representations: 
Positive Pointwise Mutual Information (PPMI) 
■Problems with raw frequency 
■Not very discriminative (need normalization)
■Words such as “it, the, they, a, an, the” occur very frequently
■, but are not very informative
■PPMI incorporates the idea of mutual information to determine the informative 
context words
■We need a measure which tells us which context words are informative about 
the target word
17
3

Page 18:
+ Sparse vector representations: 
Positive Pointwise Mutual Information (PPMI) (cont.)
18
How often the two words occur together 
w – target word
c – context word
Do words “w” and “c” co-occur more than if they were 
independent?
How often the two words occur if they 
occur independently (occur by chance)
+ : occur together > occur by chance
0 : occur together = occur by chance
- : occur together < occur by chance

Page 19:
+ Sparse vector representations: 
Positive Pointwise Mutual Information (PPMI) (cont.)
19
Negative PMI values tend to be unreliable. 
It is common to replace all negative PMI values with zero

Page 20:
+ Sparse vector representations: 
Positive Pointwise Mutual Information (PPMI) (4)
20
Reference: Jurafsky, Dan, and James H. Martin. Speech and language processing.  3rd edition draft, 
https://web.stanford.edu/~jurafsky/slp3/, August  2017 
vocabulary
PPMI matrix
word co-occurrence matrix

Page 21:
+ Sparse vector representations: TF-IDF 
Need for normalization in TF
■Term Frequency (TF) – per each document
■Inverse Document Frequency (IDF) – per corpus (all documents)
■TF-IDF
21
4
penalty score
i.e., a, an, the
Doc1
cat = 5/10
Doc2
cat = 50/1000

Page 22:
+ Sparse vector representations: TF-IDF (Cont.)
22
Reference: Jurafsky, Dan, and James H. Martin. Speech and language processing.  3rd edition draft, 
https://web.stanford.edu/~jurafsky/slp3/6.pdf, Feb  2020 
tf-idf matrix

Page 23:
+ Sparse vector representations: TF-IDF (Cont.)
■A very popular way of weighting in Information Retrieval (document similarity)
■Not commonly used as a component to measure word similarity (it is designed for 
document similarity)
23

Page 24:
+ BM25 (Best Matching)
24
– Term Saturation and diminishing return
– Document Length Normalization
https://towardsdatascience.com/understanding-term-based-retrieval-methods-in-information-retrieval-2be5eb3dde9f
If a document contains 100 
occurrences of the term 
“computer,” is it really twice as 
relevant as a document that 
contains 50 occurrences?
If a document is very short and it 
contains “computer” once, that 
might already be a good indicator 
of relevance. But if the document is 
really long and the term 
“computer” only appears once, it 
is likely that the document is not 
about computers.
RAG performance using different document representation

Page 25:
+ Part 3: Distributional: Dense vector 
representations
25

Page 26:
+ Dense vector representations 
■Sparse vector representations such as Term-Document vectors are:
■Long (length of vector ≈ 20,000 to 50,000 )
■Sparse (most elements are zero)
■Dense vector representations  are introduced to:
■Reduce length of vectors (length of vector ≈ 200 to 1,000 )
■Reduce sparsity  (hence the name “dense”; most elements are not zero)
26

Page 27:
+ Dense vector representations (cont.)
■Advantages of dense vector representation
■Less parameters to tune
■Generalize better
■Better at capturing synonyms
27

Page 28:
+ Dense vector representations : 
neural networks
How do we train embeddings in neural 
networks?
1. Initialize randomly and train it on a 
your target task.
2. Pre-train on language modeling task 
(e.g. next word prediction, cloze 
(predict missing words))
3. Pre-train on supervised task 
■
e.g. train on POS tagging task and use 
it on other tasks
28
reference: Neubig (2020), https://www.youtube.com/watch?v=RRaU7pz2eT4
1

Page 29:
+ Dense vector representations : 
neural networks (cont.)
■Tomas Mikolov introduced Skip-gram in 2013
■CBOW was proposed before by other researchers.
■Train a neural network to predict neighboring words
■Word representation can be learned as a part of the process of word prediction.  
■Part of a neural network (embedding layer) can be used as word representation in 
various NLP tasks
■Advantages:
■Fast
■Pre-trained word representations are available online!  
29

Page 30:
+ Dense vector representations : 
neural networks (cont.)
30
Image reference:  Boonkwan, Prachya. “Word2Vec: When Language Meets Number Crunching”, 
https://goo.gl/hhA3hO,  Feb 2017
Weight Matrix (V × N dim)
Each vocab
Each hidden (latent)
dog
Lookup Table

Page 31:
+ Dense vector representations : 
neural networks (cont.)
■CBOW and Skip-Gram (not covered) intuition: Iteratively make the embeddings 
for a word 
■(positive class) more like the embeddings of its neighbors and
■(negative class) less like the embeddings of other words.
31
Reference: Jurafsky, Dan, and James H. Martin. Speech and language processing.  3rd edition draft, 
https://web.stanford.edu/~jurafsky/slp3/, August  2017 
Outside of China there have been more than 500 cases in nearly 30 countries. Four people 
have died - in France, Hong Kong, the Philippines and Japan.
positive words
negative  words
target word
https://www.bbc.com/news/world-asia-china-51519055

Page 32:
+ 1) CBOW
■Continuous Bag-of-Words (CBOW)
■In CBOW neural language model,  ONE target word is 
predicted from SEVERAL context words.
32
Image reference:  Boonkwan, Prachya. “Word2Vec: When Language Meets Number Crunching”, 
https://goo.gl/hhA3hO,  Feb 2017
predicted target word
context
words
context
words
Outside of China there have been more than 500 cases in nearly 30 
countries. Four people have died - in France, Hong Kong, the 
Philippines and Japan.

Page 33:
+ 1) CBOW (cont.)
■We simply average the encoding vectors 
33
Image reference:  Boonkwan, Prachya. “Word2Vec: When Language Meets Number Crunching”, 
https://goo.gl/hhA3hO,  Feb 2017
predicted target word
context
words
context 
words
dog
Outside of China there have been more than 500 cases in nearly 30 
countries. Four people have died - in France, Hong Kong, the 
Philippines and Japan.

Page 34:
+ 2) Skip-gram 
■In skip-gram neural language model,  SEVERAL context 
words are predicted from ONE target word.
■In “Efficient Estimation of Word Representations in 
Vector Space”, Mikolov shows that skip-gram performs 
better than CBOW in several tasks. BUT skip-gram 
model requires more training time.
■In this lecture, we will show you how skip-gram works
34
target
word
predicted
context words
predicted
context words
Outside of China there have been more than 500 cases in nearly 30 
countries. Four people have died - in France, Hong Kong, the 
Philippines and Japan.

Page 35:
+ 2) Skip-gram (cont.)
■Skip-gram prediction
■Consider the following passage:
■“I think it is much more likely that human language learning 
involves something like probabilistic and statistical inference but 
we just don't know yet.”
35

Page 36:
+ 2) Skip-gram (cont.)
■For each word t = 1…T,  predict its surrounding context words (next m words 
and previous m words)
■“m” is the window size
36

Page 37:
+ 2) Skip-gram (cont.)
■Likelihood function: Given the target word (aka center word), maximize the 
probability of each context word
■Cost/Loss Function (Negative Log-Likelihood): (minimize)
37
Reference: http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture2.pdf
j = -m   →previous words
j = +m →next words
j = 0     →the input word (wt)

Page 38:
+ 2) Skip-gram (cont.)
Softmax
■How to calculate 
?
■
for each word w, we will use two vectors
■
when w is a target/center word
■
when  w is a context word
■
Then for a center word ‘c’ and a context word ‘o’
38
Reference: http://web.stanford.edu/class/cs224n/lectures/lecture2.pdf
Dot product compares similarity of o and c. 
Larger dot product = larger probability
After taking exponent, normalize over entire 
vocabulary
38
statistical
the

Page 39:
+ 2) Skip-gram (cont.)
Softmax
■This is basically a softmax function
■The softmax function maps arbitrary 
values xi to a probability distribution pi
■“max” because amplifies probability of 
largest xi
■“soft” because still assigns some 
probability to smaller xi
39
Reference: http://web.stanford.edu/class/cs224n/lectures/lecture2.pdf

Page 40:
+ 2) Skip-gram (cont.)
■Negative Log-Likelihood of Skip-gram 
model:
■Notice the difference between skip-gram’s 
cost function and the cost function of neural 
language model from the previous lectures 
40
Predict context words 
Predict the next word 

Page 41:
+ Dealing with large vocab sizes
■Word2Vec training methods 
■Softmax is not very efficient (slow)
■Computational Cost : O(|V|)
■Solution: Two efficient training methods
■Hierarchical Softmax: O(log(|V|)) – not covered
■Negative Sampling
41

Page 42:
+ Word2Vec training methods : 
Solution1: Hierarchical Softmax
■Softmax as tree traversal
■Hierarchical softmax uses a binary tree to 
represent words
■Each leaf is a word
■There’s a unique path from root to leaf
■The probability of each word is the product of 
branch selection decisions from the root to 
the word’s leaf
42
Image reference: http://building-babylon.net/2017/08/01/hierarchical-
softmax/

Page 43:
+ Word2Vec training methods : 
Solution2: Negative Sampling (2)
43
■The objective function for skip-gram with negative sampling:
Negative samples
(negative, -1)
Context word
(positive, +1)
= sigmoid

Page 44:
+ Word2Vec training methods : 
Solution2: Negative Sampling (2)
■Why ¾?
■Chosen based on empirical experiments
■Intuition:
■at: 0.93/4 = 0.92
■farmer: 0.093/4 = 0.16
■superfluous: 0.013/4 = 0.032 
■A rare word such as ‘superfluous’ is now 3x  more likely to be 
sampled 
■While the probability of a frequent word “at” only went up 
marginally
44

Page 45:
+ Summary: CBOW vs Skip-gram (recap)
45
https://fasttext.cc/docs/en/unsupervised-tutorial.html

Page 46:
+ Pre-trained Word2Vec
■1) Non-contextualized Word Embeddings (fixed vector)
■1.1) GloVe (Stanford) [not support Thai language]
■https://nlp.stanford.edu/projects/glove/
■1.2) fastText [Available in Thai language] (Facebook)
■https://github.com/facebookresearch/fastText
■1.3) Word2Vec in TLTK (Aj.Wirote)
■tltk.corpus.w2v(w) 
■1.4) Large Thai Word2Vec (LTWV): CBOW
■2) Contextualized Word Embeddings
■2.1) thai2fit 
■ULMFit
■https://github.com/cstorm125/thai2fit/
■2.2) BERT family
■2.3) A lot more…
46

Page 47:
+ Pre-trained Word2Vec: fastText
Skip-Gram (or CBOW) of “sum of subwords (char n-grams)”
■fastText is a library for efficient learning of word representations and sentence 
classification.
■Character n-grams as additional features to capture some partial information 
about the local word order.
■Good for rare words, since rare words can share these n-grams with common words
■Pre-trained word vectors for 157 languages (including Thai) trained on Wikipedia. 
47
Reference:  Bojanowski, Piotr, et al. "Enriching word vectors with subword information." arXiv preprint arXiv:1607.04606 (2016).
Joulin, Armand, et al. "Bag of tricks for efficient text classification." arXiv preprint arXiv:1607.01759 (2016).
https://cai.tools.sap/blog/glove-and-fasttext-two-popular-word-vector-models-in-nlp/

Page 48:
+ fastText library by Facebook
48
https://fasttext.cc/docs/en/crawl-vectors.html

Page 49:
+ Compositionality
■Now, we know how to create a dense 
vector representation for a word
■What about larger linguistic units? (e.g. 
phrase, sentence )
■We can combine smaller units into a 
larger unit
49
Image ref: Prof. Regina Barzilay , NLP@MIT

Page 50:
+ Outline
■1) How to represent words?
■Symbolic vs distributional word 
representations
■2) Distributional: Sparse vector 
representations (discrete representation)
■Term-document matrix
■Co-occurence
■PPMI
■TF-IDF
■3) Distributional: Dense vector representations
■Word2Vec
■CBOW
■Skip-gram
■Word2Vec training methods
■Pre-trained vector representations
■Adaptation
■Compositionality
50

