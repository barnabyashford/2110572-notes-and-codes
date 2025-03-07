Page 1:
TEXT CLASSIFICATION
Intent, topic, sentiment, etc.

Page 2:
Wongnai Challenge
• Predict star rating from review text
input
output

Page 3:
Yelp reviews
Duyu Tang Document Modeling with Gated Recurrent Neural Network
for Sentiment Classification, 2015 http://aclweb.org/anthology/D15-1167
Wongnai challenge top model Accuracy 0.5844
Thai2fit (contextualized word embedding with adaptation) Accuracy 0.60925

Page 4:
Text/document classification
Documents
Classification 
model

Page 5:
Document classification
Type
Focus
Example
Topic
Subject matter
Sports vs Technology
Sentiment/opinion
Emotion (current state)
Negative vs Positive
Intent
Action (future state)
Order vs Inquiry
Topic: บุพเพสันนิวาส
Sentiment: positive
Action: watch
Action: order_hawaiian
https://www.theatlantic.com/technology/archive/2011/03/does-anne-hathaway-news-drive-berkshire-hathaways-stock/72661/

Page 6:
Other classification applications
• Spam filtering
• Authorship id
• Auto tagging (information retrieval)
• Trend analysis

Page 7:
Text classification definition
• Input
• Set of documents: D = {d1, d2, d3, …, dM}
• Each document is composed of words
• d1 = [w11, w12, … w1N]
• Set of classes: C = {c1, c2, c3, .., cJ}
• Output
• The predicted class c from the set C

Page 8:
Rule-based classification
• Rules based on phrases or other features
• Wongnai Rating
• แมลงสาบ-> 2 ดาว
• อร่อย-> 4 ดาว
• ไม่อร่อย-> 2 ดาว
• …
• What if the phrase is ไม่ค่อยอร่อย
• New rule
• อร่อยโดยที่ไม่มีค าว่าไม่อยู่แถวๆนั้น-> 4 ดาว
• What if the phrase is ไม่ถูกแต่อร่อย
• This can yield very good results but…
• Building and maintaining rules is expensive!
• Or keep a word list of positive and negative words

Page 9:
Supervised text classification definition
• Input
• Set of documents: D = {d1, d2, d3, …, dM}
• And labels Y = {y1,y2,y3,…,yM}
• Each document is composed of words
• d1 = [w11, w12, … w1N]
• Set of classes: C = {c1, c2, c3, .., cJ}
• Output
• A classifier H: d -> c

Page 10:
What classifier?
• Any classifier you like
• k-NN
• Naïve Bayes
• Logistic regression
• SVM
• Neural networks
We use this kind of classifier 
before in the previous homework

Page 11:
Outline
• Naïve Bayes
• Neural methods
• Topic Models
• Latent topic models (LDA)

Page 12:
Bag of words representation
= 3
H

Page 13:
Bag of words representation
= 3
H
Bag of words only care about the presence of words or features but 
ignore word position and context
= 3
H
ชอบ, อร่อย, ไม่, ไม่, กลมกล่อม, ทานง่าย

Page 14:
Bag of words representation
= 3
H
Bag of words only care about the presence of words or features but 
ignore word position and context
= 3
H
Word
Count
ชอบ
1
อร่อย
1
ไม่
2
กลมกล่อม
1
ทานง่าย
1

Page 15:
Bag of words for classification intuition
Test review
1star
5star
3star
แมงสาบ
สกปรก
แย่
ใช้ได้
ถูก
อร่อย
คาดฝัน
เหาะ
เชฟ
อร่อย
ยอด
แมงสาบ
ใช้ได้
ถูก
อร่อย

Page 16:
Bag of words for classification intuition

Page 17:
Bayes’ Rule for classification
• A simple classification model
• Given document d, find the class c
• Argmax P(c|d)   
=Argmax P(d|c) P(c)
P(d)
=Argmax P(d|c) P(c)
=Argmax P(x1, x2, …, xn | c) P(c)
c
c
c
Bayes’ Rule
P(d) is constant wrt to c
The document is represented by features
x1, x2, …, xn
c

Page 18:
Bayes’ Rule for classification
• A simple classification model
• Given document d, find the class c
• Argmax P(c|d)   
=Argmax P(d|c) P(c)
P(d)
=Argmax P(d|c) P(c)
=Argmax P(x1, x2, …, xn | c) P(c)
c
c
c
c
P(x1, x2, …, xn | c) requires O(|X|n |c|) 
parameters. Cannot train
likelihood
prior

Page 19:
Bag of words assumption
• P(x1, x2, …, xn | c) P(c) = P(x1|c) P(x2|c) P(x3|c) …P(xn|c) P(c)
• Conditional independence
C
X1
X3
Xn
X2
O(n|X| |c|) parameters
Naïve Bayes
Naïve – conditional ind
Bayes – Bayes rule for classification
|X| |c|
List of possible X
List of possible c

Page 20:
Bags of words and NB
Probability of drawing words from the bag
Word
Distribution
(class=1)
ชอบ
0.1
อร่อย
0.1
ไม่
0.5
กลมกล่อม
0.2
ทานง่าย
0.1
P(“ไม่อร่อยไม่ชอบ”| c = 1) 
= P(ไม่| c= 1) P( อร่อย| c= 1) P(ไม่| c= 1) P( ชอบ| c= 1)
= 0.5 * 0.1 * 0.5*  0.1

Page 21:
Learning the Naïve Bayes model
• As usual counts
• P(x|c)
• P(x = “ยอด”| c=5) = count(x =“ยอด”, c = 5)
count( c = 5)
• P(c)
• P(c = 5) = count (c = 5)
count (all reviews)
• This is the Maximum Likelihood Estimate (MLE)
|X| |c|
List of possible counts
List of classes
xn is a feature that counts word occurrence
x1 how many times ยอดappear

Page 22:
Learning the Naïve Bayes model
• What if we encounter zeroes in our table
• P(x|c)
• P(x = “ยอด”| c=5) = count(x =“ยอด”, c = 5)
count( c = 5)
• P(‘ร้านนี้ราดหน้ายอดผักไม่อร่อย’|c = 2) 
= P(x= “ร้าน”|c=2)*P(x= “นี้”|c=2)… P(x= “ยอด”|c=2) *...
= 0
Zero probability regardless of other words
What about unknown words (OOV)? Drop them (no calculation)
|X| |c|
List of possible counts
List of classes
xn is a feature that counts word occurrence
x1 how many times ยอดappear
One solution: add-1 smoothing  (a hyperparameter to tune)

Page 23:
Naives Bayes
• Can use other features beside word counts
• Feature engineering – restaurant name, location, price range, 
reviewer id, date of review
• Tedious but very powerful
• Features > 10000
• Pros: very fast, very small model
• Need to remove stop words
• Robust especially for small training data (hand-crafted 
rules)
• A good fast baseline. Always try Naive Bayes or logistic 
regression in model search.
• Even with lots of data and rich features, Naives Bayes 
can be very competitive and fast!

Page 24:
Naive Bayes vs Logistic regression
Generative vs Discriminative modeling
Given data x, predict y
• Naïve Bayes are generative models
• Logistic regression are discriminative models
• Note P(y|x) can be any function that outputs y given x (a neural 
network)
• Logistic regression and Naive Bayes are linear models 
(linear decision boundary)
• They are quite interchangeable.

Page 25:
Naive Bayes vs Logistic regression
Generative vs Discriminative modeling
When training data is small, Naive Bayes performs better. When 
training data is large, Logistic regression performs better.
dashed line - logistic regression
solid line - naive bayes
http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf

Page 26:
Fast and good classification using n-
grams
• Features: n-grams (bag of phrases)
• Model: logistic regression
• Very competitive results
Bag of Tricks for Efficient Text Classification
https://arxiv.org/pdf/1607.01759.pdf

Page 27:
Fast and good classification using n-
grams
• Features: n-grams (bag of phrases)
• Model: logistic regression
• Very competitive results
Bag of Tricks for Efficient Text Classification
https://arxiv.org/pdf/1607.01759.pdf

Page 28:
Tag prediction
Bag of Tricks for Efficient Text Classification
https://arxiv.org/pdf/1607.01759.pdf

Page 29:
Naïve Bayes tricks for text classification
•Domain specific features
• Count words after “not” as a different word
• I don’t go there. -> I don’t go_not there_not
• Upweighting: double counting words at important 
locations
• Words in titles
• First sentence of each paragraph
• Sentences that contain title words
Context-Sensitive Learning Methods for 
Text Categorization
https://www.researchgate.net/publication
/2478208_Context-
Sensitive_Learning_Methods_for_Text_
Categorization
Information retrieval using
location and category information
https://www.jstage.jst.go.jp/article/jnlp1
994/7/2/7_2_141/_article
Automatic text categorization using the 
importance of sentences
https://dl.acm.org/citation.cfm?id=1072331

Page 30:
Different variants of Naive Bayes
•
What we described was Multinomial Naive Bayes
•
Takes in word counts (Term frequency - TF)
•
Assumes length independent of class, TF follows Poisson dist
•
Can also take in a binary version of word counts
•
There’s also Multi-variate Bernoulli Naive Bayes
•
Takes in binary version of word counts
•
Slightly different assumptions, also consider probability when 
count = 0
•
SVM-NB (SVM with NB as features)
•
etc.
Additional readings
“Spam Filtering with Naive Bayes – Which Naive Bayes?” 
http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf
“Baselines and Bigrams: Simple, Good Sentiment and Topic Classification”
http://www.aclweb.org/anthology/P12-2018
https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline

Page 31:
Neural methods
-
Sentence/document embedding
-
Deep Averaging Networks, USE, sentence embeddings

Page 32:
Deep Averaging Networks (DAN)
sums words
in the sentence
https://www.aclweb.org/anthology/P15-1162/
2015

Page 33:
Universal Sentence Encoder (USE)
A model focusing on sentence representation
Use sentencepiece tokenization
Pre-trained then used anywhere
Based on (1) DAN (lite version) or (2) Transformer
Official implementation with pretrained weights
https://tfhub.dev/google/collections/universal-sentence-encoder/1
https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html

Page 34:
Final words on text classification
Current state-of-the-art are about learning representations
Unsupervised pre-training of text (Word2Vec, BERT, 
ULMFit, simCSE, ConGen, etc)
Trend
Word representation (non-contextualized) 
-> Sentence representation (contextualized)

Page 35:
Zero/few shot classification
•
With good sentence/document representations one can 
use it to perform zero or few shot classification
Model
Test
Test B
Embedding A
Embedding B
Distance comparison
Same or different?
Model
Class A data 1
Class A data 2
Class B data 1
Class B data 2
Class B data 3
Embedding space

Page 36:
Classification benchmarks
•
https://github.com/mrpeerat/Thai-Sentence-Vector-
Benchmark

Page 37:
Classification benchmarks
•
https://github.com/mrpeerat/Thai-Sentence-Vector-
Benchmark

Page 38:
Search/retrieval benchmarks
Types of search
Typical engine: LUCENE, FAISS 
https://github.com/facebookresearch/faiss
Typical engine: LUCENE

Page 39:
MPNET
•
A pretrained transformer model
•
Pretrained using Masked Langauge Modeling (MLM) and 
Permuted Language Modeling (PLM)
•
PLM is similar to decoder only but trained on permuted versions 
of the sentences.
https://arxiv.org/pdf/2004.09297

Page 40:
MPNET
The Sentence Transformer authors 
then finetune MPNET using 
paraphrase datasets to do additional 
contrastive learning.
PLM
MLM

Page 41:
Classification Benchmarks 
https://github.com/PyThaiNLP/classification-benchmarks
(delisted from pythaiNLP due to license concerns)
Chula still has Educational/research license

Page 42:
ConGEN: Unsupervised Control and Generalization 
Distillation For Sentence Representation
•
Want a smaller model for sentence representation
•
But training a small model is hard
https://aclanthology.org/2022.findings-emnlp.483/

Page 43:
Knowledge Distillation
•
Student model learns from a teacher model
•
Training a small model is hard, but training a larger sophisticate 
model is easy.
•
Have the teacher teach the smaller model.
https://blog.roboflow.com/what-is-knowledge-distillation/

Page 44:
Distillation use cases
•
Distillation for smaller model
Teacher
Student

Page 45:
Distillation use cases
•
Distillation for model improvement (self-distillation)
•
Keep re-initializing the teacher model
Teacher
Student
Teacher2
Student2

Page 46:
Instance Queue
•
In contrastive learning, a large mini-batch is preferred.
•
Every mini-batch new samples need to be computed <-
compute bounded.
•
We can keep some of the old embeddings in a queue

Page 47:
ConGen

Page 48:
Backtranslate is a very good 
augmentation method

Page 49:
Outline
• Naïve Bayes
• Neural methods
• Topic Models
• Latent topic models (LDA)

Page 50:
Text classification and language 
modeling
• P(x|c)
• P(x = ยอด| c=5) = count(x = ยอด, c = 5)
count( c = 5)
• P(c)
• P(c = 5) = count (c = 5)
count (all reviews)
|X| |c|
List of words
List of classes
This looks like… n-grams, but instead of conditioning on
the past, we condition on the topic –
bag of words model for topic modeling (unigram with topic)
x1 is a feature that looks at the first word

Page 51:
Language modeling view
• Which class is this review
• P(w|c)
อร่อยแต่ไม่ถูก
Class= 1
อร่อย0.01
แต่0.4
ไม่0.4
ถูก0.03
…
Class= 5
อร่อย0.4
แต่0.05
ไม่0.25
ถูก0.15
…
P(s|c =1) = 0.01*0.4*0.4*0.03 = 0.000048
P(s|c = 5) = 0.4*0.05*0.25*0.15 = 0.00075

Page 52:
Topic modeling
• Sometimes you want to model the topic of a document
Class= 
บรรยากาศ
อร่อย0.01
แต่0.4
ไม่0.4
ถูก0.03
…
Class= อาหาร
อร่อย0.4
แต่0.05
ไม่0.25
ถูก0.15
…
P(s|c=บรรยากาศ) = ?
P(s|c=อาหาร) = ?
อาหารที่นี่ไม่ค่อยอร่อยแต่ขนม
ใช้ได้เลยถ้าว่างอาจจะกลับมากิน
อีกแนะน าให้สั่งเค้กใบเตย
ด้านบรรยากาศมีเสียงก่อสร้างมา
จากตึกข้างๆแต่นอกนั้นตกแต่ง
โอเคแต่ยังขาดอะไรไปหลายๆ
อย่าง

Page 53:
Naïve Bayes Topic modeling 
issues
• Most document have multiple topics (multi-label). 
• Our model assumes 1 document 1 topic. 
• Solution: Let a document be a mixture of topics (language 
model interpolation). Each word has its own topic, z.
• P(w) = P(w is topic A) P(w | topicA) + P(w is topic B) P(w | topicB)
• P(w is topic A) + P(w is topic B) = 1

Page 54:
Naïve Bayes Topic modeling 
issues
• Most document have multiple topics. Our model assumes 
1 document 1 topic.
• Let a document be a mixture of topics (language model 
interpolation). Each word has its own topic, z.
• P(w) = P(z = A) P(w | z = A) + P(z = B) P(w | z = B)
• P(z = A) + P(z = B) = 1,       θ = P(z = A)
z1
w1
w3
wn
w2
z2
z3
zn
θ
Parameter that governs how 
likely each topic is for the 
document
βA
βB
Parameter that 
governs how 
likely a word is 
for a topic
βA = P(w| z = A)

Page 55:
Graphical model and generation
zn
Topic A
zn-1
Topic B
zn-2
Topic B
βA
βB
Cat = 0.5
Dog = 0.5
sad = 0.7
bad = 0.3
θ
A = 0.3
B = 0.7
wn
wn-1
wn-2
Dog
sad
bad
How likely a sentence is likely to be generated follows this generation process
P(sad,bad,Dog,B,B,A) = P(B)P(B)P(A)P(sad|B)P(bad|B)P(Dog|A)
Note
P(sad,bad,Dog) = P(sad,bad,Dog,A,A,A) + P(sad,bad,Dog,A,A,B) 
P(sad,bad,Dog,A,B,A) + P(sad,bad,Dog,A,B,B) +...
Total probability theorem
which makes 
P(w) = P(z = A) P(w | z = A) + 
P(z = B) P(w | z = B)

Page 56:
pLSA (probabilistic Latent 
Semantic Analysis) model
zn
Topic A
zn-1
Topic B
zn-2
Topic B
βA
βB
Cat = 0.5
Dog = 0.5
sad = 0.7
bad = 0.3
θ
A = 0.8
B = 0.2
wn
wn-1
wn-2
Dog
sad
Cat
pLSA models a document with their own topic mixture θ

Page 57:
pLSA
• pLSA automatically clusters words into topic unigrams
• Requires user to specify number of topics
• Automatically learn document representation based on 
the learned topics
• DocA = [0.7 0.3]  DocB = [0.2 0.8] DocC = [0.5 0.5]
• Overfits easily to data outside of the training set
• Nothing that ties all document together
• A document from a document collection should be have topic 
distributions that are similar
• Solution: LDA (Latent Dirichlet Allocation)

Page 58:
pLSA
β
θ
w
z
n
k
d
P(บรรยากาศ) = 0.3
P(อาหาร) = 0.7
P(บรรยากาศ) = 0.3
P(อาหาร) = 0.7
P(บรรยากาศ) = 0.3
P(อาหาร) = 0.7

Page 59:
LDA
β
θ
w
z
n
k
d
α
η
Governs how topics should be in general
Governs how documents should be 
in general

Page 60:
Introduction to Probabilistic Topic Models, Blei 2011
http://menome.com/wp/wp-content/uploads/2014/12/Blei2011.pdf

Page 61:
LDA
• Automatically learns topics, and the word distribution of 
each topic
• Just give a bunch of documents!
• Each document is given a mixture of topics
• Dirichlet prior prefers sparse topics – each document only have probability 
in few topics – easy for interpretability
• Requires user to pick number of topic
• Requires user to make sense of the learned topics
• For more information on how to help visualize/evaluate unsupervised 
topic models
• https://youtu.be/UkmIljRIG_M

Page 62:
Unsupervised topic modeling for real 
estate
Can we learn real estate characteristics from unstructured 
data?
คอนโดหรูสไตล์อังกฤษแห่งแรกในเขา
ใหญ่ที่ติดถ.ธนะรัชต์มากที่สุด1 
ห้องนอน1 ห้องน ้า1 ห้องนั่งเล่น
พร้อมห้องครัวแยกเป็นสัดส่วน
คอนโดหรูสไตล์อังกฤษแห่งแรกในเขา
ใหญ่ที่ติดถ.ธนะรัชต์มากที่สุด1 
ห้องนอน1 ห้องน ้า1 ห้องนั่งเล่น
พร้อมห้องครัวแยกเป็นสัดส่วน
Just give it a bunch of descriptions

Page 63:
LDA Examples
Pretty examples of learned topics here    

Page 64:
สีม่วง= Topic 9
พื้นที่บริเวณเขาใหญ่
จ.นครราชสีมา

Page 65:
LDA Examples
สีน ้าเงินเข้ม= โครงการที่มีTopic 40 อยู่มาก(หรู,ระดับ)
สีเขียว= โครงการที่มีTopic 17 (โครงการทั่วไป)

Page 66:
Time and advertising trends

Page 67:
Advertisement niche of each 
developer

Page 68:
More ideas
https://journals.sagepub.com/doi/full/10.1177/0022242919873106

Page 69:
Tweet analysis
https://www.facebook.com/teattapol/posts/3337686199651109?_rdc=1&_rdr
https://stacks.stanford.edu/file/druid:ym245nv3149/twitter-TH-202009.pdf
Visualized using pyLDAvis https://github.com/bmabey/pyLDAvis

Page 70:
LDA with deep learning
• LDA was develop on discrete inputs (words)
• Modified to work with dense representation (word vectors)
• “Gaussian LDA for Topic Models with Word Embeddings”
• http://www.aclweb.org/anthology/P15-1077
• Modified network structure and loss function to include 
LDA traits
• LDA2vec 
https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/

Page 71:
Clustering on document 
embeddings?
•
Top2Vec proposes a simple method to cluster document 
embeddings
•
Use UMAP+HDBSCAN to identify number of clusters and the cluster
•
Represents cluster using most representative word in the cluster
https://aclanthology.org/2024.findings-emnlp.790.pdf
https://github.com/ddangelov/Top2Vec
https://arxiv.org/pdf/2008.09470

Page 72:
UMAP
•
UMAP is a data visualization/dimensionality reduction 
technique that focuses on preserving local and global 
structure of high dimensional data
https://www.nature.com/articles/s41592-024-02301-x

Page 73:
HDBSCAN
•
DBSCAN – a common technique for clustering. Finds a 
cluster by looking for a group of points within a certain 
distance.

Page 74:
HDBSCAN
•
HDBSCAN – does DBSCAN over different epsilon 
making the method more robust to scaling.
https://www.dailydoseofds.com/hdbscan-the-supercharged-version-of-dbscan-an-algorithmic-deep-dive/

Page 75:
Summary
• Text classification task
• Bag of words model
• Naïve Bayes
• Neural based
• Text clustering
• LDA
• Top2vec

