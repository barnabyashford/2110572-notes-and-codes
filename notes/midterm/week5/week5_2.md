Page 1:
Text Representation
Representation learning

Page 2:
What is an embedding?
•
“Latent space” or “embedding space” refers to a low-
dimensional representation of high-dimensional data
•
In neural network, the mapping from original data to the 
embedding space is often linear.
•
Ex of linear mapping/projection: PCA
•
Mapping of these embeddings are one of the key tricks 
in deep learning today
High dimension
Low dimension

Page 3:
Embeddings
•
Can be trained by supervised or self-supervised 
techniques
https://twitter.com/ylecun/status/1226838002787344391?lang=en

Page 4:
Outline
•
Contrastive learning
•
Sentence embeddings
•
MUSE
•
SimCSE
•
BGE
•
CLIP

Page 5:
Contrastive learning
•
An important technique for self-supervised training is 
contrastive learning
•
Similar things should have similar embeddings
•
Different things should have different embeddings
•
Example: negative sampling loss in word2vec
Negative samples
(negative, -1)
Context word
(positive, +1)

Page 6:
Types of contrastive learning
•
Triplet loss
•
InfoNCE loss

Page 7:
Triplet loss
•
Triplet loss considers an anchor, a positive, and a 
negative
•
Requires mining of hard negative samples
https://arxiv.org/abs/1503.03832
Example triplet loss
Take positive only max(0,x)
margin

Page 8:
Dealing with minibatches
•
Since we train in minibatches, most modern losses pair 
positive and negative samples within a minibatch for 
more efficient computation
•
Compute all pairwise distance within the minibatch
https://arxiv.org/pdf/1511.06452.pdf

Page 9:
NCE (Noise constrastive 
estimation) loss
•
Maximize training data probability while reducing noise 
probability.
•
Learn in a constrastive way to reduce overhead for 
normalization
•
Max LogP(data) – Log P(noise or negative samples)
•
Ex: used to train word embeddings such as W2V, too many 
classes in the softmax output
http://proceedings.mlr.press/v9/gutmann10a.html

Page 10:
InfoNCE
•
Similar to NCE but just for categorical cross entropy 
(instead of binary cross entropy) 
https://arxiv.org/pdf/1807.03748.pdf
•
Effectively maximize mutual information between c and 
positive x
•
f( ) can be any function that describes similarity
•
Can be extended to have multiple positive examples in 
a batch (soft nearest neighbor loss) 
https://arxiv.org/abs/1902.01889
z is encoded x

Page 11:
Soft nearest neighbor loss
•
Multiple positive and negative
•
Adds temperature (either hyperparameter, or learned)
•
Weights the gradient size, helps model learn form hard 
negatives
https://arxiv.org/pdf/1902.01889.pdf

Page 12:
Contrastive summary
•
The most common form you will see 
for contrastive learning is
•
People often refer to this as contrastive loss, InfoNCE 
loss, normalized temperature scaled CE loss, …
Encourage embeddings to spread uniformly
in the hypersphere
Encourage similar things to align
https://arxiv.org/abs/2005.10242 https://arxiv.org/abs/2011.02803 https://arxiv.org/abs/2002.05709

Page 13:
Key details to contrastive loss 
works
•
Large batch
•
Hard/semi-hard negative mining
•
Augmentation on the anchor and postive
•
Other tricks includes - adding classification/supervised 
loss (CE/softmax loss)

Page 14:
Outline
•
Contrastive learning
•
Sentence embeddings
•
MUSE
•
SimCSE
•
BGE
•
CLIP

Page 15:
Sentence representation
•
How would we create a sentence embedding?
•
Compositionality from words/tokens!
•
Sum, max
•
Recurrence
•
Attention

Page 16:
MUSE

Page 17:
Deep Averaging Networks (DAN)
sums words
in the sentence
https://www.aclweb.org/anthology/P15-1162/
2015

Page 18:
Universal Sentence Encoder (USE)
A model focusing on sentence representation
Use sentencepiece tokenization
Pre-trained then used anywhere
Based on (1) DAN (lite version) or (2) Transformer
Official implementation with pretrained weights
https://tfhub.dev/google/collections/universal-sentence-encoder/1
https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html
https://www.kaggle.com/models/google/universal-sentence-encoder

Page 19:
Pretraining USE
Training done using multi-task
1) Skip-thought
2) Response prediction
3) Natural language inference (NLI)
Picture credit: https://amitness.com/2020/06/universal-sentence-encoder/

Page 20:
Skip-thought task
Similar to skip-gram, use the middle to predict context
Unsupervised
Picture credit: https://amitness.com/2020/06/universal-sentence-encoder/
shared encoder 
(DAN/Transformer)

Page 21:
Response prediction
Match questions and answers in internet forum (scraped)
Supervised (free labels)
To map answer to
the same space
as question
similar sentence gives similar response
Picture credit: https://amitness.com/2020/06/universal-sentence-encoder/

Page 22:
Natural Language Inference
Predict relationship between sentence
Supervised

Page 23:
Multilingual USE
Can train to map multiple languages to the same 
presentation.
Can handle code switching, has Thai!
https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html

Page 24:
Download-ables
Pytorch conversion
https://huggingface.co/dayyass/universal-sentence-encoder-multilingual-large-3-pytorch
Trained with Masked Language Model loss (BERT-like)

Page 25:
BERT-Based 
embeddings

Page 26:
Sentence representation with 
BERT
With BERT, we found that MLM training create good sentence 
representation too!
We can use NSP embedding or pool the token embeddings to create a 
sentence representation 

Page 27:
SBERT
Language Understanding
Semantic Understanding
https://arxiv.org/abs/1908.10084 2019

Page 28:
Sentence level contrastive learning
•
We can learn better sentence representation with some 
additional supervised (or unsupervised) sentence level 
contrastive learning
Augment1, Augment2
Augment1, Augment2
Not augment2
ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer (2021)

Page 29:
ConSERT augmentations

Page 30:
ConSERT alignment

Page 31:
SimCSE
•
Use simple dropout in the model to create different 
versions of the same sentence

Page 32:
SimCSE
Other augmentations technique
Rather than contrastive, predict next 
sentence, 1 of 3 next sentences

Page 33:
What’s other use of embeddings?
•
Retrieval and recommendation
https://developer.nvidia.com/merlin

Page 34:
What’s other use of embeddings?
•
Learn joint embeddings between different modalities
Joint interaction model
Two tower model
https://nvidia-merlin.github.io/models/main/models_overview.html#deep-learning-recommender-model

Page 35:
BGE-M3
•
A retreival model (Query -> Document)
•
Built on top of BGE (Chinese embedding model)
•
BGE: Masked LM finetuned with contrastive and task specific 
losses
•
BGE-M3 (multilingual, multifunction, multigranularity)
•
Trained by multiple losses terms that utilizes different 
parts of the model embeddings
https://arxiv.org/abs/2402.03216

Page 36:
CLIP
•
Contrastive learning on image-text pairs
https://arxiv.org/abs/2103.00020

Page 37:
Outline
•
Contrastive learning
•
Sentence embeddings
•
MUSE
•
SimCSE
•
BGE
•
CLIP

