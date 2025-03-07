Page 1:
Deep tokenization
1

Page 2:
Tokenization
•
Recent tokenization techniques are based on deep 
learning models
•
Better to handle out-of-vocabulary (OOV), misspellings, etc.
2

Page 3:
NEURAL NETWORKS
Deep learning = Deep neural networks = 
neural networks
3

Page 4:
Why neural networks
4
https://arxiv.org/abs/2006.11239
Digits recognition
Speech-to-text
Object classification
QA
QA with no answer
Multitask NLP
Deep learning starts

Page 5:
5
1) AI researchers have often tried to build 
knowledge into their agents, 2) this always 
helps in the short term, and is personally 
satisfying to the researcher, but 3) in the 
long run it plateaus and even inhibits 
further progress, and 4) breakthrough 
progress eventually arrives by an 
opposing approach based on scaling 
computation by search and learning.
- Richard Sutton
The Bitter Lesson
https://arxiv.org/pdf/2402.08797
Scaling matters

Page 6:
Deep learning in NLP
Easy task modest gains
Harder task larger gains
6
Traditional ML
Deep learning
Wisesight Sentiment (th)
72%
76% (WangchanBERTa)
71% (Phayathaibert unsup)    
(60% QWEN2-72B zeroshot)
Topic classification (th)
67%
70%
PoS (th)
96%
97%
Traditional ML
Deep learning
QA
51%*
90%
Creating image from text
???
very good
wangchanbert
https://www.aclweb.org/anthology/
D16-1264/
https://openai.com/blog/dall-e/

Page 7:
Neural networks
• Fully connected networks
• Neuron
• Non-linearity
• Softmax layer
• Dropout
• Batchnorm
• CNN, RNN, LSTM, GRU
• If you are unaware of these, please watch last year’s 
Pattern lecture 7 and 7.5
7

Page 8:
Non-linearity
• The Non-linearity is important in order to stack neurons
• If F is linear, a multi layered network can be collapsed as a single 
layer (by just multiplying weights together)
• Sigmoid or logistic function
• tanh
• Rectified Linear Unit (ReLU)
• LeakyReLU, ELU, PreLU
• Sigmoid Linear Units (SiLU)
• Swish, Mish, GELU, SwiGLU
8

Page 9:
SiLU, GELU
9
(or Swish – Swish paper comes after 
SiLU but is more popular)

Page 10:
GLU (Gated Linear Unit)
•
A gated dense layer
GLU(x) = (Wx+b)*sigmoid(Vx+c)
10
https://jcarlosroldan.com/post/348/what-is-swiglu

Page 11:
SwiGLU
•
GLU with a Swish/SiLU gating function
•
Many LLMs use this. Example: Llama
GLU(x) = (Wx+b)*sigmoid(Vx+c)
Swish(x) = x*sigmoid(Bx)
SwiGLU(x) = (Wx+b)*Swish(Vx+c)
11
https://jcarlosroldan.com/post/348/what-is-swiglu

Page 12:
Batch normalization
• Recent technique for (implicit) regularization
• Normalize every mini-batch at various batch norm layers 
to standard Gaussian (different from global normalization 
of the inputs)
• Place batch norm layers before non-linearities
• Faster training and better generalizations
https://arxiv.org/abs/1502.03167
For each mini-batch that goes through 
batch norm
1.
Normalize by the mean and variance 
of the mini-batch for each dimension
2.
Shift and scale by learnable 
parameters
Replaces dropout in some networks
12

Page 13:
Other normalizations
•
Other normalizations are out there
https://theaisummer.com/normalization/
Weight norm
NLP and CV layer norm are not the same (In NLP, layer norm is applied 
separately for each element in the sequence)
https://proceedings.mlr.press/v119/shen20e/shen20e.pdf
Layer norm in 
transformers
13

Page 14:
What toolkit
Tradeoff between customizability and ease of use
GPU
CPU
CUDA
BLAS
cuDNN
cuBLAS
Tensor 
flow
CNTK
Keras
Lines of code
Customizability
Caffe
Torch
MXNet
lightning
14

Page 15:
Pytorch steps
•
Setting up dataloader
•
Gives minibatch
•
Define a network
•
Init weights
•
Define computation graph
•
Setup optimization method
•
Pick LR scheduler
•
Pick optimizer
•
Training loop
•
Forward (compute Loss)
•
Backward (compute gradient and apply gradient)
•
Let’s demo
15

Page 16:
Lightning
•
Setting up dataloader
•
Gives minibatch
•
Define a network
•
Init weights
•
Define computation graph
•
Setup optimization method
•
Pick LR scheduler
•
Pick optimizer
•
Training loop
•
Forward (compute Loss)
•
Backward (compute gradient and apply gradient)
•
Let’s demo
Pytorch 
Lightning 
helps with 
this
and much 
more
16

Page 17:
Lab/HW
•
Word segmentation using pytorch
•
Given a letter with 10 letters before and after, determine 
whether it’s a start of a word
17
ทรงอย่ย่างแบดแซดอย่ย่างบdบ

Page 18:
Word segmentation with fully 
connected networks
5
3
4
2
4
Logistic function
1 = word beginning, 0 = word middle 
ก
า
ร
ร
ู 
18

Page 19:
Embeddings
• A way to encode information to a lower dimensional 
space
• We can learn about this lower dimensional space through data
19
CAT
DOG
CAP
[67, 65, 84]
[67, 65, 80]
[68, 79, 71]
PIG
[80, 73, 71]

Page 20:
One hot encoding
• Categorical representation is usually represented by one 
hot encoding
• Categorical representations examples: 
• Words in a vocabulary, characters in Thai language
Apple -> 1 -> [1, 0, 0, 0, …]
Bird -> 2 -> [0, 1, 0, 0, …]
Cat -> 3 -> [0, 0, 1, 0, …]
• Sparse representation
• Spare means most dimension are zero
20

Page 21:
One hot encoding
• Sparse – but lots of dimension
• Curse of dimensionality
• Does not represent meaning.
Apple -> 1 -> [1, 0, 0, 0, …]
Bird -> 2 -> [0, 1, 0, 0, …]
Cat -> 3 -> [0, 0, 1, 0, …]
|Apple – Bird| = |Bird – Cat|
21

Page 22:
Getting meaning into the feature 
vectors
• You can add back meanings by hand-crafted rules
• Old-school NLP is all about feature engineering
• Word segmentation example:
• Cluster Numbers
• Cluster letters
• Concatenate them 
• 1 = [0 0 0 0 1 0 0 0, 1, 0]
• ก= [0 0 0 1 0 0 0 0, 0, 1]
• า= [1 0 0 0 0 0 0 0, 0, 2]
• Which rules to use?
• Try as many as you can think of, and do feature selection or use 
models that can do feature selection
22

Page 23:
Dense representation
• We can encode sparse representation into a lower 
dimensional space
• F: RN -> RM, where N>M
• We can do this by using an embedding layer
• This is just a (learnable) lookup table!
Apple -> 1 -> [1, 0, 0, 0, …] -> [2.3, 1.2]
Bird -> 2 -> [0, 1, 0, 0, …] -> [-1.0, 2.4]
Cat -> 3 -> [0, 0, 1, 0, …] -> [-3.0, 4.0]
23

Page 24:
Word segmentation with fully 
connected networks
5
3
4
2
4
Logistic function
1 = word beginning, 0 = word middle 
ก
า
ร
ร
ู 
24

Page 25:
Adding embedding layer
5
3
4
2
4
E
E
E
E
E
[1, -1] [3, -2]
[5.3, -2.1]
[5.3, -3.1] [3, -2]
Embedding layer
shares the same 
weights
Parameter sharing!
More on embeddings
in the next two 
lectures!
25

Page 26:
Embedding and meaning 
(semantics)
• Meaning is inferred from the task
• Embedding of 32 dimensions -> t-SNE into 2 dimension 
for visualization
• Automatically!
26

Page 27:

Page 28:

Page 29:

Page 30:
Debugging guide
•
https://uvadlc-
notebooks.readthedocs.io/en/latest/tutorial_notebooks/g
uide3/Debugging_PyTorch.html
has list of common errors and best practices
•
http://karpathy.github.io/2019/04/25/recipe/
has guide for end-to-end model building (start simple 
and go more advance)
30

Page 31:
Back to tokenization…
BEST 2009 : Thai word segmentation software contest
http://ieeexplore.ieee.org/document/5340941/
https://sertiscorp.com/thai-word-segmentation-with-bi-directional_rnn/
31

Page 32:
Best 2009 standard
•
Based on “Minimal Integrity Unit”
•
A compound that its meaning is not so different from its 
part should be segmented.
32
https://web.archive.org/web/20240608223632/https://pioneer.chula.ac.th/~awirote/ling/snlp2007-wirote.pdf
Segmented into single words
Segmented into multiple words

Page 33:
33
https://github.com/rkcosmos/deepcut

Page 34:
34
https://www.sertiscorp.com/november-
20-2017

Page 35:
35
https://github.com/PyThaiNLP/attacut

Page 36:
36
https://github.com/PyThaiNLP/attacut

Page 37:
ผมเห็นคนวงการนี้เมื่อ20-30 ปีที่แล้วท าเรื่องตัดค าวงการนี้มันไม่ไปไหนเลยใช่ไหมเนี่ย
มิตรสหายBusiness Development ท่านนึง
37

Page 38:
Words of caution
Statistical tokenizers fail on mismatched data
A tokenizer trained on social text might not be able to 
cut simple words like
มะม่วงมะละกอ
Statistical tokenizers fails unpredictably
หม กรอบ=> |หม |กรอบ|
ข้าวผัดคะน้าหม กรอบหนึ่งจาน=> |ข้าวผัด|คะน้า|หม ก|รอบ|หนึ่ง|จาน|
Might need rule-based to override (Deepcut has this)
For speed, maximal matching (newmm) is reliable.
-
drawbacks?
38
WS160
TNHC
Deepcut
93.8
93.5
Attacut
93.5
80.8
https://www.aclweb.org/anthology/2020.emnlp-main.315/
https://github.com/mrpeerat/OSKut

Page 39:
Words of caution
Tokenization performance effects downstream task 
performance
Can be small (1%) or large (10%)
Specialized tokenizer can help your downstream task
Example: e-commerce search |ห |ฟัง| |ต่าง|ห |
39

Page 40:
Words of caution
Be careful of what tokenization you used to train the model.
If there’s a mismatch in training and testing tokenization, 
the results can be devastating. 
40
Training
Testing tokenization
Deepcut
Deepcut
Longest matching+ 
noise0.1
Longest matching+ 
noise0.4
Longest matching+ 
noise0.7
76.8
60.4
50.9
42
TrueVoice
Training
Testing
Manual
Manual
Longest matching+ 
noise0.1
Longest matching+ 
noise0.4
Longest matching+ 
noise0.7
52.1
48.2
38.1
32.7
Wisesight1000
https://ieeexplore.ieee.org/document/9836268

Page 41:
Another important note
•
In Thai, due to visualization magic, these are the same
• น+ ู า+ น้
• น+ น้+ ู า
• เ+ เ+ ก
• แ+ ก
• ก+ ก้+ ก้+ ก้+ ก้
• ก+ ก้
•
You might want to normalize these.
41

Page 42:
Tokenization - English
• Even English has tokenization issues!
• Space is usually not enough
• aren’t
• are + n’t
• aren’t
• arent
• aren t
• are + not
• San Francisco
• Usually includes the text normalization step
• This depends on application
• “aren’t” might be different from “are not” for sentiment analysis
42

Page 43:
End-to-end models
• Classical machine learning systems usually break the 
problem into smaller subtasks
• Self-driving:
• Image -> objects detection -> path finding -> steering
• Speech2speech translation: 
• Speech A -> text A -> text B -> Speech B
• End-to-end models use one large neural networks 
process the input and generate the desired output
• Image -> steering
• Speech A -> Speech B
43

Page 44:
End-to-end NLP?
44

Page 45:
Towards no tokenization
• Text classification using charCNN on Thai
Word-based
methods
A character-level convolutional neural network with
dynamic input length for Thai text categorization
http://ieeexplore.ieee.org/document/7886102/
45

Page 46:
Caveats of end-to-end models
• Requires lots of data for the specific task
• Hard to fix specific mistakes by the model
46

Page 47:
Things to consider when thinking 
about tokenization
Know your use cases
47
Word
Subword
Character
Large vocabulary (100k)
Medium vocabulary (20k)
Small vocabulary (100)
Can use simpler model
Moderate complexity in 
modeling
Needs a powerful model 
to learn long range 
influences
High OOVs
Few OOVs
No OOVs
Individual tokens are 
meaningful
Individual tokens might be 
meaningful
Individual tokens are not 
meaningful
Note: to handle multilingual data, some models use bytes as 
tokens
Large embedding lookup table

Page 48:
Conclusion
•
Tokenization is far from solved but don’t let this 
discourage you
•
No tokenization is perfect
•
Pick one that is suited for your task
•
Speed
•
Robustness to misspelling and unseen words
•
Consistency
•
Certain tools assume you are using a particular type of 
tokenization, check!
48

