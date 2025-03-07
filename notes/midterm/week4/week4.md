Page 1:
Token Classification
PoS and NER
HMM, CRF, and search

Page 2:
Token Classification
•
A broad term for classifying tokens: PoS, NER
This lecture
Tokenization
Word embeddings

Page 3:
Part-Of-Speech tagging
• Categorize words into similar 
grammatical properties (syntax)
• Examples: Nouns, Verbs, 
Adjectives
• Actual applications often use 
more granular PoS labels
• PoS tags are often
• Language specific
• Application/corpus specific
https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

Page 4:
Part-Of-Speech tagging
• Input
• They refuse to permit us to obtain the refuse permit.
• Output
• They/PRP refuse/VBP to/To permit/VB us/PRP to/TO 
obtain/VB the/DT refuse/NN permit/NN 
Example from MIT 6.864

Page 5:
PoS usage
• Word disambiguation
• Different word vectors for different PoS of the same words
• Helps other NLP tasks
• PoS provides additional information that helps other tasks
• Tokenization
• Name-Entity Recognition
• Identify group of words that refer to the same entity
• ลุงพลร้องเพลงคุกกี้เสี่ยงทาย
• [ลุงพล]/person ร้องเพลง[คุกกี้เสี่ยงทาย]/title
• Parsing Ex parsing name and address from a sentence
• Search Ex disambiguation in keywords
• Text-to-speech Ex Nice, France vs nice french toast.

Page 6:
Thai PoS standards
•
https://pythainlp.org/dev-docs/api/tag.html
•
LST20 and Orchid are most famous Thai datasets
•
There is also Universal POS tags (not used much for 
Thai)

Page 7:
Orchid PoS corpus
• Building A Thai Part-Of-Speech Tagged Corpus 
(ORCHID) (1999) 
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.
34.3496
• 47 tags

Page 8:

Page 9:
NER
• Name-Entity Recognition wants to extract all name 
entities in a sentence and classify into types
• Can tag groups of words into the same category
• ลุงพลร้องเพลงคุกกี้เสี่ยงทาย
• [ลุงพล]/person ร้องเพลง[คุกกี้เสี่ยงทาย]/title
Biomedical publication mining
IL-2 gene expression and NF-kappa B 
activation through CD28 requires 
reactive oxygen production by            
5-lipoxygenase.

Page 10:
NER in applications
• Classifying/tagging content
• Information retrieval
• Trends analysis
https://blog.paralleldots.com/product/applications-named-entity-recognition-api/

Page 11:
Labeling standard (IOB format)
•
Consider the following sentence and NE tags
ลุงพลป๋าเบิร์ดร้องเพลงชาติ
ไทย
•
The two names are merged into a single entity. To 
separate the two entities, we sometimes add B 
(beginning) and I (inside) to the tags
ลุงพลป๋าเบิร์ดร้องเพลงชาติ
ไทย
Name
Name
Name
Name
Other
Song
NameB
NameI
NameB
NameI
Other
SongB
Song
SongI
Song
SongI

Page 12:
Thai Nested NER
•
Thai NER has multiple levels/layers and can be nested
https://medium.com/airesearch-in-th/thai-n-ner-thai-nested-named-entity-recognition-1969f8fe91f0

Page 13:
Thai Nested NER
https://aclanthology.org/2022.findings-acl.116/
https://github.com/vistec-AI/Thai-NNER
1 model per layer
1 model all layer

Page 14:
Overview
• What is Token Classification?
• Traditional methods
• Sequence methods
• HMM
• CRF
• Viterbi and beam search
• Neural network methods

Page 15:
Sequence methods
• They refuse to permit us to obtain the refuse permit.
• They/PRP refuse/VBP to/To permit/VB us/PRP to/TO 
obtain/VB the/DT refuse/NN permit/NN 
• Determining the PoS tag depends on the decision of the 
words around it
• A sequence problem

Page 16:
Problem setup
• Sequence of words
• W:= {w1,w2,w3,…,wn}
• Sequence of tags
• T:= {t1,t2,t3,…tn}
• Given W predict T
• argmaxT P(T|W)
• Or
• argmaxT P(T,W)      =   argmaxT P(T,W)
P(W)
Discriminative Model
Generative Model
P(W) is constant does not affect the argmax
P(a,b) joint distribution
P(a|b) conditional distribution
P(a) marginal distribution
P(a) = Σb P(a,b)

Page 17:
Modeling P(T,W)
• P(T,W) = P(w1,w2,w3,…,wn,t1,t2,t3,…tn)
• Is there a problem with this?

Page 18:
Modeling P(T,W)
• P(T,W) = P(w1,w2,w3,…,wn,t1,t2,t3,…tn)
• Is there a problem with this? Curse of dimensionality
• Language modeling
• P(wt) requires N table values
• P(wt|wt-1) requires N2 table values
• P(wt|wt-1,wt-2) requires N3 table values
• Many values have 0 counts (needs many tricks)
• We can use Markov Assumptions
• Or more generally, we use independence assumptions (conditional 
independence) to simplify the distribution to model

Page 19:
Hidden Markov Model
w
w
w
w
t
t
t
t
Transition probabilities
Emission probabilities
Markov assumption
Current value only depends on 
the immediate pass
T are called the hidden states. Usually comes from a finite set of possibilities
Ex:  t        {Noun, Adv, Adj, Verb}
P(T,W) = P(t1)P(t2|t1)P(t3|t2)P(t4|t3)P(w1|t1)P(w2|t2)P(w3|t3)P(w4|t4)
Transition probabilities
Emission probabilities
Initial state prob
What does the Markov 
assumption imply?

Page 20:
Hidden Markov Model
• Defining HMM requires
• 1. Starting state probability p0 = [0.7 0.3]
• 2. Transition probability, Aij
• 3. Emission probably, Bik
• If emits discrete values
• Discrete HMM
• If emits continuous values
• Continuous HMM
Aij
To N
To NN
From N
0.6
0.4
From 
NN
0.5
0.5
Bik
I
eat
Chinese
State N
0.8
0.01
0.19
State NN
0.1
0.45
0.45
Question: What’s the probability of P([I eat chinese] , [N NN N]) ? 
= p0(N) * A12 * A21 * B11 * B22 * B13
= 0.7 * 0.4 * 0.5 * 0.8 * 0.45 * 0.19
N = Noun
NN = Not Noun
i,j – state (tag) index
k – emission (word) index

Page 21:
Hidden Markov Model
• How to estimate A and B?
• Counts!
P(A11) = Count(from N to N)
Count(N)
• Counts with interpolation to avoid 0 counts
• P(A11) = ρ Count(from N to N)   +     (1-ρ) Count (N)
Count(N)                             Count (words)
Aij
To N
To NN
From N
0.6
0.4
From 
NN
0.5
0.5
Bik
I
eat
Chinese
State N
0.8
0.01
0.19
State NN
0.1
0.45
0.45
bigram
unigram
ρ is interpolation weight

Page 22:
Hidden Markov Model
• How to estimate A and B?
• Counts!
P(B11) = Count(“I” , N)
Count(N)
Aij
To N
To NN
From N
0.6
0.4
From 
NN
0.5
0.5
Bik
I
eat
Chinese
State N
0.8
0.01
0.19
State NN
0.1
0.45
0.45

Page 23:
Decoding
• Recall we want to find the sequence of tags that 
maximizes the joint probability
• argmaxT P(T,W)
• How to do this?
• Brute force
• Find all possible sequence of T, calculate P(T,W) and compare
• Length N words, B possible tags
• Big O = ?
• Depth first search?
• Breadth first search?

Page 24:
Breadth first/depth first search
N
NN
N
NN
N
NN
N
NN
N
NN
N
NN
N
NN
I
eat
chinese

Page 25:
The Viterbi Algorithm (Dynamic 
programming)
• Some computation are redundant
• We can save computation from previous steps

Page 26:
Dynamic programing
• Saving computation for future use. How?
• Example: Find best route from A to B
A
B
c
1
5
5
2
3
4

Page 27:
Redundancy
N
NN
N
NN
N
NN
N
NN
N
NN
N
NN
N
NN
I
eat
chinese
The nodes with lower probability is redundant 

Page 28:
The Viterbi Algorithm (Dynamic 
programming)
• Some computation are redundant
• We can save computation from previous steps
• Creates two matrices
•
saves the best probability at word position i for hidden state t
• B[i, t] saves the previous hidden state that maximize this current 
state probability

Page 29:
Viterbi

Page 30:
Viterbi
N
NN
I
eat
chinese
N
NN
N
NN
Save only the best path till that point

Page 31:
Decoding example
Aij
To N
To NN
From N
0.6
0.4
From 
NN
0.5
0.5
Bik
I
eat
Chinese
State N
0.8
0.01
0.19
State NN
0.1
0.45
0.45
𝝅[i, t]
I
eat
Chinese
State N
0.8
State NN
0.1
B[i, t] 
I
eat
Chinese
State N
-
State NN
-
We ignore the log to do easy compute
We also ignoring initial state probability.
What if we want to include it?

Page 32:
Decoding example
Aij
To N
To NN
From N
0.6
0.4
From 
NN
0.5
0.5
Bik
I
eat
Chinese
State N
0.8
0.01
0.19
State NN
0.1
0.45
0.45
𝝅[i, t]
I
eat
Chinese
State N
0.8
0.005
State NN
0.1
B[i, t] 
I
eat
Chinese
State N
-
N
State NN
-
0.8 * 0.6 * 0.01 vs 0.1 * 0.5 * 0.01
0.0048              vs 0.0005

Page 33:
Decoding example
Aij
To N
To NN
From N
0.6
0.4
From 
NN
0.5
0.5
Bik
I
eat
Chinese
State N
0.8
0.01
0.19
State NN
0.1
0.45
0.45
𝝅[i, t]
I
eat
Chinese
State N
0.8
0.005
0.014
State NN
0.1
0.144
0.032
B[i, t] 
I
eat
Chinese
State N
-
N
NN
State NN
-
N
NN

Page 34:
Decoding example (backtrack)
Aij
To N
To NN
From N
0.6
0.4
From 
NN
0.5
0.5
Bik
I
eat
Chinese
State N
0.8
0.01
0.19
State NN
0.1
0.45
0.45
𝝅[i, t]
I
eat
Chinese
State N
0.8
0.005
0.014
State NN
0.1
0.144
0.032
B[i, t] 
I
eat
Chinese
State N
-
N
NN
State NN
-
N
NN
N, NN, NN

Page 35:
Decoding (Reconstructing T)
• We can find the best T by
• Find that best probability at the end
• Backtrack according to B[i,t]
• This gives a big O of
• We need to compute and create a table of size O( B N)
• For each value we need to perform B computations
• O( B2 N),  Space complexity of O(2 B N)
• What happens if B is big?

Page 36:
Large hidden states
Very expensive
B2

Page 37:
Pruning with Beamsearch
Keep K active states at each step
K = 3

Page 38:
Pruning with Beamsearch
Keep K active states at each step
K = 3
Prune, keep K best nodes

Page 39:
Pruning with Beamsearch
Keep K active states at each step
K = 3

Page 40:
Pruning with Beamsearch
Keep K active states at each step
K = 3

Page 41:
Pruning with Beamsearch
Keep K active states at each step
K = 3

Page 42:
Pruning with Beamsearch
Keep k active states at each step
K = 3
prune

Page 43:
Pruning with Beamsearch
Keep k active states at each step
Beam size = K = 3
O(kB) per word index

Page 44:
Inadmissibility
Pruning can give the wrong answer
True answer is lost because of the pruning

Page 45:
Beam search
• Beam search is inadmissible (can be wrong)
• Size of beam affect the quality of the answer
• Practically still useful even for small size (K < 10 in machine 
translation)
• We will use beam search again for text generation.

Page 46:
HMM assumptions and disadvantages
• No dependency across words
• HMM is a generative model P(T, W )
• But we care about P(T | W)
• Mismatch between final objective and model learning objective
w
w
w
w
t
t
t
t

Page 47:
Solution: Conditional Random Fields 
(CRF)
• Every point in the chain now depends on the entire 
sentence
• CRF is a discriminative model
W
t
t
t
t
P(T|W) = π P(tn | tn-1, W) = P( t1 | W) P( t2 | t1 W) P(t3 | t2 W) P(t4 | t3 W)
Random variable
Random/stochastic process
Random field

Page 48:
Linear chain CRF
Linear-chain CRF models with these independent 
assumption:
(1) each label tn only depends on previous label tn-1
(2) each label tn globally depends on x
48
tN-1
tN
t1
t2
t3
W
...
Problem: This is a big function (depends on n + 2 things). Hard to estimate using 
our previous method (counting)

Page 49:
Workaround
• Probability distribution is a function (with special constraints)
• Find a function that will represent “probabilities”
• We can turn functions into probabilities easily.
• Softmax function normalization
• We just need to have a function that give higher values to more likely 
inputs

Page 50:
Goal
• Find a function that will represent “probabilities”
• Turn functions into probabilities
• Softmax function normalization
• We just need to have function that give higher to more likely inputs
• Building functions that represent the whole sequence is 
hard
• We’ll build by combining pieces
• But each piece should have the form
• This is from our independence assumption.
• We call these functions, feature functions

Page 51:
At each time step, a feature function 
is used to capture some characteristics of current label and 
the observation.
A feature function in linear-CRF:
Feature function
51
current label
previous 
label
input 
sequence
current index
returns a real value

Page 52:
Example features
In general, we often define a feature function as a binary 
function, taking current label and its dependent variable into 
account. For example:
•
transition function
52
n = 3
tn-1
tn
t1
t2
t3
W
...

Page 53:
Feature function: More examples
•
state function
•
The whole input sequences can be used in a feature 
function.
53
tN-1
tT
t1
t2
t3
W
...

Page 54:
Feature function: More example
Other features other than word form can be used too.
54

Page 55:
At each time step, a potential                            is a function 
that takes all feature functions into account, by summing 
their products with the associated weight 
Potential
55
number of all feature functions
kth feature function
weight associated with each 
feature function, estimated 
within training process
exponential function: used to 
convert the summation to the 
positive range
factors at time n

Page 56:
Potential: Example 
56
n
n=1
n=2
n=3
n=4
..
.
t*
NOUN
VERB
NOUN
VERB
..
.
w
The
fastest
fox
jumps
..
.
From feature functions and trained 
weights on the right, we can 
compute potentials for the 
predicted label sequence t* at 
time step n=3 as following:

Page 57:
Probability of the whole sequence
57
tN-1
tN
t1
t2
t3
W
...
<s>
</s>
Ψ1
ΨN
Ψ2
Ψ3
ΨN-1
x
x
x
x
x
...
Sum of scores for all possible 
labels with all possible input 
sequences
Score of a sequence label T for 
a given sequence input W
Joint probability distribution of input and output sequence             
can be represented as:
With p(T, W) we can compare
and pick the best T

Page 58:
Special states and characters
To simplify modeling, we add two new special states and 
characters:
<s> indicates the beginning of the sequence
</s> indicates the end of the sequence
tN-1
tN
t1
t2
t3
W
...
<s>
</s>

Page 59:
From the definition of factors, the joint distribution can be 
represented by
Product of sum over feature functions
59
Computing Z is intractable:
Imagine a sentence of 20 words with vocabulary size of 
100,000
we have to consider all (100000)20 possible input 
sequences!

Page 60:
Linear-chain CRF
Modeling conditional probability distribution    P(T|W)    is 
enough for classification tasks.
So, in linear-chain CRF, we model the conditional distribution 
by using these two equations:
60

Page 61:
Linear-chain CRF
A linear-chain CRF is a conditional distribution 
, where Z(x) is an instance-specific normalization function
61

Page 62:
Linear-chain CRF
62
normalization 
function
the sum of 
products of all 
possible output 
sequences
not the same as Z in 
the joint distribution
sum of weighted 
feature functions at 
one time step, then 
taken to the 
exponential function
multiply 
over 
all time 
steps

Page 63:
Linear-chain CRF big picture
• Wants P(T|W)
• Assumes independence, where we only consider P(tt-1,tt,W)
• How to model P(tt-1,tt,W)?
• Still too hard, let’s make it into a function where high value means 
high probability – potential functions
• Still too hard, let’s build it from pieces – feature functions
• We can get P(T,W) by multiplying all 
• This is not a probability, need a normalization
• We can also get P(T|W) from multiplying all                  
• Still need a normalization, but easier.
• This is our model, but!
• How to inference? How to train? What features functions?

Page 64:
How to inference?
• If we are given the model, and W
• Find T
• Not so straight forward, many possible T
• Noun, adjective, verb
• Noun, noun, verb
• Verb, noun, noun
• Too many possibilities to compare
• Solution: Dynamic programming, just like HMM

Page 65:
Viterbi algorithm
Viterbi algorithm is an algorithm for decoding based on 
dynamic programming.
From the equation, we can see the Z(W) is the same for all 
possible label sequences, so we can consider only the part 
in the rectangle 
65
Find the label 
sequence T that 
maximize this value
Goes over time step just like HMM viterbi

Page 66:
Viterbi: Structure 
Create two 2D arrays: VTB and Var
66
ADJ
ADP
...
<S>
</S
>
0
1
...
N
N+1
ADJ
ADP
...
<S>
</S
>
0
1
...
N
N+1
VTB(<s>, T) stores the highest 
value a label sequence that yT is 
<s> can take 
Var(<s>, T) stores the label yT-1
in the sequence that yields the 
value in VTB(<s>, T)

Page 67:
Viterbi: Initialization
67
0
0.12
0
0.03
...
...
0
10e-
9
1
10e-
3
ADJ
ADP
...
</S
>
<S>
0
1
...
N
N+1
ADJ
ADP
...
</S
>
<S>
0
1
...
N
N+1
-
<s>
-
<s>
...
...
-
<s>
-
<s>
The first label of the output 
sequence must be <s>
VTB
Var
For all other labels, 
apply the same way 
as ADJ
Factors at time t=1, for 
current label=ADJ, 
previous label=<s>
1

Page 68:
Viterbi: Iteration
68
0
0.12
...
2.32
0.22
0
0.03
...
0.02
0.10
...
...
...
...
...
0
10e-
9
...
0.03
3.09
1
10e-
3
...
1.12
0.02
ADJ
ADP
...
</S
>
<S>
0
1
...
N
N+1
ADJ
ADP
...
</S
>
<S>
0
1
...
N
N+1
-
<s>
...
NOUN
PREPO
-
<s>
...
NOUN
ADJ
...
...
...
...
...
-
<s>
...
VERB
ADJ
-
<s>
...
X
X
Var(t, n) Stores the value i
that maximize the value of 
VTB(t,n)
VTB
Var
Factors at 
time n
Maximum value 
that label i can take 
at time n-1
Iterate from n=2 to n=N+1
Find the max 
among values 
from all 
previous label i

Page 69:
Backtrack from Var(</s>, N+1) to get the label sequences 
that maximize P(T|W)
Viterbi: Finalize
ADJ
ADP
...
</S
>
<S>
0
1
...
T
N+1
-
<s>
...
NOUN
PREPO
-
<s>
...
NOUN
ADJ
...
...
...
...
...
-
<s>
...
VERB
ADJ
-
<s>
...
X
X
Var
For example:
output sequence = 
<s>, NOUN, …, NOUN, ADJ, </s>

Page 70:
Linear-chain CRF big picture
• Wants P(T|W)
• Assumes independence, where we only consider P(tt-1,tt,W)
• How to model P(tt-1,tt,W)?
• Still too hard, let’s make it into a function where high value means 
high probability – potential functions
• Still too hard, let’s build it from pieces – feature functions
• We can get P(T,W) by multiplying all 
• This is not a probability, need a normalization
• We can also get P(T|W) from multiplying all                    and 
use chain rule.
• Still need a normalization, but easier.
• This is our model, but!
• How to inference? Viterbi
• How to train? What features functions?

Page 71:
Training Parameters?
Parameters to be learned are weights associated to each 
feature functions. So, the number of parameters equals the 
number of feature functions.
71
parameters

Page 72:
Training objective
For linear-chain CRF, parameters are trained by maximum 
likelihood.
To clarify, parameters θ are trained to maximize the log 
probability of all pairs of label T(i) and input W(i) in the 
training set. (i) represents the ith training sentence.
72
(maximize)

Page 73:
Learning algorithm
To learn parameters from the loss function          , several 
learning algorithm can be used. Some popular learning 
algorithms for linear-chain CRFs are
• Limited-memory BFGS
• Stochastic Gradient Descent
73

Page 74:
Feature functions?
• Anything you can think of, the more the better.
• The model will learn what is important.

Page 75:
CRFsuite
•
An implementation of CRFs for labeling sequential data in 
C++
SWIG API is provided to be an interface for various 
languages
http://www.chokkan.org/software/crfsuite/
•
python-crfsuite: Python binding for crfsuite
https://github.com/scrapinghub/python-crfsuite
•
An example use of python-crfsuite can be found at 
https://github.com/scrapinghub/python-
crfsuite/blob/master/examples/CoNLL%202002.ipynb
75

Page 76:
CRF with neural networks
• Change the softmax layer and loss function
• CRFlayer: P(y|yt-1,x) not just P(yt|x) 
Typical softmax only considers current word label
not the sequence

Page 77:
Neural network for POS
GRU
GRU
GRU
GRU
embedding
embedding
embedding
embedding
Fully 
connect
Fully 
connect
Fully 
connect
Fully 
connect
Softmax
Softmax
Softmax
Softmax

Page 78:
Neural network for POS with CRF output
GRU
GRU
GRU
GRU
embedding
embedding
embedding
embedding
Fully 
connect
Fully 
connect
Fully 
connect
Fully 
connect
Linear chain CRF
hn(tn|W)
h1(t1|W)
h2(t2|W)
h3(t3|W)
h4(t4|W)
maximize likelihood of sequence of tags instead

Page 79:
Neural network for POS with CRF output
Fully 
connect
Fully 
connect
Fully 
connect
Fully 
connect
[ hn(tn|W)T(tn|tn-1)]
h1(t1|W)
h2(t2|W,t1)
h3(t3|W,t2)
h4(t4|W,t3)
Trans
ition 
score
Trans
ition 
score
Trans
ition 
score
Additional param with 
size of |T|^2
a slight problem in gradient computation
needs to use forward-backward algorithm to compute the gradient 
in this layer for efficient computation see Appendix A in the book

Page 80:
Neural network for POS with CRF output
• Need to use Viterbi for finding the best sequence
• Or instead of using the full sequence when decoding, consider each 
time step instead (marginal inference – faster decoding)
• This is pretty much a regular softmax during decoding
• Loss function: computed likelihood as a sequence
• Loss = -log(P(T*|W)) where T* is the true output
Example code: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
[ hn(tn|W)T(tn|tn-1)]

Page 81:
Performance
End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
https://arxiv.org/pdf/1603.01354.pdf

Page 82:
Thai Nested NER
https://aclanthology.org/2022.findings-acl.116/
https://github.com/vistec-AI/Thai-NNER
1 model per layer
1 model all layer

Page 83:
BERT and PoS

Page 84:
Additional reading
•
https://web.stanford.edu/~jurafsky/slp3/
•
Chap 17 (PoS)
•
HMM (Appendix A)
•
Huggingface’s token classification tutorial
https://huggingface.co/docs/transformers/en/tasks/token_c
lassification

Page 85:
Conclusion
• What is token classification?
• Traditional methods
• Sequence methods
• HMM
• CRF
• Viterbi and beamsearch
• Neural network methods

