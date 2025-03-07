Page 1:
+
RNN & Attention Mechanisms
2110572: Natural Language Processing Systems
Peerapon Vateekul & Ekapol Chuangsuwanich
Department of Computer Engineering, 
Faculty of Engineering, Chulalongkorn University
Credit: Can Udomcharoenchaikit & Nattachai  Tretasayuth
1

Page 2:
+ Outline
RNN & Attention Mechanism
2
■RNN
■Attention Mechanism

Page 3:
+
RNN Architectures
3

Page 4:
+ Different types of RNN architectures
Reference: The Unreasonable Effectiveness of Recurrent Neural Networks, http://karpathy.github.io/2015/05/21/rnn-effectiveness/
many-to-many
many-to-one
one-to-many
many-to-many 
(encoder-decoder)
4

Page 5:
+ Many-to-many 
■You have seen and implemented this type of RNN architecture in your homework already.
■E.g. Tokenization, POS tagging
■Sequence Input, Sequence Output
Alice       talked        to            Bob
Noun      VerbPast    Prep       Noun
5

Page 6:
+ Many-to-one
■E.g. Sentiment Analysis, Text classification
■Sequence input
I            liked          this        food
Positive
6

Page 7:
+ One-to-many  
■Sequence output
■E.g. Music Generation, Image caption generation
■Music generation
■Input: Initial seed
■Output: Sequence of music notes
■Image caption generation
■Input: Image features extracted by CNN
■Output: Sequence of text
เดิน          หนา        ประเทศ        ไทย
<S>
7

Page 8:
+ Many-to-many (encoder-decoder)
■Sequence Input, Sequence output
■These two sequences can be of different length
■E.g. Machine Translation
■Input: English Sentence
■Output: Thai Sentence
■Machine Translation is also a text generation task
Encoder
Decoder
Water     Filter   
เครื่อง        กรอง        น้ำ
8

Page 9:
+ Text generation model (training)
■One-to-Many RNN (autoregressive)
■The only real input is x<1>
■a<0> is the initial hidden state.
■ŷ is the predicted output.
■y is an actual output.
■During the training phase, instead of using the predicted output 
to feed into the next time-step, we use the actual output.
Training
Inference
REAL SEQUENCE!!! (y, not ŷ )
*** Teacher forcing ***
9
a<t>=Wa<t-1>+Wx<t>+b
<s>
1
2
2
1
This
is
an
pen
</s>
This
is
a
pen
</s>
pred
ground truth
This
is
a
pen

Page 10:
+ Text generation model (inference; testing)
■To generate a novel sequence, the inference model  (testing phase)  randomly samples an output  from a softmax 
distribution.
Training
Inference
10
<s>
This
is
an
This
is
a
pen
</s>
ground truth
This
is

Page 11:
+ In class demo1: One-to-Many RNN Text generation
In-class demo: Generating a piece of text using RNN; Random Date Generation “2018-03-19”
 2              0               1                8                -           03-19
<S>
11
 2              0               1                8                -           03-19

Page 12:
+
Attention Mechanism
12

Page 13:
+ Attention Mechanism (Many-to-Many) 
Attention is commonly used in sequence-to-sequence model, it allows the decoder part of the network to 
focus/attend on a different part of the encoder outputs for every step of the decoder’s own outputs.
Why attention?
       This is what we want you to think about: How can information travel from one end to another in neural 
networks?
Machine Translation Problem:  English to Thai
a1
a2
a3
a4
a5
13

Page 14:
+ Attention Mechanism (cont.)
Why attention?
      “You can’t cram the meaning of a whole %&!$# sentence into a single $&!#* vector!” - Raymond 
Mooney (2014)
Reference:http://yoavartzi.com/sp14/slides/mooney.sp14.pdf
14
Machine Translation Problem:  English to Thai
Will it forget with 
long sentences?

Page 15:
+ Attention Mechanism (cont.)
Why attention?
Main idea: We can use multiple vectors based on the length of the sentence instead of one.
Attention mechanism = Instead of encoding all the information into a fixed-length vector, the 
decoder gets to decide parts of the input source to pay attention.
a1
a2
a3
a4
a5
15
Machine Translation Problem:  English to Thai

Page 16:
+ Graphical Example: English-to-Thai machine translation
■This is a rough estimate of what might occur for English-to-Thai translation 
My
name 
is
Sam
ฉัน
ชื่อ
แซม
decoder
encoder
min
max
16
Machine Translation Problem:  English to Thai

Page 17:
+ Graphical Example: English-to-French machine translation
Reference: Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly 
learning to align and translate." ICLR(2015).
17
min
max

Page 18:
+ Attention Mechanism: Recap Basic Idea
■
Encode each word in the sequence into a vector 
■
When DECODING, perform a linear combination of these encoded vectors from 
the encoding step with their corresponding “attention weights”.
■
(scalar 1)(encoded vector1) + (scalar 2)(encoded vector 2) + (scalar 3)(encoded vector 3)
■
A vector formed by this linear combination is called “context vector”
■
Use context vectors as inputs for the decoding step 
 
Reference: Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural 
machine translation by jointly learning to align and translate." ICLR(2015).
18
j = each encoder’s input
i = each decoder’s input

Page 19:
19
Reference: Bahdanau, D., Cho, K., & Bengio, Y.. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015
target word = decoder
source = encoder
Bidirectional LSTM
Decoder
Encoder
My name
…
Sam

Page 20:
+ RNN and attention mechanism 
20
My
name 
is
Sam
ฉัน
ชื่อ
แซม

Page 21:
Attention Mechanism (1): Ci
context vector
attention score
encoder state at index j
previous hidden state/
decoder state
attention score(weight vector) 
of encoder state at index j
We want to calculate a context vector c based 
on hidden states h0….hj that can be used with 
the previous state si-1  for prediction. The 
context vector ci at position “i” is calculated as 
an average of the previous states weighted with 
the attention scores ai.
Reference: https://lilianweng.github.io/posts/2018-06-24-attention/  
encoder state at index j
21
i = decoder index
j = encoder index

Page 22:
Attention Mechanism (2): fatt
context vector
attention score
encoder state at index j
previous hidden state/
decoder state
encoder state at index j
22
The attention function fatt(si-1,hj) calculates an 
unnormalized alignment score between the current 
hidden state si-1 and the previous hidden state hj. 
There are many variants of the attention function fatt.
Reference: https://lilianweng.github.io/posts/2018-06-24-attention/  
i = decoder index
j = encoder index
attention score(weight vector) 
of encoder state at index j

Page 23:
Attention Calculation Example (1): Attention Scores
Now we want to predict แซม (i=2)
My
name
is
Sam
ฉัน
ชื่อ
encoder state at index j
previous hidden state/
decoder state
attention scores
fattn(s1,h1)
fattn(s1,h2)
s0
s1
h1
h2
h3
h4
fattn(s1,h3)
fattn(s1,h4)
softmax
a2=[a2,1 a2,2 a2,3 a2,4]
23
encoder
decoder
j=0
j=1
j=2
j=4
i = decoder index
j = encoder index

Page 24:
Attention Calculation Example (2): Context Vector
My
name
is
Sam
h1
h2
h3
h4
*
*
*
*
a2,1
a2,2
a2,3
a2,4
ชื่อ
S1 
  (S2-1)
context 
vector
attention score
encoder state at index j
∑
C2
24
decoder
encoder
RNN
Si-1
Ci
Yi

Page 25:
+ Type of Attention mechanisms
(Remember that there are many variants of attention function fattn )
Additive attention: The original attention mechanism (Bahdanau et al., 2015)  uses a one-hidden layer 
feed-forward network to calculate the attention alignment:
Multiplicative attention: Multiplicative attention (Luong et al., 2015) simplifies the attention operation by 
calculating the following function:
Self-attention: Without any additional information, however, we can still extract relevant aspects from the 
sentence by allowing it to attend to itself using self-attention (Lin et al., 2017) 
Key-value attention: key-value attention (Daniluk et al., 2017)  is a recent attention variant that separates 
form from function by keeping separate vectors for the attention calculation.
25
Reference: https://lilianweng.github.io/posts/2018-06-24-attention/ 

Page 26:
+ 1) Additive Attention
■The original attention mechanism (Bahdanau et al., 2015)  uses a one-hidden layer feed-forward network to 
calculate the attention alignment:
■Where Wa are learned attention parameters. Analogously, we can also use matrices W1 and W2 to learn separate 
transformations for si-1 and hj respectively, which are then summed (hence the name additive):
26
Reference: https://lilianweng.github.io/posts/2018-06-24-attention/ 
concatenation
One-hidden layer 
(Dense)

Page 27:
+ 2) Multiplicative Attention
■Multiplicative attention (Luong et al., 2015) [16] simplifies the attention operation by calculating the following 
function:
■Faster, more efficient than additive attention BUT additive attention performs better for larger dimensions
■One way to mitigate this is to scale fattn by
■Dot product of high dimensional vectors has high variance -> softmax is peaky -> small gradient -> harder to train
27
ds = #dimensions of hidden states in LSTM
        (context vector; latent factors) 

Page 28:
+ 3) Self Attention (1)
■
Without any additional information, we can still extract 
relevant aspects from the sentence by allowing it to 
attend to itself using self-attention (Lin et al., 2017)
■
ws1 is a weight matrix, ws2 is a vector of parameters. Note 
that these parameters are tuned by the neural networks.
■
The objective is to improve a quality of embedding 
vector by adding context information.
28
One-hidden layer 
(Dense)
Fully connected layer
I          like           this           food
Positive

Page 29:
+ Self-attention (2)
29

Page 30:
+ 4) Key-value attention (1)
30
Reference:  Daniluk, M., Rockt, T., Welbl, J., & Riedel, S. (2017). Frustratingly Short Attention Spans in Neural 
Language Modeling. In ICLR 2017.
Key=used for attention 
score calculation
Value=encoded vector
search value

Page 31:
Key-value attention (2)
31
key
value
previous outputs
(memory) with 
window L
current output
Context vector
(vt  not included)
The final 
representation
Attention score
Context vector
(vt not included)
Original vector 
at index t
k - output dimension
L - output length

Page 32:
Pictorial view of KV attention
32
K
V
Q

Page 33:
+ Scaled Dot-Product Attention (1)
-
Introduced in Attention is all you need (Viswani et al., 2017)
-
NO recurrence nor convolution
-
Widely used today in all Transformer-based model
-
“Relating different positions of a single sequence in order to 
compute a representation of the sequence”
33
Attention Is All You Need. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
and dk is a number.

Page 34:
+ Scaled Dot-Product Attention (2)
What are the “query”, “key”, and “value” vectors?
As an analogy, think of Google Search.
Query - what we want to know
Key - how to index information
Value - what kind of info is in each website
34
Attention Is All You Need. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
Query
Key
Value

Page 35:
http://jalammar.github.io/illustrated-gpt2/ 
35

Page 36:
http://jalammar.github.io/illustrated-gpt2/ 
36

Page 37:
http://jalammar.github.io/illustrated-transformer/ 
WQ, WK, and WV are 
linear layers
37
Input1
Input2

Page 38:
http://jalammar.github.io/illustrated-transformer/ 
1 x 512
1 x 512
512 x 64
1 x 64
1 x 64
512 x 64
1 x 64
1 x 64
512 x 64
1 x 64
1 x 64
38

Page 39:
http://jalammar.github.io/illustrated-transformer/ 
39
Attention scores

Page 40:
http://jalammar.github.io/illustrated-transformer/ 
40

Page 41:
+ Demo2: Neural Machine Translation with Attention
(Additive Attention)
■Translate: various date formats to ISO date format
■27 January 2018
2018-01-27
■27 JAN 2018
2018-01-27
41

Page 42:
+ Homework 3.1: Neural Machine Translation with 
Attention (Key-Value Attention)
■Translate: Thai name to English name
42

