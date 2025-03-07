Page 1:
+
Language Modeling
2110572: Natural Language Processing Systems
Assoc. Prof. Peerapon Vateekul, Ph.D.
Department of Computer Engineering, 
Faculty of Engineering, Chulalongkorn University
Peerapon.v@chula.ac.th
Credits to: Aj.Ekapol & TA team (TA.Pluem, TA.Knight, and all TA alumni) 

Page 2:
+ Outline
2
■Introduction
■N-grams
■Evaluation and Perplexity
■Smoothing
■Neural Language Model

Page 3:
+
Introduction
3

Page 4:
+ Introduction
■Language Model (or Probabilistic Language Model for this course) ’s goal is 
(1) to assign probability to a sentence, or 
(2) to predict the next word
■“Do you live in Bangkok?” and “Live in Bangkok do you?”
■Which sentence is more likely to occur?
4
“… the problem is to predict the next word given the previous 
words. The task is fundamental to speech or optical character 
recognition and is also used for spelling correction, handwriting 
recognition, and statistical machine translation.”
— Page 191, Foundations of Statistical Natural Language Processing, 1999.
Maximal matching = 3
We need to verify with Language Model (LM)

Page 5:
+ Introduction (cont.)
■Application
■1) Text Generation
■Generating new article headlines
■Generating new sentences, paragraphs, 
or documents
■Generating suggested continuation of a 
sentence
■For example: The Pollen Forecast for 
Scotland system [Perara R., ECAL2006]
■Given six numbers of predicted pollen 
levels in different parts of Scotland
■The system generates a short textual 
summary of pollen levels
■
https://en.wikipedia.org/wiki/Natural_language_generation
■2) Machine Translation
■3) Speech Recognition
5
Grass pollen levels for Friday have increased from the moderate to high levels of 
yesterday with values of around 6 to 7 across most parts of the country. However, 
in Northern areas, pollen levels will be moderate with values of 4. [as of 1-July-
2005]

Page 6:
+ Introduction (cont.)
■How to compute this sentence probability?
■S = “It was raining cat and dog yesterday”
■What is P(S)?
6

Page 7:
+ Introduction (cont.)
■
Conditional Probability and Chain Rule
■
Do you still remembe ?
■
Chain Rule:
■
Now, we can write P(It, was, raining, cat, and, dog, yesterday) as :
■
P(it) × P(was | it) × P(raining | it was) × P(cats | it was raining) × P(and | it was 
raining cats) × P(dogs | it was raining cats and) × P(yesterday | it was raining cats 
and dogs)
7

Page 8:
+ Problem with full estimation
■
Language is creative.
■
New sentences are created all the time.
■
...and we won’t be able to count all of them
8
Training:
<s> I am a student . </s>
<s> I live in Bangkok . </s>
<s> I like to read . </s>
Test:
<s> I am a teacher . </s>
→ P(teacher|<s> I am a) = 0
→ P(<s> I am a teacher . </s>) = 0

Page 9:
+
N-grams
9

Page 10:
+ N-grams: a probability of next word
■Markov Assumption
■Markov models are the class of probabilistic models that assume we can predict the 
probability of some future unit (next word) without looking too far into the past
■In other word, we can approximate our conditions to unigram, bigrams, trigrams or n-
grams
■E.g.,  Bi-grams 
■P(F | A, B, C, D, E) ~ P(F| E)
■P(class | There, are, ten, students, in, the) 
■Unigrams ~ P( class )
■Bigrams ~ P(class | the)
■Trigrams ~ P(class | in the)
10
There are ten students in the class.

Page 11:
+ N-grams (cont.): a probability of the whole 
sentence
■Now, we can write our sentence probability using Chain rule (full estimation)
= P(it, was, raining, cats, and, dogs, yesterday)
= P(it) x P(was | it) x P(raining | it was) x P(cats | it was raining) x P(and | it was raining cats) x 
P(dogs | it was raining cats and) x P(yesterday | it was raining cats and dogs) 
■And, with Markov assumption (tri-grams)
= P(it, was, raining, cats, and, dogs, yesterday) = 
= P(it) x P(was | it) x P(raining | it was) x P(cats |was raining) x P(and | raining cats) x P(dogs 
| cats and) x P(yesterday | and dogs)
11

Page 12:
+ N-grams (cont.): a probability of the whole 
sentence – Start & Stop
■And, with Markov assumption (tri-grams)
= P(it, was, raining, cats, and, dogs, yesterday) = 
= P(it) x P(was | it) x P(raining | it was) x P(cats |was raining) x P(and | raining cats) x P(dogs 
| cats and) x P(yesterday | and dogs)
■And, with Markov assumption (tri-grams) with start & stop
= P(<s>, it, was, raining, cats, and, dogs, yesterday, </s>) = 
= P(<s>) x P(it|<s>) x P(was | <s> it) x P(raining | it was) x P(cats |was raining) x P(and | 
raining cats) x P(dogs | cats and) x P(yesterday | and dogs) x P(</s>| dogs yesterday)
■
Start tokens give context for start of the sentence
■
End tokens give an end to the sentence for language generation (sample till 
end token)
■
P(<s>) is always 1.
12

Page 13:
+ N-grams (cont.): Example of Bigrams Prob.
■Estimating Bigrams Probability
■Assume there are three documents
■<s> I am Sam </s>
■<s> Sam I am </s>
■<s> I am not Sam </s>
13
From : https://web.stanford.edu/class/cs124/ by Dan Jurafsky
Bigrams Unit
Bigrams Probability
P( I | <s> )
= 2/3 = 0.67
P ( am |I )
= 3/3 =1.0
P ( Sam| am)
= 1/3 = 0.33
P (</s> | Sam )
= 2/3 =0.67
P (  Sam | <s>)
= 1/3 =0.33
P ( I| Sam )
= 1/3 =0.33
P (</s> | am )
= 1/3 =0.33
P ( not| am)
= 1/3 =0.33
P (Sam | not)
= 1/1 =1.0

Page 14:
+ N-grams (cont.): Example
14
From : https://web.stanford.edu/class/cs124/ by Dan Jurafsky
Bigrams Unit
Bigrams Probability
P( I | <s> )
= 2/3 = 0.67
P ( am |I )
= 3/3 =1.0
P ( Sam| am)
= 1/3 = 0.33
P (</s> | Sam )
= 2/3 =0.67
P (  Sam | <s>)
= 1/3 =0.33
P ( I| Sam )
= 1/3 =0.33
P (</s> | am )
= 1/3 =0.33
P ( not| am)
= 1/3 =0.33
P (Sam | not)
= 1/1 =1.0
Bigrams Unit
Bigrams Probability
P( I | <s> )
= 2/3 = 0.67
P ( am |I )
= 3/3 =1.0
P ( Sam| am)
= 1/3 = 0.33
P (</s> | Sam )
= 2/3 =0.67
P(<s>, I, am, Sam, </s>)
= 0.148137
P (  Sam | <s>)
= 1/3 =0.33
P ( I| Sam )
= 1/3 =0.33
P ( am |I )
= 3/3 =1.0
P (</s> | am )
= 1/3 =0.33
P(<s>, Sam, I, am , </s>)
= 0.035937
P( I | <s> )
= 2/3 = 0.67
P ( am |I )
= 3/3 =1.0
P ( not| am)
= 1/3 =0.33
P (Sam | not)
= 1/1 =1.0
P (</s> | Sam )
= 2/3 =0.67
P(<s>, I, am, not, Sam, </s>)
= 0.148137
■Estimating Bigrams Probability
■<s> I am Sam </s>
■<s> Sam I am </s>
■<s> I am not Sam </s>

Page 15:
+ N-grams (cont.): Counting table
assume on real data
■Estimating N-grams Probability
■Uni-gram counting
■Bi-grams counting (column given row)
■“i want” →c(prev, cur) = c(wi-1, wi) = c(want, i) = 827
15
From : https://web.stanford.edu/class/cs124/ by Dan Jurafsky
i
want
to 
eat
chinese
food
lunch
spend
2533
927
2417
746
158
1093
341
278
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
prev
curr

Page 16:
+ N-grams (cont.): Bi-grams probability table
tables
■Estimating N-grams Probability
■Divided by Unigram 
16
From : https://web.stanford.edu/class/cs124/ by Dan Jurafsky
P(<s>,I, eat, Chinese, food,</s>) = 1*0.0036*0.021*0.52*0.5 = 1.9 x 10-5
P(<s>,I, spend, to, lunch,</s>) = 1*0.00079*0.0036*0.0025*0.5 = 3.5 x 10-9
Assume P(I|<s>)=1, P(</s>|food)=0.5, P(</s>|lunch)=0.5
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
i
want
to 
eat
chinese
food
lunch
spend
2533
927
2417
746
158
1093
341
278
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
Sentence = “i want” & curr = “want”, prev = “i” 
p(want | i) = p(i, want) / p(i) = 827 / 2533 = 0.33

Page 17:
+ N-grams (cont.): Log likelihood
■We do everything in log space  ( ln(P(S)) ) to
■Avoid underflow (numbers too small)
■Also, adding is faster than multiplying
17

Page 18:
+
Evaluation
Which model is better?
19

Page 19:
+ Evaluation
■We train our model on a training set.
■We test the model’s performance on data we haven’t seen.
■A test set is an unseen dataset that is different from our training set, totally unused.
■An evaluation metric tells us how well our model does on the test set.
■Sometimes, we allocate some training set to create a validation set
■Which is a pseudo-test set, so we can tune performance
20

Page 20:
+ Evaluation 
■
Extrinsic Evaluation:
■
Measure the performance of a downstream task (e.g. spelling correction, machine 
translation, etc.)
■
Cons: Time-consuming
■
Intrinsic Evaluation:
■
Evaluate the performance of a language model on a hold-out dataset (test set)
■
Perplexity!
■
Cons: An intrinsic improvement does not guarantee an improvement of a 
downstream task, but perplexity often correlates with such improvements
■
Improvement in perplexity should be confirmed by an evaluation of a real task
21

Page 21:
+ Perplexity (1)
22
■
Perplexity is a quick evaluation metric for language models.
■
A better language model is one that assigns a higher probability to the test set
■
Perplexity can be seen a normalized version of the probability of the test set

Page 22:
+ Perplexity (2)
■
Perplexity is the inverse probability of the test set, normalized by the number of words:
■
Minimizing it is the same as maximizing probability
■
Lower perplexity is better!
23
P(<s>,I, eat, Chinese, food,</s>) = 1*0.0036 * 0.021 * 0.52*0.5 = 1.9 x 10-5
P(<s>,I, spend, to, lunch,</s>) = 1*0.00079*0.0036*0.0025*0.5 = 3.5 x 10-9

Page 23:
+ Perplexity (3)
■
Perplexity:
■
Logarithmic Version:
■
Logarithmic Version Intuition:
■
The exponent is number of bits to encode each word  
24

Page 24:
+ Perplexity (4): Intuition of Perplexity
■
Perplexity as branching factor:
■
number of possible next words that can follow any word
■
Average branching factor:
■
Consider the task of recognizing a string of random digits of length N, given that 
each of the 10 digits (0-9) occurs with equal probability.
■
How hard is this task?
25
Note:
Each of the digits occurs with equal 
probability: P = 1/10
10 times

Page 25:
Perplexity example
Perplexity is related to vocabulary size. 
Comparing perplexity between different vocabulary size is unfair!

Page 26:
+ Perplexity (5): PP(W) of “I eat chinese food”
Bi-grams
■
Perplexity:                                                                  or after taking log:
■
PP(<s>,I,eat,Chinese,food,</s>)
■
= 
■
= 
■
=    8.74
28
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
Assume P(I|<s>)=1, P(</s>|food)=0.5, P(</s>|lunch)=0.5

Page 27:
+ Smoothing
Motivation: Zeros and Unknown words
29

Page 28:
+ Zeros
●
Zeros 
○
Things that don’t occur in the training set 
○
but occur in the test set
○
and it is still in vocab lists.
P(BNK48| is into) = 0
30
Test set:
… is into BNK48
… is into ping-pong
Training set:
… is into health
… is into food
… is into fashion
… is into yoga

Page 29:
+ Zeros (cont.)
●
P(BNK48| is into) = 0
●
n-grams with zero probability
○
mean that we will assign 0 probability to the test set!
●
We cannot compute perplexity
○
division by zero (/0)
31

Page 30:
+ Unknown words (UNK)
■Words we have never seen before in training set and not in vocab list
■Sometimes call OOV (out of vocabulary) words
■There are ways to deal with this problem
■1) Assign it as a probability of normal word
■Step1) Create a set of vocabulary with minimum frequency threshold
■This is fixed in advance.
■Or from top n frequency
■Or words that have frequency more than 1,2,..,v
■Step2) Convert any words in training and testing that is not in this predefined set 
■to ‘UNK’ token.
■Simply, deal with UNK word as a normal word
■2) Or just define probability of UNK word with constant value
32
However, this still cannot solve the zero issue.
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

Page 31:
+
Smoothing Techniques
33

Page 32:
+ Smoothing
■Our training data is very sparse, sometimes we cannot find the n-grams (0) that we 
want.
■In some cases where we do not even have a unigram (a word or OOV), we will use “UNK” 
token instead
■Notable smoothing techniques
■Add-one estimation (or Laplace smoothing)
■Back-off
■Interpolation
■Kneser–Ney Smoothing 
34
ln(0) is undefined!
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

Page 33:
+ Smoothing#1: Add-one estimation 
35
From : https://web.stanford.edu/class/cs124/ by Dan Jurafsky
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
i
want
to 
eat
chinese
food
lunch
spend
i
6
828
1
10
1
1
1
3
want
3
1
609
2
7
7
6
2
to
3
1
5
687
3
1
7
212
eat
1
1
3
1
17
3
43
1
chinese
2
1
1
1
1
83
2
1
food
16
1
16
1
2
5
1
1
lunch
3
1
1
1
1
2
1
1
spend
2
1
2
1
1
1
1
1
■Add-one estimation (or Laplace smoothing)
■We add one to all the n-grams counts
■For bigram, where V is the number of unique words in the corpus:

Page 34:
+ Smoothing#1: Add-one estimation (cont.)
■Add-one estimation (or Laplace smoothing)
■Pros
■Easiest to implement
■Cons
■Usually perform poorly compared to other techniques
■The probabilities change a lot if there are too many zeros n-grams
■useful in domains where the number of zeros isn’t so huge
36

Page 35:
+
■Use less context for contexts you don’t know about
■Backoff
■use only the best available n-grams if you have good evidence
■otherwise backoff!
■Example:
■Tri-gram > Bi-grams > Unigram
■Continue until we get some counts
37
Smoothing#2: Backoff

Page 36:
+ Smoothing#3: Interpolation
38
■
Interpolation
■
mix unigram, bigram, trigram
■
Where C is a constant, often (1/vocabulary) in corpus
■
𝜆is chosen from testing on validation data set, and the summation of 𝜆i is 1 (Σ𝜆i=1)
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

