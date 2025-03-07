Page 1:
+
Subword
2110572: Natural Language Processing Systems
Assoc. Prof. Peerapon Vateekul, Ph.D.
Department of Computer Engineering, 
Faculty of Engineering, Chulalongkorn University
peerapon.v@chula.ac.th
Credits to: Aj.Ekapol & TA team (TA.Pluem, TA.Knight, and all TA alumni) 

Page 2:
+ Introduction
Problem: 
1) out-of-vocab
2) large vocabulary size
Solution: subword embedding
1) Byte-Pair Encoding (BPE)
2) WordPiece
3) Unigram
4) Sentencepiece
2

Page 3:
+
Byte-Pair Encoding (BPE)
3

Page 4:
+ Byte-Pair Encoding (BPE)
BPE was introduced in Neural Machine Translation of Rare Words with Subword 
Units (Sennrich et al., 2015).
Used in GPT-2, Roberta, and even ChatGPT
Relies on a pre-tokenizer that splits the training data into words.
Next, BPE creates a base vocabulary consisting of all symbols that occur in the set of 
unique words and learns to merge rules to form a new symbol from two symbols of 
the base vocabulary (similar to huffman coding; frequencies).
4

Page 5:
+ BPE example (1 sentence)
●
aaabdaaaba
●
ZabdZabac
○
Z=aa
●
ZYdZYac
○
Y=ab
○
Z=aa
●
XdXac
○
X=ZY
○
Y=ab
○
Z=aa
5
●
พราวและขาวนั+งบนราวดูข่าวคราวบนดาว
●
พรx และขx นั+งบนรx ดูข่x ครx บนดx
○
x=าว
●
พy และขx นั+งบนy ดูข่x คy บนดx
○
x=าว
○
y=รx
●
พy และขx นั+งz y ดูข่x คy z ดx
○
x=าว
○
y=รx
○
z=บน

Page 6:
+ BPE - training
Example corpus
The most frequent symbol pair is "u" followed by "g", occurring 10 + 5 + 5 = 20 times 
in total. Thus, the first merge rule the tokenizer learns is to group all "u" symbols 
followed by a "g" symbol together. Next, "ug" is added to the vocabulary.
6
https://huggingface.co/course/chapter6/6?fw=pt

Page 7:
+ BPE - usage
7
How to use
-
bug = [“b”, “ug”] (“b” in dict)
-
mug = [“UNK”, “ug”] (“m” not in dict)
-
thug = [“UNK”, “hug”] (“t” not in dict)

Page 8:
+
Wordpiece
8

Page 9:
+ WordPiece
●
Google NMT(GNMT) uses a variant of this 
○
V1: wordpiece model
○
V2: sentencepiece model
●
WordPiece is the subword tokenization algorithm used for models such as BERT, 
DistilBERT, and Electra. 
●
Rather than char n-gram count, uses a greedy approximation to maximize 
language model log likelihood to choose the pieces (add n-gram that maximally 
reduces perplexity)
●
Like BPE, WordPiece learns merge rules. The main difference is the way the pair 
to be merged is selected. Instead of selecting the most frequent pair, WordPiece 
computes a score for each pair using the following formula:
9

Page 10:
+ WordPiece (cont.)
■There are 2 types of tokens: start token (not ##), and continuing token (##)
10
https://jacky2wong.medium.com/understanding-sentencepiece-under-standing-sentence-piece-ac8da59f6b08

Page 11:
+ WordPiece - training
Corpus
11
https://huggingface.co/course/chapter6/6?fw=pt

Page 12:
+ WordPiece - training (cont.)
Corpus
From initial vocab ["b", "h", "p", "##g", "##n", "##s", "##u"]
the best score goes to the pair ("##g", "##s") — the only one without a "##u" — = 5 / (20 * 5) =  1 / 20, and the first 
merge learned is ("##g", "##s") -> ("##gs")
Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs"]
Corpus: ("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 
4), ("h" "##u" "##gs", 5)
12
https://huggingface.co/course/chapter6/6?fw=pt

Page 13:
+ WordPiece - usage
Tokenization differs in WordPiece and BPE in that WordPiece only saves the final 
vocabulary, not the merge rules learned.
13
How to use: “the longest subword”
-
hugs = [“hug”, “##s”]
If not possible to find subwords, tokenize the whole word as UNK.
-
mug = [“UNK”]
-
bum = [“UNK”] (not [“b”, “##u”, UNK])

Page 14:
+
Unigram
15

Page 15:
+ Unigram
Start with a big vocab and reduce it based on a unigram LM loss
16

Page 16:
+ Unigram - training
Initial vocab = all substring of corpus
17

Page 17:
+ Unigram - training (cont.)
1st iteration of EM
The E step. Select the split for each word in the corpus with the highest prob.
18
Choose 1 
randomly

Page 18:
+ Unigram - training (cont.)
1st iteration of EM
The E step. Calculate loss.
19

Page 19:
+ Unigram - training (cont.)
1st iteration of EM
The M step. Remove the tokens that least impact the loss (remove p% at a time)
20
Try removing ug
Loss is still the same
Removing any token results in 
the same loss so choose 
randomly again

Page 20:
+ Unigram - training (cont.)
2nd iteration of EM
The E step. Select the split for each word in the corpus with highest prob.
21
(10/144) * (36/144) = 1.7e-02

Page 21:
+ Unigram - training (cont.)
2nd iteration of EM
The E step. Calculate loss.
22
1.7e-02

Page 22:
+ Unigram - training (cont.)
2nd iteration of EM
The M step. Remove the tokens that least impact the loss (remove p% at a time)
23
Removing “bu” gives the least 
loss so “bu” is removed

Page 23:
+ Stop?
1. Convergence of Likelihood: the likelihood of the data (given the current 
subword vocabulary) between consecutive iterations is smaller than a 
predefined threshold
2. Minimal Vocabulary Updates: the changes in the subword vocabulary between 
iterations become negligible.
3. Maximum Iterations
4. Fixed Vocabulary Size
5. Performance Metrics: Stop when the metric (e.g., perplexity) improvement 
between iterations plateaus.
24

Page 24:
+
SentencePiece
25

Page 25:
+ SentencePiece
■SentencePiece: A simple and language-independent subword tokenizer and 
detokenizer for Neural Text Processing (Kudo et al., 2018) 
■It aims to solve 2 issues.
■Issue 1: Which one should be the correct detokenization?
■
Tokenize(“World.”) == Tokenize(“World_.”)
■Issue 2: End-to-End to avoid the need of language-specific tokenization.

Page 26:
+ SentencePiece (cont.)
Introduces “_ (U+2581)” to preserve whitespace for detokenization
27
https://aclanthology.org/D18-2012.pdf
https://github.com/google/sentencepiece

