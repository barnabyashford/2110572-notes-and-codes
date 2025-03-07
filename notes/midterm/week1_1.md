Page 1:
Introduction to Natural Language Processing
2110572: NLP SYS
Assoc. Prof. Peerapon Vateekul, Ph.D.
Department of Computer Engineering, Faculty of Engineering, Chulalongkorn University
peerapon.v@chula.ac.th
Credits to: TA.Pluem, TA.Knight, and all TA alumni 

Page 2:
Outlines
1) What is NLP?
2) History of NLP & Deep Learning
3) NLP System Building Overview & Demo
4) This Course
5) NLP Tools
2

Page 3:
1) What is NLP?
➢
Definition & Levels of Understanding
➢
NLP today
3

Page 4:
1.1) What is NLP?
4

Page 5:
Natural Language Processing (NLP)
Technology to handle human language (usually text) 
using computers 
●
Aid human-machine communication (e.g. question 
answering, dialog, code generation)
●
Aid human-human communication (e.g. machine 
translation, spell checking, assisted writing)
NLP (Interpretation)
AI/ML (Generation)
I want to eat Japanese food.
May I order Yayoi for you?
Language
(Text, Speech)
Meaning
5

Page 6:
Going beyond string matching
Goal: Analyze/understand language (NOT just string matching!)
•
Syntactic structures, Text classification, Entity/relation linking
6
We use NLP many times a day without knowing it!

Page 7:
Level of understanding in NLP
https://www.tutorialspoint.com/artificial_intelligence/artificial_intelligence_natural_language_processing.htm 
7
Lexical Analysis: 
Text → Paragraphs, Sentences, and Words
Syntactic Analysis (Parsing): 
Grammar/Relationship between words
Semantic Analysis: 
Exact meaning of the sentence
Discourse Integration: 
Meaning of the sentence based on the previous sentence (pronouns)  
Pragmatic Analysis: 
Actual Meaning based on the context and real-world knowledge

Page 8:
8

Page 9:
9
Tokenization
• Input: Mr.Smith goes to Washington 
• Output: [Mr.Smith, goes, to, Washington ]
Part of Speech 
tagging
• Input: [Mr.Smith,goes,to,Washington ]
• Output:[(Mr.Smith,NNP), (goes,VBZ), (to,TO), (Washington,NNP) ]
NER
• Input:[(Mr.Smith,NNP), (goes,VBZ), (to,TO), (Washington,NNP) ]
• Output:[(Mr.Smith,NNP,PER), (goes,VBZ,O), (to,TO,O), (Washington,NNP,LOC) ]
Application
• e.g. Word Cloud (Named Entity Only) 
PENN Part Of Speech Tags
• NNP – proper noun
• VBZ - Verb, 3rd person singular 
present
• TO –to
Ref: 
https://www.ling.upenn.edu/courses/Fall
_2003/ling001/penn_treebank_pos.html 
Named Entity Tags
• PER –Person
• LOC – Location
• ORG – Organization
• O – Other

Page 10:
10
Tokenization
●
Input:ขสมก.เล็งจัดหารถ
●
Output: ขสมก.,เล็ง,จัดหา,รถ
Part of Speech 
tagging
●
Input:[ขสมก.,เล็ง,จัดหา,รถ]
●
Output:[(ขสมก.,NR),(เล็ง,VV),(จัดหา,VV),(รถ,NN) ]
NER
●
Input:[(ขสมก.,NR),(เล็ง,VV),(จัดหา,VV),(รถ,NN) ]
●
Output:[(ขสมก.,NR,ORG),(เล็ง,VV,O),(จัดหา,VV,O),(รถ,NN,O)]
Application
●
e.g. Word Cloud (Named Entity Only) 
PENN Part Of Speech Tags
• NR – proper noun
• VV - Main verbs in clauses, 
verb-form 
• NN – Non-proper noun 
Ref: BEST2010 dataset 
Named Entity Tags
• PER –Person
• LOC – Location
• ORG – Organization
• O – Other

Page 11:
1.2) NLP Applications
11

Page 12:
NLP can Answer Questions
12
Retrieved Dec 31, 2024
1

Page 13:
…but (sometimes) makes up facts
13
Retrieved Dec 31, 2024 https://www.researchgate.net/publication/309344097_Alleviating_Freezing_of_Gait_using_phase-dependent_tactile_biofeedback 
2
3

Page 14:
NLP can Translate Text
14
http://www.bbc.com/news/business-42581934 

Page 15:
…but (sometimes) loses Translation Meaning
15
Retrieved Dec 31, 2024
Sarcasm
Uncommon 
Words
Idioms

Page 16:
NLP asks “why” are we Searching
16
What is the query’s intent?
Watch movie?
Review/
Summarize?
Webs related 
to query?

Page 17:
NLP can Extract Information from Text
Data science perspective on clinical research
17
Parsing pathology reports into database
Abstract clinical records into a database

Page 18:
…but sometimes fail at Basic Tasks
From Bangkok Post Moo Deng story 31 Dec. 2024
18
Bangkok Post - Moo Deng voted best story of 2024, beats Icon Group fraud 31 Dec 2024
https://www.bangkokpost.com/thailand/general/2930405/moo-deng-voted-best-story-of-2024-beats-icon-group-fraud 
NER by Stanford CoreNLP
NER by spaCy
Different tools can be failed in various cases:
●
spaCy: CANNOT capture “president” (title)
●
Stanford: CANNOT capture “Move Forward” (ORG)
●
Both fail: “Suan Dusit” as PERSON

Page 19:
NLP can Analyze Trends
19

Page 20:
20
Ref: Prof. Regina Barzilay, NLP @MIT
…but (sometimes) fails on random things
NLP is difficult! 
Word-level ambiguity!
26 September 2008 the movie 
"Passengers" was released and a 
stock rise in BRK was observed at 
1.43%
October 3, 2008, ‘Rachel Getting 
Married’ was released and BRK shoot 
up 0.44%

Page 21:
Many problems are trivialized in the LLM Era
In November 2022, OpenAI released ChatGPT and thus began the LLM era.
21

Page 22:
2) History of NLP & 
Deep Learning
22
➢
History of NLP
➢
Deep Learning

Page 23:
A Brief Timeline of NLP
23
1950
●
Alan Turing’s Turing Test
●
Chomsky’s ‘universal grammar’1
●
Interest in automatic translation2
●
MIT ELIZA rule-based rephrasing
●
MIT SHRDLU moving blocks
●
Conceptual ontologies to 
chatterbots
Logical Era
1980
Statistical Era
●
Decision trees (hard rule-based)
●
Statistical models
●
Emergence of large textual corpora
●
1993 IBM alignment models for 
statistical machine translation
1990
Neural Era
●
1990 Elman network word 
embedding
●
Recurrent Neural Networks (RNN)
●
1995 Improved RNN as LSTMs
●
1997 Bidirectional recurrent neural 
networks (BRNN)
●
2006 Bi-LSTM in speech recognition 
and text-to-speech
2010
●
2012 ImageNet Alexnet
●
2013 Word2Vec and GloVe
●
2015 Google BERT
●
Attention and Transformers
●
Pre-trained Models and Transfer 
Learning (ULMFiT)
●
2019 OpenAI GPT-2 
Bag of words
Word Embeddings
RNN
LSTM
Bi-LSTM
Attention
Transformers
1 Noam Chomsky’s Syntactic Structures, a rule-based system of syntactic structures
2 Georgetown experiment in 1954 to translate 60 Russian sentences into English
https://en.wikipedia.org/wiki/History_of_natural_language_processing 
Neural (DL) Era

Page 24:
24
A Brief Timeline of NLP since 2020 (LLM)
By company as of 2023: OpenAI, Google, Meta
2020
●
GPT-3 and Large 
Language Models
2024 Lee et. al. A Survey of Large Language Models in Finance (FinLLMs) 
https://arxiv.org/abs/2402.02315 

Page 25:
25
A timeline of existing large language models in recent years. 
2023 Zhao et. al. A Survey of Large Language Models https://arxiv.org/abs/2303.18223 [Cited by 2639]
A Brief Timeline of NLP since 2020 (LLM)
Based on timeline as of 2023 (overall companies)

Page 26:
History: Logical Era
1) Symbolic approach
●
Encode all the required information into computer
●
In the 1960s and 1970s, Noam Chomsky (an eminent linguist) believed that statistical 
techniques would never be sufficient to gain a deep understanding of human language. 
●
This led to the dominance of knowledge-based approaches, requiring human experts to 
encode knowledge into computers.
●
Disadvantage: It is required substantial human effort.
26
The Internals of SHRDLU
Ref: Prof. Regina Barzilay, NLP @MIT
Noam Chomsky, MIT

Page 27:
History: Statistical Era
2) Statistical approach
●
Infer language properties from language samples
●
In 1980s, an empirical revolution took place. Inspired by information 
theory, it began using probabilistic approaches in NLP.
●
Disadvantage: It requires handcrafted features.
27
Conditional Random Fields (CRF)
Ref: Prof. Regina Barzilay, NLP @MIT
PennTree Bank (1993):  one 
million words from WSJ, 
manually annotated with 
syntactic structure

Page 28:
Case Study: Determiner placement
Symbolic vs. statistical approaches
Goal: Where to place “the” (determiner).
28
Ref: Prof. Regina Barzilay, NLP @MIT
Scientists in United States have found way of turning lazy monkeys into 
workaholics using gene therapy.  Usually monkeys work hard only when they 
know reward is coming, but animals given this treatment did their best all time. 
Researchers at National Institute of Mental Health near Washington DC, led by 
Dr Barry Richmond, have now developed genetic treatment which changes their 
work ethic markedly. "Monkeys under influence of treatment don't 
procrastinate," Dr Richmond says.Treatment consists of anti-sense DNA - mirror 
image of piece of one of our genes - and basically prevents that gene from 
working. But for rest of us, day when such treatments fall into hands of our 
bosses may be one we would prefer to put off.

Page 29:
Case Study: Determiner placement (cont.)
1) Symbolic approach
●
Determiner placement is largely determined by:
○
Type of noun (countable, uncountable)
○
Uniqueness of reference
○
Information value (given, new)
○
Number (singular, plural)
●
However, many exceptions and special cases play a role:
○
The definite article is used with newspaper titles (The  Times),  but zero article in names of  
magazines and journals  (Time)
●
Hard to manually encode this information!
29
Ref: Prof. Regina Barzilay, NLP @MIT

Page 30:
Case Study: Determiner placement (cont.)
2) Statistical approach
●
Consider it as classification 
●
Predictions: {-1, +1}
●
Features:
○
Plural?
○
first appearance in text? 
○
head token 
○
…
30
Ref: Prof. Regina Barzilay, NLP @MIT

Page 31:
Limitation of traditional statistical approach
●
Sparsity:
○
feature vectors are typically 
high-dimensional and sparse 
(i.e., most elements are 0).
●
Feature engineering:
○
Need experts to manually 
design features
31
1
•Data collection
•Obtain the labels {-1, +1}
2
•Feature Extraction
•The most common features in NLP is word 
n-grams → encoded as one-hot vector
3
•Training
4
•Testing
Map discrete, one-hot vectors into low-dimensional
continuous representations. 
*** Self-learned features → Deep Learning ***

Page 32:
History: Neural Era
3) Deep Learning approach:
●
It is a feature-engineering embedded neural approach.
●
Since the 2010s, it has been gaining a lot of attention and showing many 
successes.
32
Ref: Prof. Regina Barzilay, NLP @MIT
LSTM

Page 33:
The Spark of DL in Various Domains: Success of AlexNet in 
Large Scale Visual Recognition Challenge (ILSVRC) 2012
33
Features trained on ILSVRC-2012 generalize to the 
SUN-397 dataset. [Donahue et al., 2014]
Visualization of the information captured by features across different 
layers in GoogLeNet trained on ImageNet. (Source: Distill)
Deng, J. and Dong, W. and Socher, R. and Li, L.-J. and Li, K. and Fei-Fei, L. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09.

Page 34:
NLP “Pretrained” Word Embeddings
Word2Vec
34
GloVe
2013
ULMFiT
2018
Bert

Page 35:
What is Deep Learning?
35
Part of the machine learning field of learning 
representations of  data. Exceptional effective at 
learning patterns.
Utilizes learning algorithms that derive meaning out 
of data by using  a hierarchy of multiple layers that 
mimic the neural networks of our  brain.
Costs lots of time

Page 36:
Deep Learning – Basics (cont.)
36
What does it learn?
●
A deep neural network consists of a hierarchy of layers, whereby each layer  transforms 
the input data into more abstract representations (e.g. edge ->  nose -> face). 
●
The output layer combines those features to make predictions.

Page 37:
Deep Learning Application
37
Speech  
Recognition
Computer  
Vision
Natural Language  
Processing
CNN
RNN (LSTM)

Page 38:
NLP + Deep Learning = Deep NLP
●Modern NLP techniques are based on deep learning models.
●These models have obtained very high performance across various NLP tasks.
●They often do not require traditional linguistic feature  engineering to perform well.
38

Page 39:
Deep NLP + LLM (since 2020)
●
Multiple orders of magnitude larger than previous generations of deep learning models.
●
State-of-the-art performance across a wide range of tasks.
●
Usually perform well out of the box.  No training is required.
39
https://stanford-cs324.github.io/winter2022/ 
https://microsoft.github.io/generative-ai-f
or-beginners/#/ 

Page 40:
Reasons for exploring Deep Learning
●Learned features are easy to adapt and fast 
to learn
●Deep learning provides a very flexible,  
universal, and learnable framework for 
representing world, visual, and linguistic 
information
40

Page 41:
Reasons for exploring Deep Learning (cont.)
●Flexible neural “Lego pieces”
○
Common representation, diversity of 
architectural choices
●Can represent any levels of NLP
○
Word
○
Phrase
○
Sentence
○
Paragraph (document)
41
1
2
3

Page 42:
Reasons for exploring Deep Learning (cont.)
Example of encoding sentences
42
RNN
(LSTM, GRU)

Page 43:
Current problems with DL-based solutions
●
Not so transparent
○
Bias unintentionaly learned from data
○
Blackbox: Don’t know when it will fail, how it will fail, hard to fix if it’s wrong
●
Resource intensive (data and compute)
43
หมูกรอบ => |หมู|กรอบ|
ขาวผัดคะนาหมูกรอบหนึ่งจาน => |ขาวผัด|คะนา|หมูก|รอบ|หนึ่ง|จาน|

Page 44:
3) NLP System Building
Overview
44
https://phontron.com/class/anlp2024/ 

Page 45:
A General Framework for NLP Systems
45
Create a function to map an input X into an output Y, where X 
and/or Y involve language
    Input X                     Output Y                                 Task
       Text                  Continuing Text               Language Modeling
       Text             Text in Other Language             Translation
       Text                            Label                        Text Classification
       Text                  Linguistic Structure         Language Analysis
      Image                         Text                           Image Captioning
From https://phontron.com/class/anlp2024/

Page 46:
Methods for Creating NLP Systems
●
Rules: Manual creation of rules
●
Prompting: Prompting a language model w/o training
●
Fine-tuning: Machine learning from paired data <X, Y>
46
From https://phontron.com/class/anlp2024/

Page 47:
Data Requirements for System Building
●
Rules/prompting based on intuition:            No data needed, but also no performance guarantees 
●
Rules/prompting based on spot-checks:       A small amount of data (small testing data) with input X only 
●
Rules/prompting with rigorous evaluation:  Development set with input X and output Y (e.g., 200-2000 
examples). Additional held-out test sets are also preferable (full testing data).
●
Fine-tuning: Additional train set. More is often better — constant accuracy increases when data size 
doubles (training & testing data). 
47
Dev
Test
Train
From https://phontron.com/class/anlp2024/
Graham Neubig

Page 48:
Example Task: Review Sentiment Analysis
Given a review on a reviewing website (X), decide whether its 
label (Y) is positive (1), negative (-1) or neutral (0)
48
From https://phontron.com/class/anlp2024/

Page 49:
A Three-step Process for 
Making Predictions
●
1) Feature extraction: Extract the salient features for 
making the decision from text
●
2) Score calculation (model): Calculate a score for one or 
more possibilities
●
3) Decision function: Choose one of the several possibilities 
49
From https://phontron.com/class/anlp2024/

Page 50:
Formally
50
From https://phontron.com/class/anlp2024/
1
2
3

Page 51:
Demo: Sentiment Classiﬁcation
●Rule-Based Model
●ML-Based Model (BOW)
●NN-Based Model (not now)
51

Page 52:
Demo1: Rule-based Sentiment 
Classiﬁcation Code Walk!
See code for all major steps: 
1. Featurization 
2. Scoring 
3. Decision rule 
4. Accuracy calculation 
5. Error analysis
52
From https://phontron.com/class/anlp2024/

Page 53:
Now let's improve!
1.
What's going wrong with my system? → Look at error analysis
2.
Modify the system (featurization, scoring function, etc.)
3.
Measure accuracy improvements, accept/reject change
4.
Repeat from 1
5.
Finally, when satisfied with dev accuracy, evaluate on test!
53
From https://phontron.com/class/anlp2024/

Page 54:
Rule-based Challenges
54
From https://phontron.com/class/anlp2024/
Challenge
Examples
Solution
Low Frequency Words
The action switches between past and 
present, but the material link is too 
tenuous to anchor the emotional 
connections that purport to span a 
125-year divide . → negative
Keep working till we get all of them? 
Incorporate external resources such as 
sentiment dictionaries?
Conjugation
It's basically an overlong episode of 
Tales from the Crypt . → negative
Use the root form and POS of the word? 
Note: Would require morphological analysis.
Negation
This one is not nearly as dreadful as 
expected . → positive
If a negation modifies a word, disregard it. 
Note: Would probably need to do syntactic 
analysis.
Metaphor, Analogy
Puts a human face on a land most 
Westerners are unfamiliar with. → 
positive
???

Page 55:
Demo: Sentiment Classiﬁcation
●Rule-Based Model
●ML-Based Model (BOW)
●NN-Based Model (not now)
55

Page 56:
Machine Learning-Based NLP
56
From https://phontron.com/class/anlp2024/

Page 57:
A First Attempt: Bag of Words (BOW)
Aim to solve low freq. words
57
From https://phontron.com/class/anlp2024/
Features f are based on word identity, weights w learned
Which problems mentioned before would this solve?
Extracted handcrafted features 
(OneHot; Dict)
Learnable weight
(ML)

Page 58:
What do Our Vectors Represent?
●
Binary classification: Each word has a 
single scalar, positive indicating “yes” 
and negative indicating "no.”
58
From https://phontron.com/class/anlp2024/

Page 59:
Demo2: Simple Training of BOW Models
Using an algorithm called “structured perceptron”
59
From https://phontron.com/class/anlp2024/

Page 60:
What’s Still Missing in BOW?
60
From https://phontron.com/class/anlp2024/

Page 61:
Demo: Sentiment Classiﬁcation
●Rule-Based Model
●ML-Based Model (BOW)
●NN-Based Model (not now)
61

Page 62:
A Better Attempt: Neural Network Models?
62
Powerful enough to perform classification, LM, any task! 
From https://phontron.com/class/anlp2024/

Page 63:
4) This Course
63
➢
Thai NLP Challenges
➢
Class Schedule
➢
Class Grading

Page 64:
In this class, we ask:
●
What goes into the building blocks of state-of-the-art NLP systems 
that work well at some tasks?
●
Where and why do current state-of-the-art NLP systems still fail?
●
How can we utilize NLP in real-world applications given real-world 
challenges?
64

Page 65:
Some NLP Challenges
●Complexity in representing, learning and using linguistic/situational/world/visual knowledge
65
Ref: Prof. Christopher Manning, CS224N/Ling284, 2017

Page 66:
Some NLP Challenges (cont.)
●Human languages are ambiguous (unlike programming and other formal languages), 
so some parts can be ignored.
●Human languages are interpretation that depend on real world, common sense, and 
contextual knowledge (pragmatic analysis)
66
At last, a computer understands you like your mother.”
Ambiguity at syntactic level: Different structures lead to different interpretations
The Pope’s baby steps on gays. [Ref: Prof. Christopher Manning, CS224N/Ling284, 2017] 
Ref: Prof. Christopher Manning, CS224N/Ling284, 2017
Ref: Prof. Regina Barzilay, NLP @MIT

Page 67:
Thai NLP Challenges
Word segmentation
●
No word delimiters
●
ฉัน|นํา|ดอก|ไม|ไป|ไหว|ศาล|พระ|ภูมิ|
ที่|โรง|เรียน|ประจํา|
●
ฉัน|นํา|ดอกไม|ไป|ไหว|ศาลพระภูมิ|ที่|
โรงเรียน|ประจํา|
●
ฉัน|นํา|ดอก|ไม|ไป|ไหว|ศาล|พระภูมิ|ที่
|โรง|เรียน|ประจํา|
●
ฉัน|นํา|ดอกไม|ไป|ไหว|ศาลพระภูมิ|ที่|
โรงเรียนประจํา|
Sentence segmentation
●No sentence boundary markers
67
Ref: Introduction to Thai NLP (Prachya Boonkwan), 2017

Page 68:
Thai NLP Challenges (cont.)
Syntax ambiguity
●
Pronouns and some constituents can 
be omitted as long as they can be 
implied from the context
Nostalgic Thai slang
68
Ref: Introduction to Thai NLP (Prachya Boonkwan), 2017

Page 69:
69
https://salehere.co.th/articles/
update-teen-words 

Page 70:
Class Schedule ttps://github.com/ekapolc/NLP_2025 
70

Page 71:
Course Grading
●
Assignments 35% (4%x10 HW capped at 35%)
●
Midterm 35% (in class exam 25% + take home 10%)
●
Paper presentation 10%
●
Project 20%
71
https://web.stanford.edu/~jurafsky/slp3/ 
3rd edition draft as of August 20, 2024 release

Page 72:
5) NLP Tools
72

Page 73:
73
Implementation

Page 74:
74
NLP Libraries
Version updated as of 2 Jan 2025

Page 75:
Thai NLP Libraries
75
https://pythainlp.org/docs/5.0/ 
Version updated as of 2 Jan 2025

Page 76:
LLM tools
76

Page 77:
Any questions? :)
77

