Page 1:
+
Neural Machine Translation (NMT)
2110572: Natural Language Processing Systems
Assoc. Prof. Peerapon Vateekul, Ph.D.
Department of Computer Engineering, 
Faculty of Engineering, Chulalongkorn University
peerapon.v@chula.ac.th 
Credit: TA.Knight, TA.Pluem, and all TAs
1

Page 2:
Outline
●
Part1) MT models
○
mBART (2020)
○
NLLB (started 2018)
■
m2m-100 (2020)
■
NLLB-200 (2022)
●
Part2) MT data sets
○
WMT
○
OPUS
○
FLORES-200
●
Part3) Evaluation
○
Accuracy-based score 
(BLEU, charF++)
○
Model-based score
■
COMET
■
Quality Estimation
2

Page 3:
Introduction
Machine Translation (MT) is a research field in NLP that aims to create a system that can 
translate text from one language to another.
3

Page 4:
Machine Translation Landscape
4
2022 Savenkov and Lopez. The State of the Machine Translation 2022. ACL Proceedings of the 15th Biennial Conference of the Association for 
Machine Translation in the Americas (Volume 2: Users and Providers Track and Government Track) https://aclanthology.org/2022.amta-upg.4/ 

Page 5:
Cloud MT Vendors with Stock Models
5
2022 Savenkov and Lopez. The State of the Machine Translation 2022. ACL Proceedings of the 15th Biennial Conference of the Association for 
Machine Translation in the Americas (Volume 2: Users and Providers Track and Government Track) https://aclanthology.org/2022.amta-upg.4/ 
Commercial (45)
AISA, Alibaba, Amazon, Apptek, Baidu, 
CloudTranslation, DeepL, Elia, Fujitsu, Globalese, 
Google, GTCom, IBM, iFlyTec, HiThink RoyalFlush, 
Lesan, Lindat, Lingvanex, Kawamura / NICT, Kingsoft, 
Masakhane, Microsoft, Mirai, ModernMT, Naver, 
Niutrans, NTT, Omniscien, Pangeanic, Prompsit, 
PROMT, Process9, Rozetta, RWS, SAP, Sogou, Systran, 
Tencent, Tilde, Ubiqus, Viscomtec, XL8, Yandex, 
YarakuZen, Youdao
Preview / Limited (5)
eBay, Kakao, QCRI, Tarjama, Birch.AI
Open Source Pretrained (3)
M2M-100, mBART, NLLB by Meta

Page 6:
Open Source MT Performance (COMET)
NLLB by Meta AI mostly show 
performance in the 2nd tier of 
commercial systems. 
For en-es (English-to-Spanish), 
NLLB scores are on par with the 
best commercial systems
For en-zh (English-to-Chinese) 
and en-ja (English-to-Japanese), 
the scores are quite low
6
2022 Savenkov and Lopez. The State of the Machine Translation 2022. ACL Proceedings of the 15th Biennial Conference of the Association for 
Machine Translation in the Americas (Volume 2: Users and Providers Track and Government Track) https://aclanthology.org/2022.amta-upg.4/ 

Page 7:
+
Part1) MT models
●
mBART (2020)
●
NLLB (started 2018)
○
m2m-100 (2020)
○
NLLB-200 (2022)
7

Page 8:
History of Machine Translation
1.
Rule-Based Machine Translation
a.
Manual, hand-crafted rules for vocabulary, grammar, etc.
b.
Low-quality translation and time-consuming.
c.
Cannot utilize context information!
2.
Statistical Machine Translation
a.
Use statistics from a parallel corpus
b.
Google translate (from 2006 to 2016)
3.
Neural Machine Translation
a.
Competitive performance but hard to debug
b.
Google translate (now)
8

Page 9:
Neural Machine Translation (NMT)
Prior to the Transformer, the dominant model in NMT was the RNN-based encoder-decoder.
9

Page 10:
Transformer
The self-attention mechanism in the Transformer allows 
for better input representations and faster computation 
due to parallelization.
This brings big performance improvements and thus 
allows researchers to scale the model more efficiently.
10

Page 11:
1) mBART
A standard encoder-decoder transformer model that pretrains on an input denoising objective. 
There are two steps: (1) pretrain on the denoising task and (2) finetune on the MT task.
11
Multilingual Denoising Pre-training for Neural Machine Translation. Liu et al 2020

Page 12:
1) mBART (cont.)
Pretraining on multilingual data (~1TB) improve translation quality, especially for low-resource 
language pairs.
12
Multilingual Denoising Pre-training for Neural Machine Translation. Liu et al 2020
improves

Page 13:
1) mBART (cont.)
It can also do zero-shot MT (with some tricks) that yields decent results.
13
Multilingual Denoising Pre-training for Neural Machine Translation. Liu et al 2020

Page 14:
1) mBART (cont.)
●
Originally trained on 25 languages, 25 more languages (50 total languages) were added 
via continual pretraining. 
●
Thai language included! The model is very large, though.
14
Multilingual Translation with Extensible Multilingual Pretraining and Finetuning. Tang et al 2020

Page 15:
2) No Language Left Behind (NLLB) [FB started in 2018]
From the webpage - “No Language Left Behind (NLLB) is a first-of-its-kind, AI 
breakthrough project that open-sources models capable of delivering evaluated, 
high-quality translations directly between 200 languages”...
15
https://ai.facebook.com/research/no-language-left-behind

Page 16:
2.1) m2m-100
Previous works mostly focus on translation from/to English (English-centric).
This paper introduces a many-to-many translation model and dataset of 100 languages (that’s 
9900 directions!)
16
Beyond English-Centric Multilingual Machine Translation. Fan et al 2020

Page 17:
2.1) m2m-100 (cont.)
The model is also an encoder-decoder model with additional language-specific (language 
token) sparse models.
17
Beyond English-Centric Multilingual Machine Translation. Fan et al 2020

Page 18:
2.1) m2m-100  (cont.)
The model outperforms mBART even though it trains on 100 languages. Thai language is also 
covered by this model.
18
Beyond English-Centric Multilingual Machine Translation. Fan et al 2020

Page 19:
2.2) NLLB-200
This model is capable of a total of 40,602 translation 
directions!
It is also an encoder-decoder transformer model. 
However, it is a sparse model (Mixture of Experts 
transformer).
19
No Language Left Behind: Scaling Human-Centered Machine Translation. NLLB Team 2022 
MoE

Page 20:
2.2) NLLB-200 (cont.)
With just 1.3B parameters, it outperforms 
Google Translate in some language pairs.
(Distilled) weight available! 
https://huggingface.co/facebook/nllb-200-
distilled-600M 
20
No Language Left Behind: Scaling Human-Centered Machine Translation. NLLB Team 2022 

Page 21:
2.2) NLLB-200 (cont.)
This paper also explains how they created a 
parallel corpus (NLLB-SEED), including a 
multilingual sentence encoder and a 
language identification model. 
The whole paper is 192 pages!
21
No Language Left Behind: Scaling Human-Centered Machine Translation. NLLB Team 2022 

Page 22:
MT on LLM Performance
LLM capabilities are evolving
GPT-4 beat supervised baseline NLLB in 41% 
of translation pairs but still has large gap to 
Google Translate
But the advantage is LLM can acquire moderate 
translation ability in zero-resource languages.
22
2024 Zhu et. al. Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis. ACL Findings of the Association for 
Computational Linguistics: NAACL 2024 https://aclanthology.org/2024.findings-naacl.176/ 

Page 23:
+
Part2) Notable datasets
WMT
OPUS
FLORES-200
23

Page 24:
1) WMT
●
WMT (Workshop on Statistical Machine Translation)—This is a machine translation dataset 
composed of a collection of various sources, including (1) news commentaries and (2) 
parliament proceedings.
●
It focuses mainly on European language pairs.
●
There are many versions from 2008 to 2014.
24
https://www.statmt.org/wmt14/translation-task.html 

Page 25:
2) OPUS
Opus - https://opus.nlpl.eu/ 
25

Page 26:
3) FLORES-200
●
FLORES-200 (Few-Shot Learning Oriented Evaluation for Machine Translation) is a 
large-scale multilingual dataset designed to evaluate machine translation (MT) models 
across 200 languages. It was developed by Meta AI (formerly Facebook AI) to benchmark 
multilingual models, such as mBART-50, NLLB (No Language Left Behind), and M2M-100.
●
FLORES-200 consists of translations from 842 distinct web articles, totaling 3,001 
sentences. These sentences are divided into three splits: dev, devtest, and test (hidden). On 
average, sentences are approximately 21 words long.
26
https://github.com/facebookresearch/flores/blob/main/flores200/README.md 

Page 27:
+
Part3) Evaluating a Translation
Syntactic Similarity (BLEU, ROUGE, METEOR, TER, chrF)
Semantic Similarity (BERTScore, COMET, BLEURT)
Human Judgement (HTER, DA, MQM)
27

Page 28:
Evaluation
Syntactic Similarity
28
Semantic Similarity
Human Judgement
BLEU
ROUGE
METEOR
TER
BERTScore
COMET
BLEURT
ChrF
HTER
DA
MQM

Page 29:
Syntactic Similarity
29
BLEU
ROUGE
METEOR
TER
ChrF
Translation Edit Rate: Measures the number of edits (insertions, deletions, shifts, and 
substitutions) required to transform a machine translation into the reference translation. 
Penalizes paraphrases/synonyms. Penalizes translations of different length.
BLEU is a precision focused metric that calculates n-gram overlap of the reference and 
generated texts (including brevity penalty–penalizing short sentences). Works well for 
structured content but struggles with paraphrasing.
Very similar to the BLEU definition, the difference being that Rouge is recall focused 
overlap. Often used for summarization but can apply to MT. ROUGE-N considers n-gram 
overlap. ROUGE-L considers longest common subsequence (LCS).
Very popular metrics
Improves over BLEU by considering synonyms, stemming, and paraphrasing (WordNet). 
The metric is a harmonic mean of unigram precision and recall (recall weighted 9x higher 
than precision). Penalty is changed to be correlated to number of adjacent chunks.
An F-score based on character n-gram precision and recall (instead of word or n-gram 
level). No need for tokenization. Better for highly inflected languages (e.g., Finnish, 
Turkish, Thai, Arabic) where words change forms frequently.

Page 30:
BLEU (Bilingual Evaluation Understudy) [2002] 
The most popular metric to (try to) measure the quality of predicted translations.
The idea is to measure the similarity of the predictions with human references.
30

Page 31:
BLEU (cont.) 
The score is calculated based on clipped n-gram precision.
Clipping prevents repetitive predicted sentence from getting a good score
For example, we could have
This means that the 1-gram precision is 3/3 or 100% (it shouldn’t be like this!)
31
https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b 

Page 32:
BLEU (cont.) 
The score is calculated based on clipped n-gram precision.
So we limit the count of each word to the maximum number of times that the word occurs in the 
target Sentence
Now the 1-gram precision becomes 3/6
There are 6 words in the predicted sentence
Note that precision now refers to the clipped precision
32
https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b 
“He” occurs max. one time in 2 reference 
sentences. So we clip it to 1.

Page 33:
BLEU (cont.) 
The score is calculated based on clipped n-gram precision.
Now we calculate the n-gram (clipped) precision for all N. The widely used number for N is 4 
and a uniform weight wn of N/4.
33
https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b 
The precision of 2-grams is 4/7. 
(#correct/#total)
[ pre(1-gram)*pre(2-gram)*pre(3-gram)*pre(4-gram) ]1/4

Page 34:
BLEU (cont.) 
The score is calculated based on clipped n-gram precision.
Next, we compute the brevity penalty. This penalizes predicted sentences that are too short.
For example, a predicted sentence can contain only one word and get a perfect n-gram 
precision score
34
https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b 
[0,1]
[0,1]
1

Page 35:
BLEU (cont.) 
The score is calculated based on clipped n-gram precision.
Finally, we can compute the BLEU score by
35
https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b 
1
(larger is better)
2
(larger is better)

Page 36:
BLEU (cont.) 
The score is calculated based on clipped n-gram precision.
Finally, we can compute the BLEU score by
Even though BLEU score is widely used, it has some important weaknesses:
-
It does not consider words that have the same meaning to be correct. For example, for the 
word “dog,” we can either use “หมา” or “สุนัข”.
-
It ignores the importance of words. With Bleu Score an incorrect word like “to” or “an” 
that is less relevant to the sentence is penalized just as heavily as a word that contributes 
significantly to the meaning of the sentence.
-
Most importantly, higher BLEU does not always mean a good score based on human 
judgment [1].
36
https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b
[1] https://aclanthology.org/J18-3002.pdf

Page 37:
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) [2004]
Introduced in 2004 Paper: Lin. ROUGE: a Package for Automatic Evaluation of Summaries. 
https://aclanthology.org/W04-1013/ 
Instead of precision measures like BLEU, ROUGE is more focused on recall. 
●
Mostly used for summarization
●
4 Versions of ROUGE:
○
ROUGE-N: Measures the n-gram overlap between the generated text and reference text.
○
ROUGE-L: Takes the longest common subsequences (LCS), useful for capturing structural similarity.
○
ROUGE-W: Weighs contiguous matches that are higher than other n-grams.
○
ROUGE-S: Measures skip-bigram overlap, where two words are considered but may not be adjacent.
Note: ROUGE-1, ROUGE-2, and ROUGE-L is mostly used
37
https://www.geeksforgeeks.org/understanding-bleu-and-rouge-score-for-nlp-evaluation/ 
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4753220 This paper says which ROUGE is mostly used in literature.

Page 38:
ROUGE-N
ROUGE-N is an n-gram recall between a candidate summary and a set of references. 
N indicates the number of N grams which can be 1 (unigram) and 2 (bigram).
In the paper, ROUGE-N is only defined for recall. 
In practice, the Python implementation shows ROUGE-N F1.
38
ROUGE-1
Candidate 1 : Summarization is cool
Reference 1 : Summarization is beneficial and cool
Reference 2 : Summarization saves time
As defined in paper
https://medium.com/@eren9677/text-summarization-387836c9e178 
Use only reference 1 since there are more overlapping words
Recall = 3/5 = 0.6
Precision = 3/3 = 1
ROUGE-1 F1= 2RP/(R+P)= 2*0.6/1.6 = 0.75

Page 39:
ROUGE-N
39
ROUGE-2
It’s the same process as above, but right now bigrams are determined.
https://medium.com/@eren9677/text-summarization-387836c9e178 
Candidate 1: (Summarization is),(is cool)
Reference 1: (Summarization is),(is beneficial),(beneficial and),(and cool)
Reference 2: (Summarization saves),(saves time)
Recall = 1/4 = 0.25
Precision = 1/2 = 0.5
ROUGE-2 F1= 2RP/(R+P)= 2*0.25*0.5/0.75 = 0.33
Huggingface Evaluate uses rouge-score package (replication of original perl package)
https://pypi.org/project/rouge-score/ 

Page 40:
ROUGE-L
ROUGE-L is Longest Common Subsequence (LCS) oriented. LCS is the longest sequence of words that 
appear in both the candidates and reference summaries, while keeping the order of the words intact.
Note that LCSes are not necessarily consecutive but still in order
40
Candidate: A fast brown fox leaps over a sleeping dog.
Reference: The quick brown fox jumps over the lazy dog.
https://medium.com/@eren9677/text-summarization-387836c9e178 
As defined in paper
The paper introduced β in F1-score. 
β = 1 means F1. 
β > 1, recall is given more weight 
than precision.
Recall = 4/9 = 0.44
Precision = 4/9 = 0.44
ROUGE-L = 2RP/(R+P)= 2*0.44*0.44/0.89 = 0.44

Page 41:
Other ROUGE Flavours
●
Previous ROUGE-L is at the sentence level. In summary level, newlines are 
interpreted as sentence boundaries, and the LCS is computed between each pair of 
reference and candidate sentences, and something called union-LCS is computed.
●
ROUGE-LSum evaluates LCS across multiple sentences, unlike ROUGE-L, which is 
sentence-level.
41
ROUGE-L Sum
Reference Summary (Ground Truth) [20 tokens]
“The President held a press conference today. He discussed the economy and upcoming policies. 
Several journalists asked about inflation concerns.”
Generated Summary (Model Output) [19 tokens]
“The President spoke at a press conference. He mentioned new policies and economic conditions. 
Reporters questioned him about inflation.”
LCS length = 7

Page 42:
Other ROUGE Flavours (cont.)
42
ROUGE-W
https://medium.com/@eren9677/text-summarization-387836c9e178 
ROUGE-L LCS does not differentiate consecutiveness. In the 
following example, Y1 and Y2 will have the same ROUGE-L 
score. However, Y1 should be a better match than Y1 
because it has consecutive matches. ROUGE-W improves 
the ROUGE-L by assigning more weight to consecutive 
LCSes. More details in this link.
ROUGE-S
ROUGE-S stands for skip-grams. The order of the words in 
each sequence is preserved, but arbitrary gaps are allowed 
between words. 
Sentence   : police killed the gunman
Skip-Grams : (“police killed”, “police the”, “police gunman”, 
              “killed the”,  “killed gunman”, “the gunman”)
Formula of ROUGE-S is almost same 
as ROUGE-L with an addition of 
combination function(C).

Page 43:
METEOR [2005]
Introduced in 2005 Paper: Banerjee and Lavie. METEOR: An Automatic Metric for MT Evaluation with Improved 
Correlation with Human Judgments https://aclanthology.org/W05-0909/ 
It’s a metric aimed to improve BLEU for machine translation based on the harmonic mean of unigram precision 
and recall (with recall weighted higher than precision). 
43
Several features that are not found in other metrics, such 
as stemming and synonymy matching, along with the 
standard exact word matching. (Uses WordNet)
METEOR has a correlation of up to 0.964 with human judgment at the corpus level, compared to BLEU's 
achievement of 0.817 on the same data set.
https://spotintelligence.com/2024/08/26/meteor-metric-in-nlp-how-it-works-how-to-tutorial-in-python/ 
https://en.wikipedia.org/wiki/METEOR 

Page 44:
METEOR (cont.)
The algorithm creates an alignment between unigrams 
of the candidate string and the reference string.
44
https://en.wikipedia.org/wiki/METEOR 
We then calculate the precision P, recall R, and harmonic mean (with recall weighted 9x more than precision)
m is the intersecting unigrams, wt is candidate unigrams and wr is reference unigrams
These measures account only for single words; they need to account 
for larger segments too! A penalty p is introduced for the grouping 
of the fewest possible adjacent chunks c in the candidate. 
●
um is is the number of unigrams that have been 
mapped.
●
c is the number of chunks.
●
This penalty has the effect of reducing Fmean by 
up to 50% without bigram or longer matches. 
The final score is Fmean with a penalty applied. 

Page 45:
METEOR (cont.)
45
Candidate : the
cat
sat
on
the
mat
Reference : on
the
mat
sat
the
cat
P = 6/6 = 1.0
R = 6/6 = 1.0
Fmean = 10 * 1.0 * 1.0 / (9*1.0 + 1.0) = 1.0
p = 0.5 * (3 / 6)3  = 0.0625
METEOR = 1.0 * (1-0.0625) = 0.9375
Chunks    : (the
cat)
(sat) (on
the
mat)
Example 1
Example 2
Candidate : the
cat
sat
on
the
mat
Reference : the
cat
sat
on
the
mat
P = 6/6 = 1.0
R = 6/6 = 1.0
Fmean = 10 * 1.0 * 1.0 / (9*1.0 + 1.0) = 1.0
p = 0.5 * (1 / 6)3  = 0.0023
METEOR = 1.0 * (1-0.0023) = 0.9977
Chunks    : (the
cat
sat
on
the
mat)
https://en.wikipedia.org/wiki/METEOR 

Page 46:
Translation Edit Rate (TER) [2006]
Introduced in 2006 Paper: Snover et. al. A Study of Translation Edit Rate with Targeted Human Annotation. 
https://aclanthology.org/2006.amta-papers.25/ . This metric was introduced with the human-edited version 
(HTER), which will be introduced later. 
It is an intuitive measure that measures the amount of editing that a human would have to perform to change
a system output so it exactly matches the reference translation. 
46
Possible edits include the insertion, deletion, and substitution of single words as well as shifts of word
sequences. A shift moves a contiguous sequence of words within the hypothesis to another location within the 
hypothesis.
Punctuation tokens are treated as normal words, and miscapitalization is counted as an edit.
https://aclanthology.org/2006.amta-papers.25.pdf 

Page 47:
TER (cont.)
Reference Translation (Ground Truth) [14 tokens]
“The President held a press conference today to discuss the economy and upcoming policies.”
Generated (Hypothesis) Translation [13 tokens]
“The President spoke at a press conference about the economy and new policies.”
Step 1: Count Edits
1.
“held” → “spoke” → Substitution (1 edit)
2.
“today to discuss” → Removed → Deletion (2 edits)
3.
“upcoming” → “new” → Substitution (1 edit)
Step 2: Compute TER Score
47

Page 48:
chrF & chrF++
-
A score based on character n-gram precision and recall.
-
No tokenization required!
-
The score averages over all n-grams.
-
The widely used number is 6 characters
-
Later, chrF++ adds word n-gram (2-gram) to the metric since it correlates more 
strongly with human judgement.
-
Now tokenization is required.
48
https://aclanthology.org/W15-3049.pdf 
https://github.com/m-popovic/chrF 

Page 49:
Semantic Similarity
49
BERTScore
COMET
BLEURT
Analyzes cosine distances between BERT representations of machine translation and 
human reference. More robust against word reordering and paraphrasing. May be 
unreliable for terminologies underrepresented in BERT model.
Uses a neural model (i.e. XLM-RoBERTa + Pooling/Feed-Forward) to predict machine 
translation quality (use source input and reference translation). More correlated with 
human judgements than BLEU. May penalize paraphrases/synonyms.
Improves over BERTScore by fine-tuning on human evaluation scores (like MQM, DA) to 
predict translation quality with strong human correlation. 
COMET https://aclanthology.org/2020.emnlp-main.213.pdf 
BLEURT https://research.google/blog/evaluating-natural-language-generation-with-bleurt/ 

Page 50:
BERTScore
Introduced in 2020 ICLR Paper: Zhang et. al. BERTScore: Evaluating Text Generation with BERT
Uses pre-trained BERT embeddings to match words in candidate and reference with cosine similarity. 
Shown to correlate with human judgment on sentence-level and system-level evaluation. 
It’s semantic nature can be useful for evaluating language generation tasks (instead of syntactic similarity).
50
https://arxiv.org/abs/1904.09675 

Page 51:
BERTScore (cont.)
Matches each token x to xˆ to compute recall and xˆ to x to compute precision. 
Uses greedy matching to maximize the matching similarity score (each token is 
matched to the most similar token in the other sentence).
F1 is the harmonic mean of the two scores.
51
https://arxiv.org/abs/1904.09675 
Importance Weighting
Rare words can be more indicative of sentence similarity than 
common words.
Inverse Document Frequency (IDF) score can be optionally used 
to put more weight on more rare words.

Page 52:
BLEURT (Bilingual Evaluation Understudy with 
Representations from Transformers)
52
Improves over BERTScore by fine-tuning on human evaluation scores (like MQM, 
DA) to predict translation quality with strong human correlation. 

Page 53:
BERTScore vs. BLEURT
●
BERTScore: Computes cosine similarity between token embeddings in the reference and 
hypothesis.
●
BLEURT: Uses a fine-tuned BERT model trained on human-rated translations, making it 
more aligned with human judgment.
53

Page 54:
COMET (with reference): variation 1
-
Given a hypothesis (prediction) h, a 
reference (answer) r, and a source s as 
inputs 
-
COMET uses a multilingual encoder (i.e. 
XLM-R) to extract the features from the 
inputs.
-
Concatenate them and feed it to a 
feed-forward regressor.
-
The target score can be anything from 
humans, such as HTER or DA. 
54
https://github.com/Unbabel/COMET

Page 55:
COMET (with reference): variation 2
-
With a different architecture, it can also 
rank two different translations
-
Given a worse translation n, a better 
translation p, a reference (answer) r, and 
a source s as inputs 
-
COMET uses a multilingual encoder (i.e. 
XLM-R) to extract the features from the 
inputs.
-
Optimizes on the triplet loss.
55
https://github.com/Unbabel/COMET
What if we don’t have a reference (answer)?
Quality Estimation 

Page 56:
Quality Estimation
An actively researched field where one 
attempts to estimate the quality of a 
translation without access to a reference 
translation.
This is a particularly hard task since neural 
networks are known to often be confident 
while giving wrong answers.
56
https://github.com/Unbabel/COMET
Even the best system does not exactly 
correlate with human judgement

Page 57:
COMET (without reference)
57
-
Everything is the same with the first 
variation of COMET with reference, except 
you don’t give it a reference text.
-
Given a hypothesis h and a source s as inputs 
-
COMET uses a multilingual encoder (XLM-R) to 
extract the features from the inputs.
-
Concatenate them and feed it to a feed-forward 
regressor.
-
The target score can be anything from humans, 
such as HTER or DA. 
https://github.com/Unbabel/COMET

Page 58:
Human Judgement
58
HTER
DA
MQM
Human-targeted Translation Edit Rate: Similar to TER, but instead of using a fixed 
reference translation, it uses an edited version of the machine translation that has been 
manually corrected by a human.
Human annotators score translations on a continuous scale (0-100).
Multidimensional Quality Metrics is a more detailed framework used in professional 
translation assessment, assigning a score for each dimensions.

Page 59:
Human Evaluation
Human judgements of MT quality usually come in the form of segment-level scores, such as:
1.
Human-targeted Translation Edit Rate (HTER) [1]
a.
the MT outputs are manually corrected (#edit)
b.
then the original outputs are compared to the edited ones by computing TER.
2.
Direct Assessment (DA)
a.
A quality score (satisfaction score) of 0 to 100 is given for a translation by human 
3.
Multidimensional Quality Metrics (MQM)
59
[1] https://www.cs.umd.edu/~snover/pub/amta06/ter_amta.pdf

Page 60:
60
https://themqm.org/error-types-2/1_scorecards/values-and-scores/

Page 61:
61

Page 62:
Which Scores to Use?
●
BLEU & ROUGE (syntactic) → If you need a quick, traditional metric.
●
METEOR (syntactic) → If BLEU seems too rigid (captures synonyms), languages with WordNet 
support.
○
Language with WordNet support: 
English, Spanish, French, German
●
ChrF (syntactic) → F1-Score for unsupported languages & morphologically rich languages (Thai)
●
BERTScore / COMET / BLEURT (semantic) → If you care about semantic meaning.
●
Human Evaluation (HTER, MQM, DA) (human) → Best for high-quality MT assessment.
62
METEOR supported languages https://www.cs.cmu.edu/~alavie/METEOR/README.html 

Page 63:
Which Scores to Use?
-
A paper from Microsoft [1] advocates for COMET and ChrF (for automatic metrics). 
63
[1] https://arxiv.org/pdf/2107.10821.pdf 

