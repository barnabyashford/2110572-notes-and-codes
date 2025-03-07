Page 1:
+
Question Answering (QA)
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
Part 1) Introduction
●
Part 2) Traditional QA
●
Part 3) Neural-based QA
●
Part 4) Transformer-based QA
○
1) Encoder, 2) Decoder, 3) Retrieval
○
SOTA: Atlas, RePlug, ChatGPT (not really QA; chatbot)
○
Demo
●
Part 5) QA data sets  (10 data sets)
●
Part 6) Evaluation
2

Page 3:
+
Part 1) Introduction
3

Page 4:
+ What’s Question Answering (QA)?
■QA is a field that combines (1) Information Retrieval, (2) Information Extraction and (3) Natural Language 
Processing.
■We will focus on the NLP part
■The most notable QA software is IBM’s Watson
■Nowadays, QA also plays a significant role in Personal Assistant (Siri, Cortana, etc.)
4
[ Figure by Sandy Jakobs (left), IBM (right) ]

Page 5:
+ Type Of QA
■By application domains
■Restricted Domain
■Open Domain 
■By source of data
■Structured data (Knowledge-based) - e.g. Freebase, Google Knowledge Graph
■Unstructured data (Document)- Web, Wiki
■By answer
■Factoid (single word - when, what, where)
■non-Factoid (e.g., list, how, why)
■The forms of answer
■Extracted text 
■Generated answer
5

Page 6:
+ Type Of QA (cont.)
■Machine Reading Comprehension (MRC)
■
Given a reference and a question
■
Find the answer in the reference text
■OpenQA 
■
Only a question is given
■
Two types of OpenQA
■
“Open-book” QA (LLM with RAG)
■
An external data source can be used, e.g. a document retriever 
■
“Closed-book” QA (just LLM)
■
Use only the knowledge stored inside a model 
6

Page 7:
+ Process Of Traditional QA
■Question Processing
■What type of question?
■Question preprocessing
■Document Processing
■Rank candidate document
■Rank candidate paragraph
■Answer Processing
■Extract candidate answer from paragraph
■Construct an answer
7
[Figure from “The Question Answering Systems: A Survey” ]

Page 8:
+
Part 2) Traditional QA
8

Page 9:
+ Types of QA systems 
Structured Knowledge Base
Unstructured Knowledge Base
9

Page 10:
+ Example of Traditional QA system 
■Open Question Answering Over Curated and Extracted Knowledge Bases (A.Fader SIGKDD 2014)
10
[Figure from “Open Question Answering Over Curated and Extracted Knowledge Bases”]
1
2
3
4

Page 11:
+ Example of Traditional Methods (cont.)
■Open Question Answering Over Curated and Extracted Knowledge Bases (A.Fader SIGKDD 2014)
■1) Paraphrase operator
■
are responsible for rewording the input question into the domain of a parsing operator
■
Source template (open domain) → Target template (predefined format)
11
[Figure from “Open Question Answering Over Curated and Extracted Knowledge Bases”]

Page 12:
+ Example of Traditional Methods (cont.)
■Open Question Answering Over Curated and Extracted Knowledge Bases (A.Fader SIGKDD 2014)
■2) Parsing operator
■
responsible for interfacing between natural language questions and the KB query language
■
Target template (predefined format) → Query
12
[Figure from “Open Question Answering Over Curated and Extracted Knowledge Bases”]

Page 13:
+ Example of Traditional Methods (cont.)
■Open Question Answering Over Curated and Extracted Knowledge Bases (A.Fader SIGKDD 2014)
■3) Query-rewrite operators
■
responsible for interfacing between the vocabulary used in the input question and the internal vocabulary 
used by the KBs
■
Source Query → Target Query (only vocab in knowledge base)
13
[Figure from “Open Question Answering Over Curated and Extracted Knowledge Bases”]

Page 14:
+ Example of Traditional Methods (cont.)
■Open Question Answering Over Curated and Extracted Knowledge Bases 
(A.Fader SIGKDD 2014)
■4) Execution operator
■
responsible for fetching and combining evidence from the Knowledge 
base, given a query
14
[Figure from “Open Question Answering Over Curated and Extracted Knowledge Bases”]

Page 15:
+ Limitation
■Require a lot of time and linguistic knowledge to create a template
■Require many templates for each question type (manual process)
■Can only answer simple factoid question 
15

Page 16:
+
Part 3) Neural-Based QA
16

Page 17:
+ Deep QA models
2015
Memory Network  
[Jason Weston, et al., 2015]
Machine Learning with memory 
component
End-to-End Memory Network 
[Sainbayar Sukhbaatar, et al., 2015]
Use attention mechanism to select 
relevant memory
2015
2016
Attention Sum Reader
[Rudolf Kadlec, et al., 2016]
uses attention to directly pick the 
answer from the context
2016
Stanford Attentive Reader
[Danqi Chen, et al., 2016]
a summary attentive reader
2017
BiDAF
[Minjoon Seo, et al., 2017]
Apply attention ﬂow mechanism 
(context-to-query, query-to-context)
17

Page 18:
+
Reference: Danqi Chen, 2018, https://stacks.stanford.edu/file/druid:gd576xb1833/thesis-augmented.pdf
18
Deep QA models
(cont.)
Transformer-based QA
RoBERTa

Page 19:
+
Human performance has already been surpassed since 2019
19
Human 
performance
March-2023

Page 20:
+ BiDAF from the NLP class in 2020
20
https://www.youtube.com/watch?v=5SH9bpQ33Xk&list=PLcBOyD1N1T-NP11DsVK9XcN54rvfGBb96&index=8 

Page 21:
+
Part 4) Transformer-Based QA
1) Encoder, 2) Decoder, 3) Retrieval
SOTA: Atlas, RePlug, ChatGPT (not really QA; chatbot)
Demo
21

Page 22:
+ 1. Encoder-Based
■
Uses any pretrained encoder-based like 
BERT 
■
Adds 2 linear layers to classify each token 
as the start and end indices
■
Requires a reference text
22
https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/ 
The start of the answer span
BERT

Page 23:
+ 2. Decoder/Encoder-Decoder Based
■
Generates the answer instead of trying to predict the start/end index
■
Uses knowledge inside a model to answer questions. For example, ChatGPT can 
answer questions without needing a reference text.
■
Although given reference text can improve the performance.
■
More practical!
23

Page 24:
+ 3. Retrieval-Augmented Model
■
Use a retrieval engine and a language model.
■
The retrieval engine fetches a list of documents
■
And the LM uses the list as reference text.
24
REALM 1  (encoder)
RAG 2 (decoder)
1 Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. REALM: retrieval-augmented language model pre-training. In Proceedings of the 37th 
International Conference on Machine Learning (ICML'20). JMLR.org, Article 368, 3929–3938.
2 Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian 
Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Proceedings of the 34th International Conference on Neural Information 
Processing Systems (NIPS'20). Curran Associates Inc., Red Hook, NY, USA, Article 793, 9459–9474.

Page 25:
+ SOTA
25
Izacard et al.(2022). Atlas: Few-shot Learning with Retrieval Augmented Language Models
■
Retrieval-augmented models are more efficient at knowledge-intensive tasks
■
However, they are usually more computationally expensive due to the retrieval step.
11B
540B
Retrieval 
models
Decoder 
models
Retrieval models 
with LLMs
GPT3 (API; off-the-shelf model) + retrieval
Shi et al. (2023). REPLUG: Retrieval-Augmented Black-Box Language Models

Page 26:
+ SOTA (cont.)
■
Atlas (retrieval-based QA) [fine-tune end-to-end, both IR & generator]
■
RePlug (retrieval + decoder-based QA) [fine-tune only IR]
■
ChatGPT (not really QA; chatbot) [no fine-tuning]
26

Page 27:
+ 1) Atlas = Contriever + FiD 
[Retrieval-based model by Meta in NeurIPS 2022]
Contriever (a transformer encoder model; contrastive retriever) is a dense retriever 
trained using contrastive learning. It uses BERT-like embeddings for document 
retrieval.
During inference, it accepts a query and encodes it into a fixed-length embedding. 
The dot-product of the query embedding and a document embedding returns how 
relevant a document is to the query. Top-k most similar documents are returned.
27
Contriever
Document
Query
Similarity 
score
G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, E. Grave Unsupervised Dense Information Retrieval with Contrastive Learning 2022

Page 28:
+ 1) Atlas = Contriever + FiD 
[Retrieval-based model by Meta in NeurIPS 2022]
Fusion in Decoder (a transformer encoder-decoder model) is a generative model 
for open QA. It uses a T5-like model with an encoder-decoder structure.
■
The model accepts a question and documents as inputs. The encoder 
independently encodes the documents (+question). The resulting 
representations are concatenated and finally given to the decoder to try to 
generate the correct answer. This allows the fusion of information from multiple 
documents (thus the name).
■
Contriever and FiD are trained together, allowing the retriever to be directly 
influenced by the generator’s feedback.
28
G. Izacard, E. Grave Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering 2021

Page 29:
+ 2) RePlug: Retrieval-Augmented Black-Box Language 
Models [Retrieval + decoder-based model][ICLR, 2024]
■
It was developed by Tencent AI, proposed in 2023 and published at ICLR 2024.
■
Given a black-box LM (such as OpenAI’s GPT API), the work attempts to add a 
tunable retrieval model to improve the LM QA performance by retrieving 
documents that boosts the probability of generating the correct answer.
■
The retriever is fine-tuned using the feedback from the frozen LM. The retrieval 
likelihood is adjusted so that documents that actually help the LM answer the 
question get chosen more frequently while the unhelpful documents get 
penalized.
29
REPLUG: Retrieval-Augmented Black-Box Language Models. Shi et al., 2023

Page 30:
+
30

Page 31:
+
31
GPT3 as LM 
(frozen)
Fine tune

Page 32:
+ 3) ChatGPT
ChatGPT is the chatbot that is made by finetuning GPT-4 to follow instructions and 
then further fine-tuned with RL algorithm to improve its helpfulness. Note that the 
knowledge in the model is mainly learned from the pretraining stage. The fine-tuning 
stages are only to align the model output with human preference. 
32
https://openai.com/blog/chatgpt

Page 33:
+ QA is still an open problem
33
https://www.reuters.com/technology/google-ai-chatbot-bard-offers-inaccurate-information-company-ad-2023-02-08/ 
In Google’s advertisement for Bard AI, the 
chatbot was asked:
“What new discoveries from the James Webb 
Space Telescope (JWST) can I tell my 
9-year-old about?”
Bard’s incorrect response: It stated that JWST 
was the first telescope to capture images of a 
planet outside our solar system (an 
exoplanet).
The reality: The first exoplanet image was 
captured in 2004 by the Very Large Telescope 
(VLT) in Chile, not JWST.

Page 34:
+ QA is still an open problem (cont.)
Yes, QA remains an open problem!
■
1. Hallucination and Misinformation
■
Current research focuses on retrieval-augmented models (e.g., RePlug, RAG-2) 
to mitigate this.
■
2. Lack of Real-Time Knowledge
■
Solutions like real-time search augmentation (e.g., Perplexity AI, Bing AI) are 
improving this.
■
3. Ambiguity and Context Understanding
■
Multimodal AI (text + image + reasoning models) aims to improve this.
■
4. Long-Context Reasoning and Multi-Hop QA
■
Long-context LLMs (e.g., Claude-2.1, GPT-4 Turbo) and retrieval-augmented 
architectures aim to improve this.
34

Page 35:
+ Demo: QA (AllenNLP - outdated)
https://demo.allennlp.org/reading-comprehension
35

Page 36:
+ Demo - Huggingface
36
https://huggingface.co/spaces/keras-io/question_answering 

Page 37:
+
Part 5) QA data sets (10 data sets)
37

Page 38:
+ QA Datasets
38
Modern Question Answering Datasets and Benchmarks: A Survey. Wang, 2022

Page 39:
39
Dataset and Predominant Techniques
Year
Tasks
Corpus  Type
Question Type
Answer Source
Answer Type
2018
SQuAD 2.0
Textual
Natural
Spans
Natural
2019
Natural Questions
Textual
Natural
Spans
Natural
2018
CoQA
Textual
Natural
Free-form
Natural
2017
TriviaQA
Textual
Natural
Free-form
Natural
2019
RACE
Textual
Natural
Free-form
Multiple-choice
2018
RecipeQA
Multi-modal
Natural
Cloze/free-form
Multiple-choice
2019
NSC (Thai)
Textual
Natural
Spans
Natural
2017
VQA
Multi-modal
Natural
Free-form
Natural
2022
MMCoQA
Multi-modal
Natural
Free-form
Natural

Page 40:
1) SQuAD 2.0
https://rajpurkar.github.io/SQuAD-explorer/ 
•
Extend from SQuAD 1.0
•
Crowdsource from Wikipedia paragraph (let worker create question from articles)
•
Has two types of questions
– Answerable (in SQuAD 1.0)
– Unanswerables (only in SQuAD 2.0)
•
Arguably, one of the most popular and well-known MRC benchmarks.
•
The top of the leaderboards is dominated by variations of pretrained LMs.
40
Year
Tasks
Corpus  
Type
Question 
Type
Answer 
Source
Answer 
Type
2018
SQuAD 2.0
Textual
Natural
Spans
Natural

Page 41:
41
●
Includes both answerable and 
unanswerable questions ("is_impossible": 
true for no-answer cases).
●
Provides answer span locations 
("answer_start").
●
Designed to evaluate reading 
comprehension and model robustness.

Page 42:
1) SQuAD 2.0
https://rajpurkar.github.io/SQuAD-explorer/ 
• Leaderboard (March-2023): RetroReader
42
- Very saturated
- Already beat human performance
Year
Tasks
Corpus  
Type
Question 
Type
Answer 
Source
Answer 
Type
2018
SQuAD 2.0
Textual
Natural
Spans
Natural

Page 43:
1) SQuAD 2.0 – RetroReader, (Zhang et al., EMNLP2020)
https://arxiv.org/abs/2001.09694 
•
Built upon pretrained LM architecture rather than just ﬁne-tuning pretrained LM on the MRC task.
•
Has external veriﬁcation and internal veriﬁcation
– Sketchy reader: contains answer or not 
– Intensive reader: ﬁnd answer spans as well as answerability 
•
Interesting choice of design of having to answer veriﬁcation (check if the context passage contains an answer or not) in both of the reader 
modules
43
Pretrained LM 
of choice

Page 44:
RetroReader [EMNLP2020] (cont.)
1. Global Evidence Selection (Retriever Stage)
•
Identiﬁes the most relevant spans from the passage.
•
Uses a bidirectional interaction mechanism to highlight important text.
2. Local Reading (Extractor Stage)
•
Extracts an initial answer span from the retrieved evidence.
3. Decoder (Veriﬁcation Stage) 
•
Veriﬁes if the extracted answer is correct by re-reading the passage.
•
Rejects low-conﬁdence answers (especially for unanswerable questions in datasets like SQuAD 2.0).
44

Page 45:
2) Natural Questions (NQ)
https://ai.google.com/research/NaturalQuestions/ 
• Based on Wikipedia article, context passage consists of 5 top Wikipedia article queried based on the natural question 
• Has 2 types of tasks: long and short answers
– Long answer: ﬁnd the paragraph that contains the answer
– Short answer: ﬁnd answer if present in the document
• Some questions are also unanswerable (but only in small percentage)
45
Year
Tasks
Corpus  
Type
Question 
Type
Answer 
Source
Answer 
Type
2019
Natural
Textual
Natural
Spans
Natural

Page 46:
2) Natural Questions
https://ai.google.com/research/NaturalQuestions/ 
• Leaderboard (March-2023): Atlas
46
Year
Tasks
Corpus  
Type
Question 
Type
Answer 
Source
Answer 
Type
2019
Natural
Textual
Natural
Spans
Natural

Page 47:
3) CoQA
https://stanfordnlp.github.io/coqa/ 
•
One of the ﬁrst conversational MRC datasets available, mimicking the process of 2 people discussing the context passage as a topic.
•
Covers 7 different domains, including Wikipedia, news articles, literature, and children’s stories to test model generalization.
•
Contains: Yes, no, unanswerable question type, so answers are not guaranteed to be found in the passage
•
Also has rationale label for each question (metadata)
47
Year
Tasks
Corpus  Type
Question 
Type
Answer 
Source
Answer Type
2018
CoQA
Textual
Natural
Free-form
Natural
In some questions, information can be 
found from the previous conservation 
turn.

Page 48:
3) CoQA
https://stanfordnlp.github.io/coqa/ 
•
Leaderboard (March-2023): RoBERTa with enhanced techniques
48
Year
Tasks
Corpus  Type
Question 
Type
Answer 
Source
Answer Type
2018
CoQA
Textual
Natural
Free-form
Natural

Page 49:
4) TriviaQA
http://nlp.cs.washington.edu/triviaqa/ 
•
Questions are curated from trivia questions website, and then the questions are paired with the closest matched document later (use 
a search engine).
•
Since questions are not crafted from the document, 
– Not all questions are guaranteed to have answer, and
– The answers that are found in the context passage might not exactly match with the semantic of the question
•
Has 2 versions of the dataset: 
– The one that is matched with the document 
(with provided documents).
– Open-Domain QA where questions are not matched
 with the article (without provided documents).
49
Year
Tasks
Corpus  
Type
Question 
Type
Answer 
Source
Answer Type
2017
TriviaQA
Textual
Natural
Free-form
Natural

Page 50:
4) TriviaQA
http://nlp.cs.washington.edu/triviaqa/ 
•
Leaderboard
50
Year
Tasks
Corpus  
Type
Question 
Type
Answer 
Source
Answer Type
2017
TriviaQA
Textual
Natural
Free-form
Natural
Leaderboard (March-2023): PaLM

Page 51:
5) RACE
https://www.cs.cmu.edu/~glai1/data/race/ 
•
Dataset collected from English examination for Chinese students
•
Many questions require reasoning ability
51
Year
Tasks
Corpus  Type
Question Type
Answer Source
Answer Type
2019
Textual
Natural
Free-form
Multiple-choice
Natural

Page 52:
5) RACE
https://www.cs.cmu.edu/~glai1/data/race/ 
•
Leaderboard: Methods with pretrained language models dominate the 
leaderboard (e.g., ALBERT).
52
-
Calculate the score for each answer candidate separately
-
https://openai.com/blog/language-unsupervised/ 
Year
Tasks
Corpus  Type
Question Type
Answer Source
Answer Type
2019
Textual
Natural
Free-form
Multiple-choice
Natural

Page 53:
6) RecipeQA
https://hucvl.github.io/recipeqa/ 
•
RecipeQA Dataset: A Multimodal Question Answering Dataset for Cooking Instructions
•
RecipeQA is a multimodal question answering (QA) dataset designed to evaluate an AI system’s ability to understand and reason 
about procedural cooking instructions using text, images, and step-by-step instructions.
•
Introduced by: Hacettepe University & Allen AI (2019)
•
1. Multimodal QA
–
Questions are based on text, images, and step-by-step cooking instructions.
–
Helps evaluate AI comprehension of procedural tasks.
•
2. Four QA Tasks
–
Textual Cloze: Fill in missing words in a cooking step.
–
Visual Cloze: Identify the missing image in a sequence.
–
Visual Coherence: Identify the wrongly placed image in a cooking step sequence.
–
Visual Ordering: Arrange images in the correct order of a recipe.
•
3. Large-Scale Dataset
–
36,000+ question-answer pairs from 2,600+ unique cooking recipes.
53
Year
Tasks
Corpus  Type
Question 
Type
Answer Source
Answer Type
2018
RecipeQA
Multi-modal
Natural
Cloze/free-form
Multiple-choice

Page 54:
6) RecipeQA
https://hucvl.github.io/recipeqa/ 
•
Four QA Tasks
54
Year
Tasks
Corpus  Type
Question 
Type
Answer Source
Answer Type
2018
RecipeQA
Multi-modal
Natural
Cloze/free-form
Multiple-choice

Page 55:
6) RecipeQA
https://hucvl.github.io/recipeqa/ 
https://docs.google.com/presentation/d/1mvuu4QTfOP6CHUfbLdXMisleiH7kIlxpsU1bgyT0oTw/edit#slide=id.g468e9cafc5_0_281 
55
 
Year
Tasks
Corpus  Type
Question 
Type
Answer Source
Answer Type
2018
RecipeQA
Multi-modal
Natural
Cloze/free-form
Multiple-choice

Page 56:
7) NSC (Thai QA)
http://copycatch.in.th/thai-qa-task.html 
•
Thai Question Answering Program competition hosted in NSC by NECTEC.
•
There were 2 round of competitions
– First round: 4,000 factoid (span extraction) questions
– Second round: 15,000 factoid questions and 2,000 yes-no questions
•
An open-domain question-answering problem: the program must also query for the context passage.
•
Only the ﬁrst round of the competition dataset went public.
56
Year
Tasks
Corpus  
Type
Question 
Type
Answer 
Source
Answer Type
2019
NSC (Thai)
Textual
Natural
Spans
Natural

Page 57:
8) iApp (Thai QA)
•
Another Thai QA dataset is IAPP wiki QA (2021)
– https://github.com/iapp-technology/iapp-wiki-qa-dataset
•
Thai Wikipedia Question Answering Dataset.
–
1,961 Documents
–
9,170 Questions
–
It is organized and formatted in the SQuAD format
•
Demo: https://ai.iapp.co.th/control/ai 
57

Page 58:
9) VQA [CVPR2017]
One of the most widely used multimodal datasets from Virginia Tech, 
composed of two parts: 
•
VQA-real: natural images 
•
VQA-abstract: cartoon images 
VQA-real comprises 123,287 training and 81,434 test images, sourced from 
COCO. 
Human annotators were encouraged to provide interesting and diverse 
questions.
Overall, it contains 614,163 questions, each having 10 answers from 10 
different annotators.
58
https://visualqa.org/ 
Visual question answering: A survey of methods and datasets 
https://www.sciencedirect.com/science/article/abs/pii/S1077314217300772 
VQA-real
VQA-abstract

Page 59:
10) MMCoQA [ACL2022]
59
https://aclanthology.org/2022.acl-long.290/ 
MMCoQA: Conversational Question Answering over Text, Tables, and 
Images 
MMConvQA contains 1,179 conversations and 5,753 QA pairs. There are 
4.88 QA pairs on average for each conversation (multiple turns).
The multimodal knowledge collection consists of 218,285 passages, 
10,042 tables, and 57,058 images.
Each question is annotated with the related evidence (a table, an image, 
or a passage in the knowledge collection) and a natural language answer.

Page 60:
+
Part 6) Evaluation
60

Page 61:
Evaluation
Syntactic Similarity
61
Semantic Similarity
Human Judgement
BLEU
ROUGE
METEOR
BERTScore
COMET
BLEURT
HTER
DA
MQM
Exact Match
LLM as a Judge
CHIE
Naive
Automatic Evaluation?

Page 62:
Exact Match
Unlike translation, there are multiple-choice/short answers in QA that is structured and fixed.
In these cases, it may be appropriate to use exact match.
62
ChatGPT answering multiple-choice questions
pred == “Chuangsuwanich”

Page 63:
LLM as a Judge
Automatic evaluation metrics in syntactic or semantic similarity 
fail to capture the different quality dimensions that a human 
can distinguish. For instance, an answer can be:
●
Not grounded in context
●
Repetitive
●
Grammatically incorrect
●
Excessively lengthy
●
Incoherent
Human assessment is more accurate but costly. 
💡 Ask an LLM to do the grading! 🤖✓ → Introduced in 2023 
Zheng et. al. Judging LLM-as-a-Judge with MT-Bench and 
Chatbot Arena. 
63
https://huggingface.co/learn/cookbook/en/llm_judge 
Judging LLM-as-a-Judgewith MT-Bench and Chatbot Arena https://arxiv.org/abs/2306.05685 
https://www.deepchecks.com/what-is-llm-as-a-judge-strategies-impact-and-best-practices/ 
Can Large Language Models Be an Alternative to Human Evaluations?
https://arxiv.org/abs/2305.01937 

Page 64:
LLM as a Judge (cont.)
However, naively prompting the model to evaluate the score (1-10) might 
not be a good idea (LLMs are not good at continuous scales).
64
https://x.com/aparnadhinak/status/1748368364395721128 

Page 65:
LLM as a Judge (cont.)
1) Need a human evaluation dataset to see the agreement (i.e. Pearson correlation). For example:
65
MT Bench
https://huggingface.co/learn/cookbook/en/llm_judge 
Judging LLM-as-a-Judgewith MT-Bench and Chatbot Arena https://arxiv.org/abs/2306.05685 
2) Revise your prompt with best practices and intuition.
i.e. 
⏳ Leave more time for thought by adding an Evaluation field before the final answer.
🔢 Use a small integer scale like 1-4 or 1-5 instead of a large float scale.
󰠅 Provide an indicative scale for guidance.
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the 
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpf
completely and helpfully addresses the user_question.
Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very p
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the con
Provide your feedback as follows:
Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)
You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
Now here are the question and answer.
Question: {question}
Answer: {answer}
Example LLM as a Judge Prompt
80 MT-bench questions, 3K expert votes, and 30K 
conversations with human preferences
Chatbot Arena
A crowdsourced platform featuring anonymous battles 
between chatbots

Page 66:
LLM as a Judge (cont.)
A paper 2025 Gu et. al. A Survey on LLM as a Judge compared various LLMs on LLMEval 
(2,553 samples from multiple data sources) with a percentage agreement metric. 
66
A Survey on LLM-as-a-Judge https://arxiv.org/abs/2411.15594 

Page 67:
67
There are frameworks that are 
proposed for LLM as a Judge.
CHIE is a framework for multi-aspect 
evaluation on 
●
Correctness,
●
Helpfulness, 
●
Irrelevance, 
●
Extraneousness.
Uses binary categorical values 
rather than continuous rating scales.
https://aclanthology.org/2024.genbench-1.10/ 
LLM as a Judge: CHIE [ACL2024]

Page 68:
68
●
Correctness,
●
Helpfulness, 
●
Irrelevance, 
●
Extraneousne

