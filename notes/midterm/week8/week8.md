Page 1:
Large Language Model
1

Page 2:
Outline
•
Generative AI
•
Emergent properties & scaling laws
•
Alignment
•
Instruction tuning
•
Preference learning
2

Page 3:
What is generative AI?
●Predictive AI
○predicts
○classify
https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1/
3

Page 4:
https://knowyourmeme.com/memes/ai-will-smith-eating-spaghetti
What is generative AI?
●Generates
○
Creative tasks that was believe to be hard for AI a couple years 
ago
https://x.com/jerrod_lew/status/1868809004400754871
https://www.tiktok.com/@willsmith/video/7337480720115371295?lang=en

Page 5:
Generative modeling
•
Learns P(X,Y) or P(X|Y)
•
Can sample (generates) X from P(X|Y)
•
Y is the controlling parameter
•
What is P(X|Y)?
•
In the past, it’s often assumed to be a parametric distribution
5

Page 6:
Enter deep generative models
•
Distribution learning landscape
https://arxiv.org/pdf/1701.00160.pdf
https://yang-song.net/blog/2021/score/
Explicit likelihood 
maximization of P(X)
Some other way besides max P(X)
Diffusion/score-based model
6

Page 7:
Text generation
•
Unlike other generative tasks, turns out that explicit 
density methods (autoregressive prediction) works well 
for text
•
There are attempts to use GANs/RLs/Diffusion/VAE for 
text generation but they are not as competitive against 
simple cross entropy based prediction
•
Scales well with very large data
•
Recent research starts looking into ways to use diffusion/VAE 
in text generation
7

Page 8:
Pretrained Foundation Models/LLM
•
With self-supervised learning, Ex. contrastive learning, 
next token prediction, mask token prediction
•
People found that large model produces better results
LLaMA: Open and Efficient Foundation Language Models https://arxiv.org/abs/2302.13971
8
https://arxiv.org/pdf/2402.08797

Page 9:
Emergent abilities
•
People have found that models perform some 
capabilities much better after a certain size
9
Emergent Abilities of Large Language Models https://arxiv.org/abs/2206.07682

Page 10:
Why large model matters?
•
In GPT3 paper, OpenAI found that large models are 
good few-shot learners
https://arxiv.org/abs/2005.14165
Tradition Fine-Tuning (BERT)
In-Context Learning (GPT-3)
Zero-Shot
One-Shot
Few-Shot
In-Context Learning (GPT-3)
Model
●
No Fine-Tune
Prompt
●
Task Description
●
Example
●
Prompt
10

Page 11:
Why large model matters
Text Input
Prompt
Dataset: BoolQ
Zero-Shot
https://arxiv.org/abs/2005.14165
11

Page 12:
Why large model matters
On par with 
BERT-Large
https://arxiv.org/abs/2005.14165
12

Page 13:
Why large model matters
General Model: 
On average, the 
accuracy in most tasks 
are not as good
Specific Model:
Fine-tune model 
towards a specific task
Fine-Tune
https://arxiv.org/abs/2005.14165
13

Page 14:
Scaling is all you need
•
This realization of scale leads to several efforts to try to 
understand the relationship between data, compute, 
and performance.
14

Page 15:
Kaplan Scaling law
•
In 2020, OpenAI release the first paper discussing the 
“optimal model size” 
•
An empirical study, think of Scaling law as Moore’s law, just an 
observation, on the log scale
•
Performance depends mostly on scale rather than other factors 
such as width or depth of network
•
Need to scale data, compute, and network size in tandem
15
Scaling Laws for Neural Language Models https://arxiv.org/pdf/2001.08361

Page 16:
Kaplan Scaling law
•
Large model are more sample-efficient.
•
Less gradient updates to reach the same performance.
•
Models converge slowly
•
Stop when you just want to stop. Late iterations give less gains.
16

Page 17:
Kaplan Scaling law
•
They curved fit to the data and got some dubious 
equations
•
In simple implications, for GPT-3, you need 1.7 text tokens per 
parameter in you LLM to be efficient
•
Often refer to as Kaplan scaling law (the first author)
17
Relationship with infinite size model
Relationship with infinite size data

Page 18:
Chinchilla Scaling law
•
In 2022, Deepmind 
published another scaling 
law paper
•
Found that model size and 
training data should be 
scaled equally. Around 20 
tokens per parameter.
•
Used smaller model, but 
trained with more data to 
get better performance
18
https://arxiv.org/abs/2203.15556

Page 19:
Chinchilla Scaling Law
•
They offer a method to think about the data size D and 
model size N that will gives compute optimal (given a 
fixed compute budget) as
19
What happens with a finite model
What happens with a finite data
Inherent loss (noise)
With experiments from OpenAI, A = 406.4, 
B = 410.7, E = 1.69
Alpha = 0.34, Beta = 0.25
If we plug in gopher numbers, L = 1.993
Chinchilla L = 1.936

Page 20:
Other Scaling Laws
•
Broken Neural Scaling laws 2023 
https://arxiv.org/abs/2210.14891
•
A work offering scaling laws that cover various tasks (besides 
NLP)
•
The laws so far are for training time, we can also 
investigate inference time scaling (test-time scaling)
20

Page 21:
Outline
•
Generative AI
•
Emergent properties & scaling laws
•
Alignment
•
Instruction tuning
•
Preference learning
21

Page 22:
Alignment
22

Page 23:
Back to prompting
Prompt Design
Input Text:A great movie!!! 
Prompt:
Q: Is it positive or negative? 
A: <blank>
Freeze
Pre-trained 
Model
Freeze
Positive
Input to model
Finetuning
Input data: A great movie!!! -> Positive
Waste of time -> Negative
Pre-trained 
Model
Finetune
Finetuned
Model
Input data: Fabulous
Positive
23

Page 24:
Verbalization and prompt design
Different Template
Different Verbalizer 
https://arxiv.org/pdf/2104.08691.pdf
24

Page 25:
Why do we even need 
Prompting?
25

Page 26:
Self-supervised models are 
stochastic parots
Input (Prompt)
Output
The patient was died. <fill next…>
The patient's body was found in a dark alley 
behind the hospital’s…
“The patient was died.” correct this 
<fill next…>
claim if you really believe such figures….
Poor English input: The patient was 
died. <fill next…>
Good English output: The patient died.
26

Page 27:
Alignment
•
It’s hard to make a self-superivsed model do what we 
want. This is called the alignment problem (aligns model 
capabilities with users’ interests). Ultimately, you will 
need supervision for this!
•
Current solutions
•
Prompt engineer (manual)
•
Low parameter finetuning – LoRa, mixture of experts, learned 
prompts, etc.
•
Preference learning
https://www.alignmentforum.org/
Web forum discussing the alignment problem
27

Page 28:
InstructGPT
•
In Mar 2022 (ChatGPT came out Dec 2022), OpenAI 
released a paper InstructGPT
28
https://arxiv.org/pdf/2203.02155

Page 29:
InstructGPT
•
Step 0: pretrain a language model eg GPT3
•
Step 1: Supervised fine-tuning (SFT)
•
Step 2: Reward Model training
•
Step 3: Reinforcement Learning with Human feedback 
(RLHF)
29
https://arxiv.org/pdf/2203.02155

Page 30:
Supervised Finetuning/Instruction 
Tuning
•
Create a dataset of questions and 
answers.
•
Often called “Instruction data”
30

Page 31:
Instruction data
•
Just example Q A pairs
•
Alpaca is one of the 
earliest open instruction 
data. They just use 
ChatGPT to generate 
the answers.
•
A kind of distillation
•
Easier and faster than 
hiring a bunch of humans. 
People can copy you! 
OpenAI terms has a non-
compete clause.
•
Deepseek uses 1.5 
million
31
https://huggingface.co/datasets/iamketan25/al
paca-instructions-dataset
https://github.com/tatsu-lab/stanford_alpaca

Page 32:
Self-instruct
•
A technique to use LLM to generate data to train more 
LLMs!
32
https://arxiv.org/pdf/2212.10560

Page 33:
Thai instruction data
•
Still super small, most are automatically generated
•
OpenThaiGPT
33
https://openthaigpt.aieat.or.th/previous-versions-and-resources/released-datasets-14-04-23

Page 34:
Thai instruction data
•
WangchangX
34
https://huggingface.co/datasets/airesearch/wa
ngchanx-seed-free-synthetic-instruct-thai-
120k
https://huggingface.co/datasets/airese
arch/WangchanThaiInstruct_7.24

Page 35:
Supervised Finetuning/Instruction 
Tuning
•
Create a dataset of questions and 
answers.
•
Often called “Instruction data”
•
Trained using next word prediction 
on instruction data.
35

Page 36:
InstructGPT
•
Step 0: pretrain a language model eg GPT3
•
Step 1: Supervised fine-tuning (SFT)
•
Step 2: Reward Model training
•
Step 3: Reinforcement Learning with Human feedback 
(RLHF)
36
https://arxiv.org/pdf/2203.02155

Page 37:
37

Page 38:
Introduction to RL
38

Page 39:
RL problem
Observations
Actions
Environment
Agent
Reward

Page 40:
3 Modes of Learning
Supervised Learning
Reinforcement Learning
Unsupervised Learning

Page 41:
3 Modes of Learning
Reinforcement Learning (RL)
●Observe:  
○The states (x1, x2, x3, … )
○The reward ( r1, r2, r3, … ) 
●Can also take actions
○a1 , a2 , a3 , … 
●What are the best actions? 
○Such that we will receive 
highest accumulative 
rewards

Page 42:
Reward (rt)
State (st) 
Action (at)
RL framework

Page 43:
Rewards-based learning
• Maximise the rewards
• Can we design any desired
behaviour with reward?
Rt = Δdistance
Rt = score
, win
, lose

Page 44:
Policy
• Policy = a mapping from a state to an action
• Objective of RL is to find the ‘’optimal’’ policy! 
Can be either 
deterministic or 
stochastic 

Page 45:
Policy
Example: tabular policy
+1
-1

Page 46:
Policy
•
Example: policy to play an arcade game
46

Page 47:
Return (Cumulative rewards)
• Return =  cumulative rewards with discount
rt = 0
rt+10 = 1
rt+34 = -1

Page 48:
What is learning?
•
Use data to find/search for the best policy 
What is the best policy?
•
Policy that give us the highest expected return! 

Page 49:
Two main ways to find the best 
policy 
•
Q-learning (Value-based)
•
Policy gradient (Policy-based)
49

Page 50:
Q-learning algorithm
•
Let’s define a state value as
•
Expectation of the return after visit s and follow 𝛑
•
Let’s define a state-action value (Q-value) as
•
Expectation of the return after visit a state s, take action a

Page 51:
Q-learning algorithm
•
There exist an optimal value function associate with an 
optimal policy, 
•
The optimal policy is the policy that achieves the 
highest value for every state

Page 52:
Q-learning algorithm
•
It follows that 
•
and ..
•
Optimal actions can be found indirectly through Q-value 

Page 53:
Policy gradient
Q Learning
-
policy is implicit
-
if we already have Q, we have policy
-
we just look at Q to get 
-
Used in AlphaGo
Policy gradient
-
learns      directly explicitly 
-
use Q, V as a helper for learning
-
Used in pretty much everything else 

Page 54:
Policy gradient
●Use Function Approximator to represent policy directly
S
a
S
P(a1| s)
P(a2| s)
P(a3| s)

Page 55:
Loss function for policy gradient
•
Start-state objective  
•
Average-reward objective
* d is a stationary distribution of a Markov chain.
𝞹

Page 56:
Policy gradient
Let’s start from the average-reward objective
For simplicity let’s assume d(s) does not depend on
Almost there...

Page 57:
Policy gradient
REINFORCE trick! 
We get this by sampling a playthrough using current policy (on-policy)

Page 58:
Policy gradient theorem
There is a theorem…called policy gradient theorem
say that we can replace                 with 

Page 59:
What is the gradient doing?
Goal maximize rewards
Push 𝝿towards directions of higher Q(s,a)
Q(1,1) = 5  
Q(1,2) = 2
▿𝝿(1,2) will have a higher weight. Policy gets push towards 
action 2 

Page 60:
Notes on Policy gradient
Also known as REINFORCE or likelihood ratio.
Used by other ML fields when original loss is not 
differentiable (-Q in this case). Push network to produce 
lower loss.
Example of non-differentiable functions
argmax
(not maxpool)
sampling
Many metrics such as accuracy, preference (ranking)

Page 61:
Baselines
has high variance
is very noisy
Slow down training a lot! 
We need to reduce variance  to speed up the training

Page 62:
Baselines reduce variance
What is a good b(s)?
is a convenient choice

Page 63:
Advantage Function
When we use V(s) as baseline:
We call this the advantage function. 
It tells the relative value of the actions.
Lower variance than using absolute value of the actions.
Also called A2C

Page 64:
Noisy updates
•
In RL, the updates can be noisy. Our learning is based 
on Q, V, and A and might not be an accurate estimate if 
the policy changes too much.
•
Limit the update to a certain size? Use KL divergence 
that new policy shouldn’t be that different from old policy
64
Thus, we want to make 1-e <        < 1+e 
(clipping the value)

Page 65:
Proximal Policy Optimization (PPO)
•
PPO is just a policy gradient update with Advantage 
function and clipping the value so that the model does 
not move outside of the trusted region
•
The min is used as a pessimistic bound between no 
clipping and clipped values.
•
PPO is one of the defaults techniques used by OpenAI
65
https://arxiv.org/pdf/1707.06347

Page 66:
Recap PPO and RL
•
RL uses a reward function to learn to optimize our 
model.
•
Key ingredient: environment (game = input+output) and a way 
to give reward (scores from output)
•
PPO is a policy gradient method with tricks.
•
Key: what moves give me higher reward than before, try to 
change the gradients so that the move is more likely
66

Page 67:
InstructGPT
•
Step 0: pretrain a language model eg GPT3
•
Step 1: Supervised fine-tuning (SFT)
•
Step 2: Reward Model training
•
Step 3: Reinforcement Learning with Human feedback 
(RLHF)
67
https://arxiv.org/pdf/2203.02155

Page 68:
Reward model
•
We need a way to give 
rewards for any possible 
input+output pairs.
•
Technically can be a metric we 
learned last week, but we talked 
about how human judgement is 
still the best
•
We also talked about training a 
model to mimic human 
judgement, and how chatGPT 
can gives scores
•
Collect data to train a reward 
model.
•
OpenAI uses GPT -> finetune -> 
reward model
68

Page 69:
ถ้าคุณมีเพื่อนสนิทหรือคนในครอบครัวที่ชอบอะไรคล้ายๆกัน 
ลองหาของที่เค้าชอบมาจับสลากดูสิคะเช่น ตุ๊กตา ของเล่น 
หนังสือ เสื้อผ้า เครื่องประดับ ฯลฯ รับรองว่าถูกใจทั้งผู้ให้และ
ผู้รับแน่นอนค่ะ
[งบ 400-500 บาท] ขอแนะน าของขวัญจับฉลากที่ดีที่สุดคะ เช่น 
ฉลากสีขาว ฉลากเงิน ฉลากแข็ง หรือฉลากขนาดใหญ่ที่มี
ส่วนผสมของฉลากเล็กๆ ที่มีรูปแบบที่สวยงามและมีความ
สวยงามมากมาย เช่น เครื่องหมายความส าคัญที่สุดในการจัด
ฉลาก เช่น เสื้อผ้าที่มีสีสันสดใส และฉลากสีเหลืองที่มีความ
แตกต่าง
Q: ขอค ำแนะน ำคะ ซื้อของขวัญจับฉลำกอะไรดี งบ 400-500 บำทค่ะ
A
B
Preference data
69
Preference data needs to be collected for the reward model

Page 70:
Preference data
•
LLAMA – ask annotators to write a prompt -> two 
sampled model response -> label them (“chosen” and 
“rejected”)
•
LLAMA3 added a “edited” response, where chosen is further 
editted
70
https://ai.meta.com/research/publications/the-llama-3-herd-of-models/

Page 71:
Reward model training
•
Many possible settings
•
Contrastive loss
•
R( x, y_chosen) > R( x, y_rejected)
•
Predict some score value
•
Use a heuristic
•
Deepseek R1 is trained on code and math problems.
•
Scores can be checked against the real solution.
71

Page 72:
Reinforcement learning with human 
feedback (RLHF)
•
The Reward model is used to 
trained with PPO
•
They also add a per-token KL 
penalty from the SFT model so that 
the optimized model does not go 
too far away from the SFT model
72

Page 73:
Reinforcement learning with human 
feedback (RLHF)
•
The Reward model is used to 
trained with PPO
•
But, let’s think again why do we 
need RL?
•
This task is hard to do supervised 
learning
•
Cannot think of all possible answer 
and scoring for creative tasks
•
RL offers a scalable solution that can 
covers multiple domain
•
Can discover new moves
•
“move 37”
•
Deepseek aha moment
•
Drawbacks: sophisticated framework. 
Hard to train
73

Page 74:
Deepseek v3 aha moment
•
Probably won’t show up in supervised data
74
https://arxiv.org/abs/2501.12948

Page 75:
Notes on LLM
•
From what we learned so far
•
LLM is autoregressive
•
Errors cascade
•
LLM is just next word prediction trained on trillions of tokens
•
Likes most probable next token, with some randomness mixed in
•
Trained to reward human preference
•
Can you see why hallucination is pretty much inherent 
in LLMs?
•
Backtracking might be an important capability for LLMs 
to do well in this setting.
75

Page 76:
Other preference 
learning technique 
76

Page 77:
History
•
In 2023-early 2024, people thinks that RLHF seems to 
be the secret sauce to make ChatGPT work.
•
Many people proposes alternatives to PPO
•
DPO
•
ORPO
•
GRPO
77

Page 78:
Direct Preference Optimization 
(DPO)
•
Skips reward model training
•
Learns directly from the preference data
78
https://arxiv.org/pdf/2305.18290

Page 79:
Why does DPO work?
•
DPO is just a derived version of PPO assuming binary 
choice in the reward model.
79
A2C update
Assuming binary reward model

Page 80:
Odds Ratio Preference 
Optimization (ORPO)
•
DPO requires you to train SFT then do DPO
•
ORPO tries to do this in one step
80
https://arxiv.org/pdf/2403.07691
Odds(x) = k means x is k times
more likely than not 

Page 81:
ORPO
•
Note: although they claimed 
that ORPO doesn’t require 
separate SFT step, in 
practice people found that it 
is beneficial to do SFT than 
ORPO
81
Odds(x) = k means x is k times
more likely than not 
Oddsratio = how much more likely to generated a 
chosen than rejected

Page 82:
Problems with fixed preference 
data.
•
While ORPO and DPO works well on paper, they are 
optimized on the SFT or Preference data only. This 
might cause the model to overfit to the data.
•
RL is still required to generalize
•
Or the preference data needs to keep being updated over 
iterations just like in LLAMA 3.1 or Deepseek
82

Page 83:
Group Relative Policy Optimization 
(GRPO)
•
GRPO is a version of PPO which calculates the 
advantage function in a different manner
83
b(s) is typically the Value function V(s)
In this framework, the policy and value model is learned
Value model is cumbersome to learn because it’s the 
same architecture as the base model.
Also, V(s) might not be a good choice for b(s) because 
it might be hard to estimate in the LLM context.
https://arxiv.org/pdf/2402.03300

Page 84:
GRPO
84
A is averaged over the different outputs from the same input
(We need b(s))

Page 85:
When Policy Gradient is just 
supervised learning
85
The only difference is where the data comes from

Page 86:
Why RL works?
•
In DeepSeekMath, they found that models trained with 
RL perform better than SFT models in “alignment”
•
Top outputs have higher scores but capabilities still remain the 
same.
•
Bottlenecks
•
Rewards learning – how to get nice reward model for 
subjective tasks
•
Data - sampling of questions and answers for the model to 
learn
•
Algorithm – most RL algorithms are based on always true 
rewards
86
https://arxiv.org/pdf/2402.03300

Page 87:
Summary
•
Scaling laws
•
ChatGPT training
•
Pre-training
•
Post-training
•
SFT
•
Preference optimization
87

Page 88:
Midterm
•
In class. Room will be announced later.
•
1 page A4 front back
•
Calculator
Part A) Aj.Peerapon (25 points)
- Tokenization
- LM + subwords
- Attention + transformer
- Text gen + eval measure
Part B) Aj.Ekapol (25 points)
- Deep learning/Pytorch
- Word representation (sparse, dense)
- PoS (HMM)
- Sentence Embedding/Text classification
- Open question (design)
88

Page 89:
Takehome
89

Page 90:
Project
Group of 2-5 people
We expect more from 
more people
Anything text/NLP 
related
Must has some 
application component 
(cannot be purely basic 
NLP task)

Page 91:
HOW TO READ A 
SCIENTIFIC ARTICLE

Page 92:
2 Paper types
●Review article/tutorial
○Give insights about the field
○Useful for learning about a new field
○Read multiple to avoid the author’s bias
○Title usually has “review” or “tutorial”
●Primary research article
○More details on the experiments and results

Page 93:
Parts of an article
●Abstract
●Introduction
●Methods
●Results and discussion
●Conclusion
●Reference

Page 94:
Things to look for before reading an 
article
●Publication date
●Author names
○Previous and newer publications
●Keywords
●Acknowledgements and funding sources
https://arxiv.org/pdf/2403.07691

Page 95:
Getting the big picture
●Read the abstract
●Read the introduction
○What is the research question?
○What is the method?
○What had been done? How is it different from other work?
●Look at figures and results
Tip: keep track of terms you don’t understand

Page 96:
First reading
●Reread the introduction
●Skim methods
●Read results and discussion
○Does the figures make sense now?
●Write on the article!

Page 97:
Understanding the article
●Reread the article (until you get what you want)
●Check references for parts you don’t understand
●Reread the abstract
○Does your understanding match the abstract?
●Note down important points. This might come in handy 
when you write you paper/thesis!

Page 98:
Evaluating the article
●Does the method make sense?
○What are the limitations that the authors mention?
○Are there other limitations?
○Can it be used in other situations?
●Are the experiments legitimate?
○The sample size is big enough? How big?
○What kind of dataset is used? Hard enough to current standards? 
Baselines?
○The evaluation criterion is sound?
●Have these results been reproduced?
○Look for articles that cite this paper

Page 99:
ML paper checklist
●What is being done?
●How is it being done?
○How is it different from previous work
●What is the dataset?
○Nature of dataset. How many training/testing samples? How many 
classes/vocab size? Etc.
●Evaluation
○What are the baselines? Metrics is okay?
●Practicality/limitations
○Prone to parameter tuning?
○Computing resource / Runtime (training and testing)
○Other assumptions? Supervised, unsupervised, etc.

Page 100:
Useful tools
●https://scholar.google.com
○For finding other articles by the same authors or paper that cites 
the article
○Deep research of various kinds are quite useful for 
researching a topic. Be careful of hallucinations even if it 
has citation!
○Many famous papers has web presence
reddit, twitter, or openreview
video explaining the paper

Page 101:
Paper presentation
Pick from the list of papers we provide (right after midterm 
exam)
8 minute presentation + 2 minute progress update + 2 
minute QA
Should cover the ML paper checklist

