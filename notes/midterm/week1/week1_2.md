Page 1:
Dictionary-based Tokenization
2110572: NLP SYS
Assoc. Prof. Peerapon Vateekul, Ph.D.
Department of Computer Engineering, Faculty of Engineering, Chulalongkorn University
peerapon.v@chula.ac.th
Credits to: TA.Pluem, TA.Knight, and all TA alumni 
Based on Aj.Ekapol’s slide in 2017

Page 2:
Outlines
The need for segmentation
Thai Tokenization
1) Longest Matching
2) Maximal Matching
2

Page 3:
The need for segmentation
●Text as a stream of characters
●We need a way to understand the meaning of text
○Break into words (assign meaning to word)
○Break into sentences (put word meanings back to sentence meaning)
3
Word Tokenization

Page 4:
Tokenization - Thai
●Thai has no space between words
●Thai has no clear sentence boundaries
4
http://www.starwars.siligon.com/swcong.html 

Page 5:
Tokenization - Thai
Social media text
#สตอรี่ของโม 😊🐙 #Days23ofMobile 🌈 ...23 วันแลวนะเจาโม พี่กําลังคิดถึงหนูอยูเลย แหนะ เมื่อวานหายเลย
นะ 🙈 พี่ก็วาหายไปไหนแอบหนีไปเลนลองบอรดนี่เอง เลนระวังๆนะพี่เปนหวง เดวลมแถมไมมีตะหลามคอยเลน
เปนเพื่อนอีก 😆 พี่จะบอกวา "หนูอยาลืมลงชื่อเลือกตั้งนะ" เดี๋ยวพวกพี่ขํากันไมออกนะ 5555 ฝากไวกันลืม ยิ่งเดอ
ๆอยู อีกอยางๆหนูตองกลับมานะพวกเรารอหนูอยู 😊 พี่นี่เตรียมรอโหวตใหหนูเต็มที่เลย 😋 ไปเรียนเปนไงบาง 
ยังเหงาอยูมั้ย แตพี่วาคงไมแลวหละมั้ง วันนี้ขึ้นสเตจอีกแลวสินะ เหนื่อยมั้ยคะ แตคงสนุกมากกวาอยูแลวเนอะ 😁 
แคไดเห็นรอยยิ้มของหนูพี่ก็สบายใจและ แตเอะ เมื่อวานใครบนอยากกลับบานอีกแลวนะ อดดู the toy เลยอะดิ้ 
😆 หวายๆๆๆ นาสงสาร 555 โอกาสหนายังมีนะ ตั้งใจทํางานนะคะ เจาหลามแฟนหนูก็อด ไมไดอดคนเดียว
ซะหนอยนะ 555 🙊 ดูมีความสุขจังนาเจาตัวเล็ก 😊💗 คิดถึงนะ เดี๋ยวก็ได 2shot กันแลว พี่โคตรตื่นเตนเลย 
ตื่นเตนกวาไปจับมือเยอะ 🙈 คิดทาไมออกเลย มีทาไหนแนะนํามั้ย ชวยพี่หนอยยยยย 😅... #Mobilebnk48 #ตู
เพลงโมบิล #ชาวเหรียญหยอดตู #MOTA09
5

Page 6:
Tokenization - Thai
●Many word boundaries depends on the context (meaning)
●Even amongst Thais the definition of word boundary is unclear
○Needs a consensus when designing a corpus
○Sometimes depends on the application
■Linguist vs machine learning concerns
6
ตา กลม vs ตาก ลม
คณะกรรมการตรวจสอบถนน

Page 7:
Dictionary-based vs Machine-learning-based
●1) Dictionary-based
○Longest matching
○Maximal matching
●2) Machine-learning-based
7

Page 8:
Dictionary-based word segmentation
●
Perform by scanning a string and matching each substring against words from a dictionary.  
(No dataset needed, just prepare a dictionary!)
●
However, there is ambiguity in matching. (There are many many ways to match.)
●
So, matching methods are developed: 
○
1. Longest matching
○
2. Maximal matching
8
ปายกลับรถ
Meknavin, Surapant, Paisarn Charoenpornsawat, and Boonserm Kijsirikul. "Feature-based Thai word segmentation." Proceedings of Natural Language Processing Pacific Rim Symposium. Vol. 97. 1997.

Page 9:
1) Longest Matching
• Scan a sentence from left to right
• Keep finding a word from the starting point until no word matches (longest), then move to the 
next point
• Backtrack if current segmentation leads to an un-segmentable chunk
• ปายกลับรถ
Start scanning with “ป” as the starting point
• ปายกลับรถ
Keep scanning …
• ปาย/กลับรถ
No more words start with “ปาย”, move to the next point
• …
• ปาย/กลับ/รถ
9

Page 10:
2) Maximal Matching
• Generate all possible segmentations
• Select the segmentations with the fewest words
10
Haruechaiyasak, Choochart, Sarawoot Kongyoung, and Matthew Dailey. "A comparative study on thai word segmentation approaches." Electrical Engineering/Electronics, Computer, 
Telecommunications and Information Technology, 2008. ECTI-CON 2008. 5th International Conference on. Vol. 1. IEEE, 2008.

Page 11:
What if?
• What if there are more than one segmentation with the fewest words?
• Other heuristics are applied, for example
• Language model score 
11

Page 12:
Maximal matching
• Maximal matching can be done using dynamic programming.
• Let  d(i,j)  be the function that returns the number of the fewest words possible, with the last word 
starting with ith character (row) and ending with jth character (column). It can be defined as:
when c[i..j] is a string of words in the sentence (assume it is started at index 1) and the base case is 
d(1,1) = 1.
12

Page 13:
Maximal matching: Initialization (1st row)
• Maximal matching can be done using dynamic programming.
• Let  d(i,j)  be the function that returns the number of the fewest words possible, with the last word 
starting with ith character (row) and ending with jth character (column). It can be defined as:
when c[i..j] is a string of words in the sentence (assume it is started at index 1) and the base case is 
d(1,1) = 1.
13
1st row and it is a word

Page 14:
Maximal matching: Find a word in dictionary
• Maximal matching can be done using dynamic programming.
• Let  d(i,j)  be the function that returns the number of the fewest words possible, with the last word 
starting with ith character (row) and ending with jth character (column). It can be defined as:
when c[i..j] is a string of words in the sentence (assume it is started at index 1) and the base case is 
d(1,1) = 1.
14
If last word is a word

Page 15:
Maximal matching: Check all possible 
segmentations before the ﬁnal word
• Maximal matching can be done using dynamic programming.
• Let  d(i,j)  be the function that returns the number of the fewest words possible, with the last word 
starting with ith character (row) and ending with jth character (column). It can be defined as:
when c[i..j] is a string of words in the sentence (assume it is started at index 1) and the base case is 
d(1,1) = 1.
15
Check all possible segmentations before the final word
Check the whole column in the previous row

Page 16:
Maximal matching: The ﬁnal word
• Maximal matching can be done using dynamic programming.
• Let  d(i,j)  be the function that returns the number of the fewest words possible, with the last word 
starting with ith character (row) and ending with jth character (column). It can be defined as:
when c[i..j] is a string of words in the sentence (assume it is started at index 1) and the base case is 
d(1,1) = 1.
16
The final word

Page 17:
Example (1)
17
5
4
∞

Page 18:
Example (2) 
18
5
4
∞

Page 19:
Example (3)
19
5
4
∞

Page 20:
Example (4)
20
5
4
∞

Page 21:
Example (5):
Backtracking
21
5
4
∞

Page 22:
22
https://pythainlp.org/docs/5.0/api/tokenize.html 

Page 23:
How to improve the tokenizer? NN
 Predict 1/0 for each character
23
Character Tokens
String-to-Integer (stoi) + Pad
Predict Beginning of Word
Complicated Function (NN)
Use neural nets to tokenize?
Reference: http://web.archive.org/web/20181005101442/https://sertiscorp.com/thai-word-segmentation-with-bi-directional_rnn/ 

Page 24:
NN-based Tokenizers Performance
24
https://github.com/PyThaiNLP/attacut 

