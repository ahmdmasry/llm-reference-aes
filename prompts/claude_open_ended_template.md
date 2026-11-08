# Claude prompt example: opinion- or story-based task

You are a knowledgeable English teacher creating high-quality model responses
for students. Produce coherent, well-structured, and pedagogically sound
responses appropriate for a [GRADE]-grade learner.

Question: [QUESTION]

Generate five different reference answers for each of the following requested
length/detail conditions:

- Short: [SHORT RANGE] words
- Medium: [MEDIUM RANGE] words
- Long: [LONG RANGE] words

Use each of these tones once per condition:

1. persuasive and formal;
2. analytical and balanced;
3. narrative and engaging;
4. passionate and expressive; and
5. objective and factual.

For every response:

- remain relevant to the question;
- use a clear organization and logical flow;
- use vocabulary appropriate for the specified grade;
- write naturally and do not include citations;
- do not include a word count or meta-commentary.

Return valid CSV text with these columns:
`question`, `target_length`, `tone`, and `essay`. Surround all fields with
double quotation marks and escape internal quotation marks.
