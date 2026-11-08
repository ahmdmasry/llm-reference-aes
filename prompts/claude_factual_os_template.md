# Claude prompt example: factual OS question

You are a knowledgeable computer-science lecturer creating high-quality model
reference answers for college freshmen.

Question: [QUESTION]

Produce three sets of answers:

- Short: [SHORT RANGE] words
- Medium: [MEDIUM RANGE] words
- Long: [LONG RANGE] words

Each set must contain five answers, using each of these styles once:

1. objective and factual;
2. concise and technical;
3. expository and clear;
4. analytical and structured; and
5. instructional and explanatory.

Use correct terminology, answer the question directly, and ensure the
explanation is coherent and appropriate for a college freshman. Do not include
word counts, citations, or meta-commentary.

Return valid CSV text with these columns: `target_length`, `tone`, and `essay`.
Surround all fields with double quotation marks and escape internal quotation
marks.
