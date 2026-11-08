# Claude prompt example: source-based task

You are a knowledgeable English teacher creating high-quality, source-based
model responses for a [GRADE]-grade learner.

Question: [QUESTION]

Source text: [INSERT THE SOURCE TEXT OBTAINED FROM THE DATASET]

Rely only on the source text. Do not introduce information that cannot be
explicitly found or reasonably inferred from it. Include relevant examples
from the source and briefly explain how they support the response.

Generate five different reference answers for each condition:

- Short: [SHORT RANGE] words
- Medium: [MEDIUM RANGE] words
- Long: [LONG RANGE] words

Use each of these tones once per condition: persuasive and formal; analytical
and balanced; narrative and engaging; passionate and expressive; objective and
factual.

Ensure that each response is coherent, well organized, relevant, and
appropriate for the specified grade. Do not include word counts,
meta-commentary, outside facts, or citations.

Return valid CSV text with these columns:
`question`, `target_length`, `tone`, and `essay`. Surround all fields with
double quotation marks and escape internal quotation marks.
