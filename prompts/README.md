# Representative Claude prompts

The files in this directory document the three prompt patterns used with
Claude: open-ended opinion/story tasks, source-based tasks, and factual OS
questions. Bracketed fields were replaced with the applicable task information
and target ranges. The same procedure was followed for GPT and Gemini.

The templates preserve the substantive generation instructions while omitting
dataset passages and the generated answers. When a generated answer fell
outside the intended range, a follow-up request asked the model to regenerate
it within the target range. These attempts were not systematically counted.
