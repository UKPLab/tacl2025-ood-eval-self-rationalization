# Self-Rationalization in the Wild: A Large Scale Out-of-Distribution Evaluation on NLI-related tasks
This repository provides the code for reproducing the results generated from the TACL paper "Self-Rationalization in the Wild: A Large Scale Out-of-Distribution Evaluation on NLI-related tasks".

## Abstract:
Free-text explanations are expressive and easy to understand, but many datasets lack annotated explanation data, making it challenging to train models for explainable predictions. To address this, we investigate how to use existing explanation datasets for self-rationalization and evaluate models' out-of-distribution (OOD) performance. We fine-tune T5-Large and OLMo-7B models and assess the impact of fine-tuning data quality, the number of fine-tuning samples, and few-shot selection methods. The models are evaluated on 19 diverse OOD datasets across three tasks: natural language inference (NLI), fact-checking, and hallucination detection in abstractive summarization. For the generated explanation evaluation, we conduct a human study on 13 selected models and study its correlation with the Acceptability score (T5-11B) and three other LLM-based reference-free metrics. Human evaluation shows that the Acceptability score correlates most strongly with human judgments, demonstrating its effectiveness in evaluating free-text explanations. Our findings reveal: 1) few annotated examples effectively adapt models for OOD explanation generation; 2) compared to sample selection strategies, fine-tuning data source has a larger impact on OOD performance; and 3) models with higher label prediction accuracy tend to produce better explanations, as reflected by higher Acceptability scores.

Contact person: [Jing Yang](jing.yang@tu-berlin.de)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Getting started

### Few-shot sample selection

### Fine-tuning T5-Large

### Fine-tuning OLMo-7B

### Inference with T5-Large

### Inference with OLMo-7B

