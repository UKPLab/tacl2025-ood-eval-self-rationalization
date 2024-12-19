# [Self-Rationalization in the Wild: A Large Scale Out-of-Distribution Evaluation on NLI-related tasks](https://openreview.net/pdf?id=KYEdQdGvAR)
This repository provides the code for reproducing the results from the TACL paper "Self-Rationalization in the Wild: A Large Scale Out-of-Distribution Evaluation on NLI-related tasks". In short, it contains the code for fine-tuning the two language models T5-large and OMLo (LoRA) for self-rationalization, and performing inference on 19 NLI-related Out-of-Distribution datasets.

## Abstract
> Free-text explanations are expressive and easy to understand, but many datasets lack annotated explanation data, making it challenging to train models for explainable predictions. To address this, we investigate how to use existing explanation datasets for self-rationalization and evaluate models' out-of-distribution (OOD) performance. We fine-tune T5-Large and OLMo-7B models and assess the impact of fine-tuning data quality, the number of fine-tuning samples, and few-shot selection methods. The models are evaluated on 19 diverse OOD datasets across three tasks: natural language inference (NLI), fact-checking, and hallucination detection in abstractive summarization. For the generated explanation evaluation, we conduct a human study on 13 selected models and study its correlation with the Acceptability score (T5-11B) and three other LLM-based reference-free metrics. Human evaluation shows that the Acceptability score correlates most strongly with human judgments, demonstrating its effectiveness in evaluating free-text explanations. Our findings reveal: 1) few annotated examples effectively adapt models for OOD explanation generation; 2) compared to sample selection strategies, fine-tuning data source has a larger impact on OOD performance; and 3) models with higher label prediction accuracy tend to produce better explanations, as reflected by higher Acceptability scores.

Contact person: [Jing Yang](mailto:jing.yang@tu-berlin.de)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Getting started

### Environment
Follow the instructions below to recreate the environment used for all our experiments
1. Create and activate a conda environment
```
conda create -n oodeval python=3.10
conda activate oodeval
```
2. Install the required packages
```

```
### Few-shot sample selection

### Fine-tuning T5-Large

### Fine-tuning OLMo-7B

### Inference with T5-Large

### Inference with OLMo-7B

Fine-tuned models and OOD generations can be found in this [drive folder](https://drive.google.com/drive/folders/0B073WIPY0sxofjhMV0E4bjdaai03ZXRYTERYQ1BTXzdnT051TkJjcEx1clBmV2xOMXRnWnM?resourcekey=0-Kx9uJNjUKuqibtO93Q0hzw&usp=drive_link).
