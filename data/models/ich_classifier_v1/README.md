---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: "ucational/resources \nprogram, containing a workbook and online resources.\
    \ \n \nProcedures: Participants will rec"
- text: ' stimulations with the same cues as the observational phase. The comparison
    of the

    vasopressin group and the saline group will allow us to investigate how vasopressin
    influences behavioral effects of observational learning on pain perception as
    well as

    its effect on the neural '
- text: "hospitalization for worsening dyspnea and clinically \nsignificant laboratory\
    \ test abnormalities, determined per the Investigator’s judgment.  "
- text: 'brolizumab: a programed death receptor-1 (PD-1)- blocking an'
- text: "ls spent in target range (defined to be between 3.9 and 10.0 mmol/L). \n\
    The following comparisons will be done:  \ni) Fast-acting insulin-plus-pramlintide\
    \ closed-loop delivery vs. fast-acting insulin-alone closed-\nloop delivery; \
    \ \nii) Regular insulin-plus-pram"
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/all-MiniLM-L6-v2
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 6 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|:------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| B.3   | <ul><li>'Halozyme, Inc. \nProtocol HALO-110-101 Amendment 3 \nCLINICAL STUDY PROTOCOL \nTitle: \nA Phase 1b, Randomized, Open-Label Study of PEGylated \nRecombinant Human Hyaluronidase (PEGPH20) in \nCombination With Cisplatin Plus Gemcitabine and \nPEGPH20 in Combination With Atezolizumab and \nCisplatin Plus Gemcitabine Compared With Cisplatin Plus \nGemcitabine in Subjects with Previously Untreated, \nUnresectabl'</li><li>'-7888 Dosing \nEmulsion administered in combination with pembrolizumab in PROC. To determine ORR, DCR and \nPFS per immune RECIST (iRECIST) of DSP-7888 Dosing Emulsion administered in combination \nwith pembrolizumab in PROC and to evaluate the safety and tolerability of DSP-7888 Dosing \nEmulsion administered with pembrolizumab \nStudy Design:  \nThis is a Phase 1b/2, open-label, multicenter study of D'</li><li>'investigate the safety of ixekizumab in participants aged ≥18 years with \nmoderate-to-severe plaque psoriasis and/or active psoriatic arthritis in India \nRationale: \nThe rationale for this post-approval Phase 4 study is to evaluate the safety and tolerability when \nixekizumab is administered to participants in I'</li></ul>                                                                                             |
| B.16  | <ul><li>'tabolism.  \nTherefore, we conservatively have set the sample size at 10. \n \n\n14-476H, DeFronzo, PI Protocol, 07-10-15, AMD \n5 \n \nREFERENCES \n1.  \nPolonsky WH, Fisher L, Guzman S, Villa-Caballero L, Edelman SV: Psychological \ninsulin resistance in patients with type 2 diabetes: the scope of the problem. Diabetes \nCare 2005;28:2543-2545 \n2.  \nDeFronzo RA, Ferrannini E, Simonson DC: Fasting hyperglyc'</li><li>'dren with \nobsessive-compulsive disorder complicated by disruptive behavior: A multiple-baseline \nacross- responses design study. Journal of Anxiety Disorders, 27(3), 298–305. \nhttps://doi.org/10.1016/j.janxdis.2013.01.005 \nTate, R. L., Perdices, M., Rosenkoetter, U., Shadish, W., Vohra, S., Barlow, D. H., Horner, R., Kazdin, \nA., Kratochwill, T., McDonald, S., Sampson, M., Shamseer, L., Togher, L'</li><li>' NY-L, Song HJJMD, Tan BKJ, et al. Association of Olfactory Impairment \nWith All-Cause Mortality: A Systematic Review and Meta-analysis. JAMA \nOtolaryngology–Head & Neck Surgery. 2022;148(5):436-445. \ndoi:10.1001/jamaoto.2022.0263 \n14. \nAiyegbusi OL, Hughes SE, Turner G, et al. Symptoms, complications and \nmanagement of long COVID: a review. J R Soc Med. Sep 2021;114(9):428-442. \ndoi:10.1177/01410'</li></ul> |
| B.8   | <ul><li>'col # Pro00001307 \n \nPhase III Randomized Study of Autologous Stem Cell Transplantation with \nHigh-Dose Melphalan Versus High-Dose Melphalan and Bortezomib in \nPatients with '</li><li>'s-pramlintide closed-loop delivery vs. fast-acting insulin-alone closed-\nloop delivery;  \nii) Regular insulin-plus-pramlintide closed-loop delivery vs. fast-acting insulin-alone closed-loop \ndelivery.  \n \n \n7.1.2 Secondary endpoints and comparisons \n1. Percentage of time of glucose levels spent in target range, comparing fast-acting insulin-plus-\npramlintide closed-loop delivery vs. regular insuli'</li><li>'s will \ncount as one antihypertensive medication. \no Grade 3 proteinuria \no Any recurrent Grade 2 nonhematological toxicity requiring ≥2 interruptions \nand dose reductions \n• Any dose interruption or reduction due to toxicity which results in administration of \nless than 75% of the planned d'</li></ul>                                                                                                                                                                                                                                                                                                                                                      |
| B.2   | <ul><li>' \nImipenem/Cilastatin/Relebactam (MK-7655A) Versus Piperacillin/Tazobactam in Subjects \nwith Hospital-Acquired Bacterial Pneumonia or Ventilator-Associated Bacterial Pneumonia\nProtocol'</li><li>'nical Study Protocol \nHCRN LUN19-427 \n1.2\nStandard of Care for Small Cell Lung Cancer Brain Metastases \n'</li><li>'l non-follicular subtypes, including marginal zone lymphoma (MZL), \nand small lymphocytic lymphoma (SLL).  Early-stage indolent NHL '</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| B.4   | <ul><li>' excipient \n Investigational Product(s), Dose, and Mode of Administration: \n Durvalumab  1500mg  plus  tremelimumab  75mg  via  IV  infusion  Q4W,  starting  on  Week  0,  for  a  total  of \n 3  doses.  N.B  If  a  patient’s  weight  falls  to  30kg  or  below  the  patient  should  receive  weight-based \n dosing  equivalent  to  20  mg/kg  of  durvalumab  Q4W  and  1mg/kg  tremelimumab  Q4W  unti'</li><li>'y. Blood work will be checked weekly, based on the discretion of the \ntreating Medical Oncologist. \n \nIf radiotherapy has to be temporarily interrupted for technical or medical reasons, \nunrelated to the temozolomide administration, then treatment with temozolomide may \ncontinue. \n \n5.2  \nLow Dose Fractionated Radiation Therapy (LDFRT) \n \nAll patients will receive 0.5 Gy of radiation therapy twice'</li><li>'computer-generated \nrecommendation, while still maintaining the 6 µg/unit ratio. The computer-generated \nrecommendations are based on a dosing algorithm [5]. The carbohydrate content for meals'</li></ul>                                                                                                                                                                                                                          |
| B.5   | <ul><li>'aboratory abnormalities: \n• \nANC < 1,000/µL (hematopoietic growth factors will not be permitted \nduring screening in the Phase 1 or Phase 2 segments of the study or \nin Cycle 1 of the Phase 1b segment of the study) in either segment of \nthe study. \n• \nPlatelet count < 75,000/µL for patients in whom < 50% of bone \nmarrow nucleated cells are plasma cells, and < 50,000/µL for \npatients in whom ≥ 50% '</li><li>'efractory to or ineligible for ESAs is defined as RBC-Transfusion \nDependence despite ESA treatment of ≥40,000 units/week recombinant \nhuman erythropoietin for 8 weeks or an equivalent dose of darbepoetin \n(150 \uf06dg/week) or serum EPO level >500 mU/mL in patients not \npreviously treated with ESAs. \n3. Patients m'</li><li>' diuretics (amiloride) or MRAs e.g., spironolactone or eplerenone \nwhich cannot be discontinued 4 weeks prior to screening visit. The patient’s primary physician, who \nis not involved in this study, will determine if discontinuation is possible. \n- \nAtrial fibrillation/flutter \n- \nCongestive heart failure (NYHA class 3-4)  \n- \nHistory of cardiac arrhythmia  \n- \nSevere forms of respiratory disease '</li></ul>                                                                                      |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("brolizumab: a programed death receptor-1 (PD-1)- blocking an")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 2   | 47.4467 | 130 |

| Label | Training Sample Count |
|:------|:----------------------|
| B.16  | 38                    |
| B.2   | 7                     |
| B.3   | 84                    |
| B.4   | 29                    |
| B.5   | 79                    |
| B.8   | 63                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 2e-05
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0013 | 1    | 0.4247        | -               |
| 0.0667 | 50   | 0.2653        | -               |
| 0.1333 | 100  | 0.2099        | -               |
| 0.2    | 150  | 0.1578        | -               |
| 0.2667 | 200  | 0.1038        | -               |
| 0.3333 | 250  | 0.0705        | -               |
| 0.4    | 300  | 0.0521        | -               |
| 0.4667 | 350  | 0.0365        | -               |
| 0.5333 | 400  | 0.0368        | -               |
| 0.6    | 450  | 0.029         | -               |
| 0.6667 | 500  | 0.029         | -               |
| 0.7333 | 550  | 0.0259        | -               |
| 0.8    | 600  | 0.0202        | -               |
| 0.8667 | 650  | 0.0154        | -               |
| 0.9333 | 700  | 0.0213        | -               |
| 1.0    | 750  | 0.0152        | -               |

### Framework Versions
- Python: 3.13.12
- SetFit: 1.1.3
- Sentence Transformers: 5.3.0
- Transformers: 4.45.2
- PyTorch: 2.10.0+cpu
- Datasets: 4.8.3
- Tokenizers: 0.20.3

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->