\# EdgeShield AI — Dataset Documentation



\## Datasets Used



EdgeShield AI uses two publicly available SMS datasets for training and evaluating its scam detection model.



\### Dataset 1: ScamShield



\- \*\*Source:\*\* Hugging Face

\- \*\*Dataset:\*\* `sidzzz07/scamshield-dataset`

\- \*\*Original samples:\*\* 85,680

\- \*\*Purpose:\*\* Multilingual scam and legitimate SMS classification



\### Dataset 2: SMS Phishing Dataset



\- \*\*Source:\*\* Mendeley Data / NLM Dataset Catalog

\- \*\*Catalog ID:\*\* `0001449541`

\- \*\*Original samples:\*\* 5,971

\- \*\*Purpose:\*\* Additional SMS phishing examples



\## Data Preparation



Both datasets were cleaned and standardized using `ml/clean\_datasets.py`.



The processed dataset contained 85,811 SMS messages:



\- Legitimate: 54,710

\- Scam: 31,101



The dataset was divided into training and testing sets using an 80:20 group-aware split.



\- Training: 68,652 messages

\- Testing: 17,159 messages



\## Languages



The cleaned dataset contains English, Hindi, Hinglish, and a small number of messages with unknown language labels.



\## Reproducibility



The dataset sources and preprocessing code are provided so the training pipeline can be reproduced.



Raw datasets are not redistributed in this repository. Users should download them from their original sources, subject to the applicable licenses.



\## Evaluation Note



A near-duplicate audit identified 209 potentially overlapping test examples across the train/test split. Sensitivity analysis was performed after excluding these flagged test examples. Results should therefore not be described as completely leakage-free.

