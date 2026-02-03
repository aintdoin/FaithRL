# Learning to Reason Faithfully through Step-Level Faithfulness Maximization

This repository contains the official implementation for the paper **"Learning to Reason Faithfully through Step-Level Faithfulness Maximization"**.

**Authors:**
Runquan Gui$^{2,1}$, Yafu Li$^{1}$, Xiaoye Qu$^{1}$, Ziyan Liu$^{2}$, Yeqiu Cheng$^{2}$, Yu Cheng$^{3}$

$^1$ Shanghai AI Laboratory
$^2$ University of Science and Technology of China
$^3$ The Chinese University of Hong Kong

---

## Preliminaries

### 1. Environment Setup
Please ensure you have Conda installed. You can set up the environment with the following commands:

```bash
# Create a new conda environment
conda create -n faithrl python=3.10
conda activate faithrl

# Install dependencies
pip install -r requirements.txt
```
### 2. Start LLM-as-a-Judge Server
Before training or evaluation, you need to start the judge server (used for faithfulness evaluation or reward modeling).

```bash 
bash scripts/LLM_as_a_Judge_server_70B.sh
```

## Data Processing
Please download the original datasets from their respective official repositories first. Place them in the data/ directory (or modify the scripts to point to your data path).

### Standard Datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue)
Run the corresponding shell scripts to preprocess the data:

```bash
# For HotpotQA
bash data_preprocess/hotpot.sh

# For 2WikiMultiHopQA
bash data_preprocess/2wikimultihop.sh

# For MuSiQue
bash data_preprocess/musique.sh
```
### Out-of-Distribution (OOD) Datasets
For mathematical reasoning datasets used in OOD evaluation:
```bash
# For MATH500
python data_preprocess/MATH500.py

# For GSM8k
python data_preprocess/GSM8k.py
```

## Training
To train the model using our proposed method:
```bash
bash main.sh
```

## Inference
### Standard Inference
Evaluate the model on standard benchmarks:
```bash
bash inference/inference.sh
```

### Out-of-Distribution (OOD) Inference
Evaluate the model's generalization capabilities on OOD datasets:
```bash
bash inference/inference_ood.sh
```