# Ineffective-Thinking

## Project Introduction

This project is a system for training and evaluating language models, supporting multi-step reasoning and judgment tasks.

## Usage Instructions

### Training

```bash
bash main.sh
```

### Inference

Standard inference:
```bash
bash inference/inference.sh
```

OOD (Out-of-Distribution) inference:
```bash
bash inference/inference_ood.sh
```

### Data Processing

```bash
bash data_preprocess/hotpot.sh
```

### Start Server

```bash
bash scripts/LLM_as_a_Judge_server_70B.sh
```
