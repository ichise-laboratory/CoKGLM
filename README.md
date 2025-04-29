# KG Retrieval

## Requirements
- Python Version: 3.8.19
- Dependencies: requirements.txt

## Inplementation

Use the following script to retrieve KG triples:
```bash
CUDA_VISIBLE_DEVICES=0 python build_dataset.py \
--input_file qa2hall_data.csv \
--out_file ../QA2HALL/retrieved_data/extract5_per_entity_new.csv \
--file_entity_id ../kg/entity2id.txt \
--file_relation_id ../kg/relation2id.txt
```

Use the following script to process data for LM only detection:
```bash
CUDA_VISIBLE_DEVICES=0 python build_dataset_LLMonly.py \
--input_file qa2hall_data.csv \
--out_file ../QA2HALL/retrieved_data/LLMonly_per_entity_new.csv \
--file_entity_id ../kg/entity2id.txt \
--file_relation_id ../kg/relation2id.txt
```

## Pre-Prepared Datasets
Pre-retrieved datasets have already been prepared for both FADE-based and QA2HALL-based data. These datasets include:
- Retrieved 5, 10, and 20 triples for each entity
- Processed data for LM-only detection

The datasets are stored in the following directories:
- ../FADE/retrieved_data/
- ../QA2HALL/retrieved_data/


# Knowledge Verification

## Requirements
- Python Version: 3.7.16
- Dependencies: requirements.txt

## Inplementation
The model supports two types of hallucination detection settings:

### KG-enhanced detection
Retrieves k triples from the KG to enhance detection.

Use the following script to perform cross-validation.

- For QA2HALL-based data:
```bash
CUDA_VISIBLE_DEVICES=0 python run_baseline_croval.py config_kg_qa2hall.json \
> extract_5_per_entity_nestcroval_output.txt 2>&1
```

- For FADE-based data:
```bash
CUDA_VISIBLE_DEVICES=0 python run_baseline_croval.py config_kg_fade.json \
> fade_extract_5_per_entity_nestcroval_output.txt 2>&1
```

To perform experiments using 10 and 20 triples, the following arguments have to be adjusted:

max_memory_bank: the number of triples to use
train_file, output_dir, logging_dir

### LM-only detection
Operate without using the KG.

Use the following script to perform cross-validation.

- For QA2HALL-based data:
```bash
CUDA_VISIBLE_DEVICES=0 python run_baseline_croval.py config_llmonly_qa2hall.json \
> LLMonly_per_entity_nestcroval_output.txt 2>&1
```

- For FADE-based data:
```bash
CUDA_VISIBLE_DEVICES=0 python run_baseline_croval.py config_llmonly_fade.json \
> fade_LLMonly_per_entity_nestcroval_output.txt 2>&1
```




