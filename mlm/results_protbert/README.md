---
base_model: Rostlab/prot_bert
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: results_protbert
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# results_protbert

This model is a fine-tuned version of [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: nan
- Accuracy: nan

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.31.0.dev0
- Pytorch 2.0.1+cu117
- Datasets 2.13.1
- Tokenizers 0.13.3
