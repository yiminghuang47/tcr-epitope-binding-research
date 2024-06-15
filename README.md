# TCR-Epitope Binding Research

*Author: Yiming Huang*

This is the repository for the NJIT Research Internship with Dr. Zhi Wei on using Protein Large Language Models to enhance TCR-epitope binding prediction. In contains experiments using the TChard dataset from [this paper](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2022.1014256/full). I first tested existing models like [NetTCR-2.0](https://github.com/mnielLab/NetTCR-2.0), [VIBTCR](https://github.com/nec-research/vibtcr). Then I used a different strategy where I first extract the embeddings of the protein sequences using Protein Large Language Models such as [prot-bert](https://huggingface.co/Rostlab/prot_bert) and [esm-2](https://huggingface.co/docs/transformers/en/model_doc/esm), then used neural networks to analyze those embeddings to get results. The detailed procedure can be found in the report below.

Report: https://docs.google.com/document/d/1O9T-Aj8ckHIk705gDd8mrLt8AZ5jCWMHUc-v5c3r-uI/edit?usp=drive_link

TChard Test Data: https://docs.google.com/spreadsheets/d/1tg9p-UdSuznDDzeWsvSEPg5CEC90PfDuUZU39SGXKDE/edit?usp=drive_link

MHC Test Data: https://docs.google.com/spreadsheets/d/1STeIJ1Yc9obHKg-WkQEIwt4mSOUlau1x3l3T69-nYdo/edit?usp=drive_link

The notebooks used can be found in this Google Drive folder: https://drive.google.com/drive/folders/1egXXsm6AYfB0hBuZRLvwPpN0BRu6-4Cm?usp=sharing.

The processed datasets can be found in this repository: https://github.com/yiminghuang47/tcr-data
