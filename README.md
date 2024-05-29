# Laboratory-Scale AI Repository
This is the official code repository for the ACM FAccT'24 paper ***Laboratory-Scale AI: Open-Weight Models are Competitive with ChatGPT Even in Low-Resource Settings***, available at [https://arxiv.org/pdf/2405.16820](https://arxiv.org/pdf/2405.16820).

### 1. Structure

The project is built primarily on the HuggingFace transformers, trl, datasets, and evaluate libraries. We used bash files to run analyses on a GCP cloud instance. Some contributors containerized the repo using docker; see the Dockerfile in this directory for an example of how this could look. Intermediate results were logged to the wandb accounts of individual contributors.

Each of the contributors cloned a main branch and customized it to some degree for their specific analysis. To make the resulting analyses easier to parse, we've organized the code for each analysis into a subdirectory for this release, with two primary directories for  performance results (entity resolution, clinical dialogue summarization, fact-checking) and values results (bias, privacy, abstention).

### 2. Requirements

The requirements file includes the libraries needed to run the analyses in each of the subdirectories; using the dp-transformers repo may necessitate installing its dependencies as well. For the privacy task, you'll need to copy the dp-transformers repository from [https://github.com/microsoft/dp-transformers](https://github.com/microsoft/dp-transformers). We've left a directory for this to be added. We recommend creating a unique environment for running the project (for example, with conda, `conda create -n "labscale" python=3.11`) and then installing the requirements (`pip install -r requirements.txt`). 

### 3. Token Access

A HuggingFace account is needed to use some of the HuggingFace Hub functionality, which includes verifying access to models like LLaMA-2. You can get a token from the account here: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

To log results during fine-tuning and evaluation, you'll also want an account with Weights and Biases; see [https://wandb.ai](https://wandb.ai). Then, make a new project, and it will provide you with a token.

### 4. Paper & Citation

Please cite the following version of the Lab-Scale AI paper, from the ACM FAccT proceedings:

> @article{wolfe2024lab-scale,\
>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={Laboratory-Scale AI: Open-Weight Models are Competitive with ChatGPT Even in Low-Resource Settings},\
>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={Wolfe, Robert and Slaughter, Isaac and Han, Bin and Wen, Bingbing and Yang, Yiwei and Rosenblatt, Lucas and Herman, Bernease and Brown, Eva and Qu, Zening and Weber, Nic and Howe, Bill},\
>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;journal={arXiv preprint arXiv:2405.16820},\
>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2024}\
> }

### 5. Other Resources

This repository is primarily a reference implementation intended for reproducibility, insofar as that's possible given the stochasticity of the models and the black box character of evaluating closed models. However, this also means that our repo might not be the best starting place for everyone who wants to customize their own open models. There are many great resources for using the technologies employed in the paper, some of which are more geared toward newer users. Some resources we found helpful include:

- [https://huggingface.co/docs/trl/en/sft_trainer](**TRL-Library-SFT-Trainer**): The TRL library's SFT trainer, one of the most straightforward ways to fine-tune an open chatbot, with clear explanations and code examples.
- [https://github.com/artidoro/qlora](**qLoRA-Library**): The official repository for the qLoRA technology that enables quantized language models to be fine-tuned with insertable weight matrices. Includes helpful colab demos.
- [https://wandb.ai/sauravmaheshkar/QLoRA/reports/What-is-QLoRA---Vmlldzo2MTI2OTc5](**W&B-qLoRA-Tutorial**): A Weights and Biases qLoRA tutorial; includes a colab notebook and detailed walkthrough of technical characteristics with W&B integration.