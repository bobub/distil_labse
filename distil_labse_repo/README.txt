This project report can be used to create a task-agnostic distillation of the multilingual transformer language model, LaBSE (Language Agnostic Sentence Embeddings). See https://arxiv.org/abs/2007.01852 for more about the teacher model.

The project was carried out as an industrial research project for Sinch AB and was also a masters thesis project for KTH Royal Institute of Technology. A link to the full thesis dissertation will be added. 

Presentation slides found here:
https://docs.google.com/presentation/d/12Fhqsvh7PK-G0jvQU8FZ7vDQ5xA_jgBxKI301dnq618/edit?usp=sharing

Executables:
distillation.ipynb : Used to create a task-agnostic distillation of LaBSE from the OpenSubtitles2018 dataset.
eval.ipynb : Evaluates the distilled model on speed, compression and cosine similarity performance.
hyperparam_search.ipynb : Used to perform hyperparameter tuning for distillation.
preprocess_opensubtitles.ipynb: Preprocesses the training dataset.
preprocess_tweets.ipynb: Preprocesses the downstream task dataset.

Important Libraries:
TextBrewer - used for distillation. See https://textbrewer.readthedocs.io/en/latest/ for usage.
LabML - used for the Switch Transformer architecture. See https://nn.labml.ai/transformers/switch/index.html
Papermill - used for hyperparameter tuning. See https://papermill.readthedocs.io/en/latest/
HuggingFace - general usage. See https://huggingface.co/

Environment:
Custom, extended versions of TextBrewer and LabML were used. Please use the extended versions on github/bobub for reproducing the functionality.

NOTE:
This repo is designed to be used in an AWS SageMaker environment. Therefore it may have compatibility issues elsewhere.
    
