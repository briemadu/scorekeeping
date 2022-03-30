# README 

This is the accompanying repository of the following publication:

Madureira & Schlangen (2022). Can Visual Dialogue Models do Scorekeeping? 
Exploring how Dialogue Representations Incrementally Encode Shared Knowledge. 
Short paper presented at ACL 2022 in Dublin, Ireland.

## What is this repository for? 

This repository contains the code to run the experiments with the probing 
classifier and the evaluation scripts to get the results reported in the paper. 

## How do I get set up?

Build the conda environemnt using the ```main_task_environment.yml``` 
file and activate the environment. You will need:

- [Comet](https://www.comet.ml/) (optional)
- [h5py](https://www.h5py.org/) 2.10.0
- [Pytorch](https://pytorch.org/) 1.7.1
- [NumPy](https://numpy.org/) 1.19.2
- [pandas](https://pandas.pydata.org/) 1.1.5
- [scikit-learn](https://scikit-learn.org/stable/index.html) 0.23.2
- [sentence-transformers](https://www.sbert.net/) 2.1.0
- [tqdm](https://pypi.org/project/tqdm/)

and other standard libraries.

Move the propositions, embeddings and dialogue representations to the ```data/``` 
folder on this directory. 

You can use 
```
sh copy_data.sh
```

## How do I replicate the results?

All experiments discussed in the paper can be replicated with the following 
steps:

1. Hyperparameter search was carried out with ```optim.py``` and with another 
version of this file with a more constricted list of hypeparameters.
2. Keep the configurations as in  ```config.py```.
3. Run experiments using ```run_experiments.sh```. If you do not want to
 use  ```comet.ml```, add the argument ```--ignore_comet``` to the commands.
4. Run ```evaluation.ipynb```, ```hypothesis-testing.ipynb``` 
and ```human-results-eval.ipynb``` to get the results of the paper.
5. Unit tests are at ```tests.py``` and ```tests_eval.py```. 

## Details of each argument

Paths

- ```-path_to_probes```: directory where the propositions datasets in JSON 
format are; they must be named ```propositions_{train, val, test}.json```.
- ```-path_to_representations```: directory where the dialogue representations 
in h5py format are; they must be named ```{a, q}Bot_{RL_DIV, SL, ICCV_RL}_representations_{train, val, test}.h5```.
- ```-path_to_embeddings```: directory where the proposition embedding 
files (output of ) are; they must be named ```embeddings_{nli-roberta-base-v2, other encoders}.p```.

Comet

We used [comet.ml](comet.ml) to organize experiments. The following arguments
- ```-ignore_comet```: use if you do not want to log experiment data on Comet.
- ```-comet_key```: personal Comet key.
- ```-comet_project```: name of Comet project.
- ```-comet_workspace```: name of Comet workspace.

Setting
- ```-random-seed```: an integer to be used as seed for modules that require 
random numbers.
- ```-device```: CPU or CUDA device.
- ```-sent_encoder```: Name of the sentence encoder.
- ```-bot```: Answerer or Questioner bot.
- ```-bot_version```: Name of the bot version.
- ```-task```: Which of the main tasks.
- ```-control_task```: Whether to run the main experiment or one of the two 
control tasks that replace the dialogue state representations in training by 
random numbers or by zeros.

Training hyperparameters
- ```-classifier```: Which of the three implemented classifiers to use 
(see ```models.py```).
- ```-batch_size```: Size of the batch.
- ```-hidden_dim```: Size of the hidden layer.
- ```-probe_dim```: Dimension of the sentence embeddings.
- ```-dialogue_rep_dim```: Dimension of the dialogue representations.
- ```-hidden_dim_2```: Dimension of the sencond hidden layer, only used if the 
classifier is Deepest.
- ```-n_epochs```: Maximum number of training epochs.
- ```-early_stopping```: Number of iterations after which to to stop training
if no improvement on validation accuracy was observed.
- ```-lr```: Learning rate.
- ```-dropout```: Probability used for dropout.
- ```-clip```: Clipping size, 0 if no clipping.

## Citation

If you use this work, please cite: (tbd)

## License

Source code licensed under MIT.

