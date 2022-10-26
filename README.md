# README

This is the accompanying repository of the following publication:

Madureira & Schlangen (2022). Can Visual Dialogue Models do Scorekeeping? 
Exploring how Dialogue Representations Incrementally Encode Shared Knowledge. 
Short paper presented at ACL 2022 in Dublin, Ireland.

## What is this repository for?

The paper proposes an evaluation method to assess how visual dialogue models 
keep track of information about an image that is private/shared at a given 
turn. It implements a probing classifier based on a neural network whose 
input is a proposition embedding and a dialogue state representations and the 
output is a probability over classes (private or shared, believed to be true 
or false by the answerer).

## Details of each directory

The set up involves three main steps, each depending on one of the three
subdirectories here:

1. ```generating_propositions/```: turn VisDial QA pairs into propositions and
get their embeddings.
2. ```retrieving_dialogue_representations/```: extract the dialogue state 
representations of the original visual dialogue encoders.
3. ```main_task/```: run experiments with the probing classifier.

## How do I get set up?

Due to the different dependencies we used different Python environments for
each part. You can re-create the environments with conda using the ```yml````
files: 

- ```python_envs/environmentcoref.yml```: to replace pronouns on VisDial.
- ```python_envs/environment.yml```: to generate propositions.
- ```python_envs/main_task_environment.ym```: to get proposition embeddings 
and run the main experiments.

To retrieve dialogue representations, follow the instructions on the original
repository to build the environment.

## How do I replicate the results on the paper?

1. Follow the instructions on ```retrieving_dialogue_representations```.
2. Follow the instruction on ```generating_propositions```.
3. Follow the instructions on ```main_task```.


**Warning**: This version of the code was used for the revised version submitted on
March 30, after fixing a problem with the extraction of the 
dialogue representations. 

## Citation

If you use this work, please cite: 

```
@inproceedings{madureira-schlangen-2022-visual,
    title = "Can Visual Dialogue Models Do Scorekeeping? Exploring How Dialogue Representations Incrementally Encode Shared Knowledge",
    author = "Madureira, Brielen  and
      Schlangen, David",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.73",
    doi = "10.18653/v1/2022.acl-short.73",
    pages = "651--664",
}

```

## License

This work is licensed mainly under two licences:

- Code deriving from Murahari et al. (2019) is licensed under BSD.
- Other source code is licensed under MIT.

We use many Python libraries. See credits on each repository for details.
