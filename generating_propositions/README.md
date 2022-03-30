# README: Turning VisDial QA pairs into propositions

These scripts were used to manipulate VisDial QA pairs into propositions that 
either entail or contradict what the answerer thinks about the image. 
It is based on 34 rules that take common word patterns and POS tags into 
account.

The generated datasets were used in the evaluation proposed in the short paper 
"Can Visual Dialogue Models do Scorekeeping? Exploring how Dialogue
Representations Incrementally Encode Shared Knowledge", presented at ACL 2022.

Disclaimer 1: The rules are not perfect. They work quite well for many cases, 
but problems with countable/uncountable nouns, singular and plural and 
other linguistic aspects occur in some cases.

Disclaimer 2: We decided to try to skip dialogues that contain words that may 
be inappropriate, to avoid generating inappropriate propositions. The approach 
is not refined, we cannot guarantee that all problematic dialogues were 
ignored and it also ends up ignoring some dialogues that are fine. 
Still, all propositions reflect the perspective and judgements of the 
crowdworkers who were questioner and answerer. Entailments/contradictions of 
QA pairs reflect their opinion and not necessarily the truth about an image or 
reality. While this is fine for the type of analysis done in the paper, 
these propositions should not be used for other purposes of inference 
on the images.

Disclaimer 3: The rule-based generation cannot handle every QA pair, only 
those that end up being caught by one of the rules.

See also the 'Scope and Limitations' section on the paper.

## Data

We use the VisDial dataset 1.0 available [here](https://visualdialog.org/data) 
and licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Reference: Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, 
José MF Moura, Devi Parikh, and Dhruv Batra. 2017a. Visual dialog. In 
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 
pages 326–335.

## Set Up

Due to conflicting versions of SpaCy, we used different conda environments:

- environmentcoref.yml builds the environment for step 2.
- environment.yml builds the environment for step 3.
- For step 7, we used the same environment used for the main experiments.

## Replicating the results

1. Run ```sh get_visdial.sh``` to download the data.
2. Run ```python3 replace_pronouns.py``` to run a coference resolution model 
and replace some pronouns by the corresponding entities. 
(This step was done but it is optional in general.)
3. Run ```python3 main.py``` to generate the propositions.
4. Run the Jupyter notebook ```filtering_probes.ipynb``` to downsample the 
captions and balance the training set.
5. Run the Jupyter notebook ```analysis_final_probes.ipynb``` to evaluate 
the propositions.
6. Run ```python3 collect_testset_lens.py``` to extract the varying dialogue 
lengths on the test set.
7. Run ```python3 get_probe_embeddings.py``` to retrieve the sentence 
embeddings for all propositions.

To run step 3, this may be necessary:
```
sudo apt-get update
sudo apt-get install libmysqlclient-dev
````

## Credits

We used [Allennlp's coreference resolution model](https://demo.allennlp.org/coreference-resolution) 
and some other libraries: 
[inflect](https://pypi.org/project/inflect/), 
[numpy](https://numpy.org/), 
[pattern](https://github.com/clips/pattern),
[sentence-transformers](https://huggingface.co/sentence-transformers),
[spaCy](https://spacy.io/), 
[tqdm](https://pypi.org/project/tqdm/), as well as standard libraries.
The list of words to be filtered is partially derived from the
[better-profanity](https://pypi.org/project/better-profanity/) Python library.

## Citation

If you use this work, please cite: (tbd)

## License
Source code licensed under MIT.
