# Extracting Dialogue State Representations

This directory contains the script we used to extract the dialogue state 
representations for the paper. It is an adaptation from the original 
code (```train.py``` and ```eval.py``` available at 
[this repository](https://github.com/vmurahari3/visdial-diversity/tree/23d8dfb4a483353e30e994b7209b1ffe90a9cd37).

Steps we took to extract the dialogue representations.

1. Clone the Murahari et al. (2019)'s repository into this folder. 
We used [this version](https://github.com/vmurahari3/visdial-diversity/tree/23d8dfb4a483353e30e994b7209b1ffe90a9cd37).

2. Follow their instructions to create the conda environment and activate it.

3. [Download](https://github.com/vmurahari3/visdial-diversity/tree/23d8dfb4a483353e30e994b7209b1ffe90a9cd37#pre-trained-checkpoints) 
their preprocessed data and their checkpoints.

3. Put the file ```extract_dialogue_states.py```  into the ```visdial-diversity``` 
directory.

4. Make a directory called ```extracted_states```.

5. Run both scripts.
````
python3 extract_dialogue_states.py
````

# IMPORTANT

Because of the different dialogue lenghts in the test set and the
batch padding, it also creates 'fake' representations for the un-used turns
in the test set. The main experiments scripts take care of not using them
because it only accesses the valid turns (see tests.py). However, any other
use of this script should be aware of that!


# License
The original code is licensed under BSD, so the adapted scripts follow the
same license.