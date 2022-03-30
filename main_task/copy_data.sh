#!/bin/sh

echo 'Copying dialogue lens file.'
mkdir data/propositions
cp ../generating_propositions/propositions/visdial_1.0_test_dialogueLens.txt data/visdial_1.0_test_dialogueLens.txt 

echo 'Copying propositions files.'
mkdir data/embeddings
cp ../generating_propositions/propositions/downsampled-propositions_test.json data/propositions/propositions_test.json
cp ../generating_propositions/propositions/downsampled-propositions_val.json data/propositions/propositions_val.json
cp ../generating_propositions/propositions/downsampled-balanced-propositions_train.json data/propositions/propositions_train.json

echo 'Copying embeddings files.'
cp -r ../generating_propositions/propositions/embeddings/* data/embeddings

# Did this on Jupyter because too slow.
echo 'Copying representations files.'
mkdir data/representations
cp -r ../retrieving_dialogue_representations/visdial-diversity/extracted_states/* data/representations
