#!/bin/bash
# Download VisDial dataset

cd data
mkdir visual_dialog

cd visual_dialog
wget https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=0
unzip visdial_1.0_train.zip?dl=0
rm -rf visdial_1.0_train.zip?dl=0

wget https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0
unzip visdial_1.0_val.zip?dl=0
rm -rf visdial_1.0_val.zip?dl=0

wget https://www.dropbox.com/s/o7mucbre2zm7i5n/visdial_1.0_test.zip?dl=0
unzip visdial_1.0_test.zip?dl=0
rm -rf visdial_1.0_test.zip?dl=0
