# KGHRec
Source code for KHGRec TPL recommendation system.

## Environment Requirement

The code has been tested running under Python 3.7.5. 

The required packages are as follows. However, later versions are acceptable.

CUDA == 10.2

torch == 1.11.0

numpy == 1.21.5

pandas == 1.3.5

scipy == 1.4.1

tqdm == 4.64.0

scikit-learn == 0.22

## Run the Codes

python main.py --data_name PyLib/ --path t06/ --use_pretrain 0 --attention 1 --knowledgegraph 1

parameter --path defines the subfolder name of the used dataset in the PyLib folder, which contains both the training set (named train.txt) and test set (named test.txt). Note that IDs are separated by "ONE SPACE" rather than "ONE TAB".

parameters "--attention" and "--knowledgegraph" define whether to use such functionalities (1) or not (0)

Currently, the dataset is split by 40%, i.e., 40% of TPLs are used as the test set while the left 60% of TPLs are used as the train set.

## Results 

Final results will be stored in subfolders of "result\PyLib\"
