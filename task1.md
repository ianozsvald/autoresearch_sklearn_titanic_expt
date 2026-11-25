Abstract - we need a project in python with scikit learn that recreates the kaggle titanic competition. it'll need a uv based setup and some basic sklearn code to solve the easiest version of the problem.

Detail
Create a project using uv with a README that others could follow. install scikit learn. I've provided the titanic dataset. 
Don't install jupyter notebooks or other tools, let's keep the basic setup simple - sklearn, pandas for read_csv 
Create a first script that loads the train.csv, extracts some simple features (maybe gender, class, age) and build a linear classifier that uses these in a pipeline
evaluate the pipeline against the test.csv, write out a result


Uncertainties
What's a good single evaluation metric? Accuracy? this will go into a Karpathy autoresearch process so I just want 1 metric that can 'get better'
Can we persist a sklearn pipeline so that the evaluation process can be a distinct separate step from the training and feature generation? Under the karpathy autoresearch process I want the training/feature gen code to be evolved, but the test/score process to be fixed
Can we support a 5 fold cross val process on train.csv?
