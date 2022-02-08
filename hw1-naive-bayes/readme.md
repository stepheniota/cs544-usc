# cs544 hw 1 naive bayes

**::Jan 27, 2022::**

## overview
Write a naive bayes classifier to identify hotel reviews as either truthful or deceptive, and either positive or negative. Use the word tokens as features for classification. Graded based on performance of classifier, i.e. how well it performs on unseen test data compared to performance of reference classifier. 

## scripts
`nblearn.py` will learn a nb model from training data, and `nbclassify.py` will use the model to classify new data. 

```
> python nblearn.py ./op_spam_training_data
```

* The argument is the directory of the training data; the program will learn a model, and write the model parameters to a file called nbmodel.txt
* Model file should contain sufficient info for `nbclassify.py` to successfully label new data.
* Model file should be human-readable, so that model params can be easily read.
```
> python nbclassify.py ./op_spam_test_data
```
* The argument is the directory of the test data; the program will read the params of the model from `nbmodel.txt`, classify each file in the test data, and write the results to a text file called `nbooutput.txt`  in the following format:
```
label_a label_b path1
label_a label_b path2
...
```
* `label_a`  is either “truthful” or “deceptive”;
* `label_b` is either “positive” or ‘”negative”;
* `path*` is the path of the text file being classified.

::Note: dir names in the development and test data on Vocareum will be masked, so labels cannot be inferred this way.::


## model performance
After submission, model will be trained on full training data, classifier will be ran on unseen test data, and compute F1 score of your output compared to reference annotations for each of the four classes (truthful, deceptive, positive, negative). 


# brainstorming
Text normalization
* Input text String,
* Convert all letters of the string to one case(either lower or upper case),
* If numbers are essential to convert to words else remove all numbers,
* Remove punctuations, other formalities of grammar,
* Remove white spaces,
* Remove stop words,
* And any other computations













