# HW 2 - Hidden Markov model

In this assignment, we write a Hidden Markov model part-of-speech (POS) tagger for Italian, Japanese, and a surprise language.
The training data are tokenized and tagged; the test data will be tokenized, and our learned HMM is tasked with adding tags.

## Model
1. `python hmmlearn.py /path/to/train_data`  will learn a HMM from the training data.
2. hmmlearn.py will write the learned model parameters to a file called `hmmmodel.txt`, in a human-readable format, containing sufficient info to successfully tag new data.
3. `python hmmdecode.py /path/to/test_data` will load model params from hmmmodel.txt and tag each word in the test data.
4. `hmmdecode.py` will write decoding results to a text file called `hmmoutput.txt` in the same format as the training data i.e.,
```
    word1/TAG1 word2/TAG2 ...
```

## Performance

Model will be trained separately for each language on a combination of the training and dev data. Learned model will be evaluated on unseen test data, and output **accuracy** will be compared to a reference annotation.
