# HW 3 - Ciphertxt classification

In this assignment, we are tasked with a simple scenario for privacy-preserving NLU, text classification of *Ciphertxt*.
This scenario is quite realistic, as privacy-preservation is an important issue in machine learning. In real world applications (e.g. healthcare, banking), data may be redacted to ML engineers and data scientists (for NLU or other tasks).

We are provided with labeled training data of labeled ciphertext, and want to learn a model to predict the unlabeled test set.
All the sentences in the dataset are encrypted on the word level (hence, words in a sentence are still separated by white spaces).


## Models
My submitted method is
```FastText Embeddings + Linear Classifier```
.

Other methods I experimented with were
* RNN, LSTM and GRU in pytorch with skipgram and bag of words embeddings in gensim. ~80% accuracy on dev set.
* Naive Bayes with TF-IDF embeddings, both in sklearn. ~89% accuracy on dev set.
* Logistic regression with TF-IDF embeddings, both in sklearn. ~87% accuracy on dev set.

The submission file is `main.ipynb`. In the other files, please find other methods I experimented with that I thought were worthwhile sharing.

One thing worth noting: although this seemed to be a difficult task, the simpler, statistical ML methods seemed to heavily outperform more complicated (neural) approaches.

## Performance
Model performance was based on accuracy of unseen test set.
Grading was based on the ranking of our submitted predictions among all those in the class. I recieved full marks, and ranked in the top 20 in the class (~20/290).
