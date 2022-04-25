# HW 4 - Precondition Inference

In this assignment, we are given a set of sentence pairs, the first one a precondition and the second a statement. The goal is to develop a natural language reasoner to decide whether the precondition enables or disables the statement.

Examples
```
precondition: "Water is clean."
statement: "A glass can be used to drink water."
label: "enable"

precondition: "The glass is broken."
statement: "A glass can be used to drink water."
label: "disable"
```

## Models
My submission model was `fine-tuning roberta-large-mnli`, and can be found in the `main.ipynb`.

I finetuned the pretrained model from huggingface and used it as a cross-encoder. For each sentence pair, I joined the two sentences (separated by a space), and used this as input to the encoder.

I found the model trained quite quickly, and although the dev set accuracy continued to increase in the first ~10 epochs of training, the dev loss also increased. To ensure the model was not overfitting to both the train and dev set, I stopped training early after 3 epochs.

Other models I tried include
* roberta-base
* roberta-large
* bert-base-uncased
* bert-large-uncased
* bart-large

I trained the model on Google Colab with GPU acceleration.

## Performance
Performance of the model is determined by **accuracy** metric. Grading is done by leaderboard protocol. I received 95% credit, and placed in the top 25% of the class.

## Dataset
The dataset used is internal to the CSCI 544 class in Spring 2022. I am not allowed to distribute the dataset (or trained model most probably) at all, under severe penalty of failing the course.
