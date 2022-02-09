"""Helper functions."""

def accuracy_score(y, y_hat):
    score = sum(i == j for i, j in zip(y, y_hat)) / len(y)
    return score


def compare_output(manual_path, tagged_path):
    raise NotImplementedError


def write_output(X, y_hat, file_name='hmmoutput.txt'):
    with open(file_name, mode='w') as f:
        for word_sentence, tag_sentence in zip(X, y_hat):
            for i, (word, tag) in enumerate(zip(word_sentence, tag_sentence)):
                f.write(word + '/' + tag)
                if len(word_sentence) - 1 == i:
                    f.write('\n')
                else:
                    f.write(' ')