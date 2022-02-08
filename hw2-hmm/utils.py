
def accuracy_score(y, y_hat):
    score = sum(i == j for i, j in zip(y, y_hat)) / len(y)
    return score
    