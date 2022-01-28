""" Misc. functions that make my life easier. """
import numpy as np


class Params:
    def __init__(self):
        self.classes = ['truthful_positive', 'truthful_negative', 
                       'deceptive_positive', 'deceptive_negative']
        self.classes_a = ['truthful', 'deceptive']
        self.classes_b = ['positive', 'negative']
        self.class_labels = {
            'truthful_positive':np.array((1, 0, 0, 0)),
            'truthful_negative':np.array((0, 1, 0, 0)),
            'deceptive_positive':np.array((0, 0, 1, 0)),
            'deceptive_negative':np.array((0, 0, 0, 1))
        }
        self.vocablist = set(['excellent', 'wonderful', 'spacious', 'ave', 'shopping', 'perfect', 'walking', 'concierge', 'beautiful', 'blocks', 'rate', 'highly', 'mile', 'distance', 'enjoyed', 'navy', 'river', 'pier', 'reviews', 'lake', 'loved', 'always', 'restaurants', 'avenue', 'quiet', 'huge', 'coffee', 'fantastic', 'dinner', 'everyone', 'magnificent', 'deal', 'street', 'within', 'love', 'return', 'buffet', 'etc', 'amazing', 'corner', 'views', 'size', 'priceline', 'pillows', 'minute', 'fabulous', 'home', 'lovely', 'lots', 'impressed', 'easy', 'tub', 'early', 'building', 'without', 'north', 'comfy', 'end', 'gym', 'block', 'screen', 'fun', 'located', 'walked', 'access', 'eat', 'absolutely', 'doorman', 'year', 'pump', 'less', 'tower', 'areas', 'construction', 'full', 'expensive', 'open', 'places', 'windows', 'future', 'probably', 'including', 'upgraded', 'conference', 'modern', 'valet', 'fresh', 'years', 'friends', 'couple', 'happy', 'returned', 'near', 'ate', 'across', 'superb', 'millenium', 'week', 'especially', 'yes', 'pleasant', 'complimentary', 'convenient', 'others', 'separate', 'included', 'cool', 'name', 'drinks', 'told', 'called', 'reservation', 'call', 'phone', 'minutes', 'nothing', 'charge', 'star', 'manager', 'bad', 'elevators', 'disappointed', 'dirty', 'rude', 'later', 'elevator', 'gave', 'looked', 'times', 'line', 'problem', 'second', 'card', 'money', 'finally', 'something', 'toilet', 'try', 'least', 'pay', 'put', 'given', 'hour', 'late', 'let', 'someone', 'broken', 'charged', 'problems', 'come', 'walls', 'seemed', 'credit', 'requested', 'wall', 'towels', 'sleep', 'tiny', 'wouldnt', 'almost', 'tell', 'waiting', 'instead', 'actually', 'expected', 'youre', 'fact', 'worst', 'may', 'bill', 'issue', 'carpet', 'outside', 'ok', 'housekeeping', 'hear', 'four', 'else', 'unfortunately', 'despite', 'clerk', 'decor', 'entire', 'heard', 'wifi', 'hours', 'poor', 'uncomfortable', 'slow', 'either', 'reception', 'cold', 'furniture', 'overpriced', 'waited', 'tried', 'renovation', 'saw', 'minibar', 'security', 'working', 'luxurious', 'vacation', 'luxury', 'fitness', 'millennium', 'atmosphere', 'elegant', 'relaxing', 'spa', 'wanted', 'center', 'dining', 'regency', 'delicious', 'flat', 'accommodations', 'visiting', 'enjoy', 'package', 'pet', 'meeting', 'relax', 'leave', 'indoor', 'wine', 'must', 'quick', 'wedding', 'towers', 'reasonable', 'gorgeous', 'courteous', 'treated', 'favorite', 'professional', 'awesome', 'greeted', 'kids', 'helped', 'polite', 'liked', 'heart', 'decorated', 'dog', 'id', 'spent', 'workout', 'pets', 'traveling', 'incredible', 'pleased', 'along', 'meal', 'bags', 'town', 'bring', 'beautifully', 'moment', 'windy', 'getaway', 'recommended', 'believe', 'speed', 'comfort', 'choice', 'fast', 'thank', 'immediately', 'pillow', 'needs', 'book', 'wireless', 'smell', 'smelled', 'ready', 'cleaned', 'smoke', 'website', 'air', 'sheets', 'checking', 'expect', 'spend', 'terrible', 'wrong', 'turned', 'cleaning', 'hilton', 'rather', 'wont', 'nonsmoking', 'reserved', 'loud', 'making', 'special', 'reservations', 'online', 'supposed', 'seems', 'offered', 'key', 'anniversary', 'ordered', 'luggage', 'hallway', 'double', 'coming', 'complain', 'smoking', 'somewhere', 'prices', 'hair', 'received', 'already', 'counter', 'girl', 'tired', 'ended', 'kind', 'half', 'brought'])


        #self.stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'dont', 'should', 'now', 'hotel', 'chicago','rock', 'michigan', 'th'])

        self.stopwords = set(['th', 'hotel', 'michigan', '', 'the', 'and', 'a', 'to', 'was', 'i', 'in', 'we', 'for', 'of', 'hard', 'rock', 'chicago', 'room', 'is', 'it', 'at', 'with', 'on', 'very', 'were', 'this', 'from', 'our', 'had', 'that', 'stay', 'my', 'but', 'you', 'staff', 'have', 'there', 'all', 'would', 'as', 'so', 'not', 'stayed', 'are', 'rooms', 'they', 'nice', 'be', 'service', 'us', 'night', 'just', 'again', 'an', 'bed', 'one', 'if', 'out', 'when', 'like', 'will', 'only', 'get', 'by', 'or', 'me', 'which', 'could', 'time', 'up', 'back', 'hotels', 'even', 'great', 'location', 'clean', 'here', 'view', 'about', 'also', 'good', 'bathroom', 'got', 'breakfast', 'well', 'some', 'desk', 'really', 'city', 'no', 'has', 'than', 'everything', 'lobby', 'other', 'right', 'can', 'its', 'area', 'front', 'bar', 'go', 'two', 'their', 'did', 'day', 'place', 'more', 'first', 'next', 'food', 'didnt', 'away', 'your', 'too', 'what', 'business', 'experience', 'trip', 'made', 'say', 'because', 'few', 'take', 'after', 'price', 'beds', 'am', 'been', 'any', 'over', 'check', 'weekend', 'never', 'he', 'around', 'internet', 'while', 'going', 'much', 'most', 'do', 'better', 'who', 'being', 'before', 'make', 'want', 'find', 'way', 'ever', 'where'])

        self.vocablist = self.vocablist - self.stopwords

def precision(true_pos, false_pos):
    return true_pos / (true_pos + false_pos)

def recall(true_pos, false_neg):
    return true_pos / (true_pos + false_neg)

def accuracy_score(y_true, y_pred):
    score = sum([true == pred for true, pred in zip(y_true, y_pred)])
    return score / len(y_true)


def f1_score(y_true, y_pred, n_classes=4):
    """ Calculates the mean f1 score across the four classes. """
    #from nbmodel import NaiveBayesClassifer
    scores = np.zeros(4, np.float64)
    classes = ['truthful', 'deceptive', 'positive', 'negative']
    for i, cls in enumerate(classes):
        true_pos, false_pos, false_neg = 0, 0, 0
        for true_labels, pred_labels in zip(y_true, y_pred):
            idx = 0 if i == 0 or i == 1 else 1
            true = true_labels.split('_')[idx]
            pred = pred_labels.split('_')[idx]
            if true == pred and true == cls:
                true_pos += 1
            elif true != pred and pred == cls:
                false_pos += 1
            elif true != pred and true == cls:
                false_neg += 1
        p = precision(true_pos, false_pos)
        r = recall(true_pos, false_neg)
        scores[i] = (2 * p * r) / (p + r)
    return np.mean(scores)

def save_predictions(data_cls, y_preds, file_name='nboutput.txt'):
    with open(file_name, 'w') as f:
        for pred, path in zip(y_preds, data_cls.paths):
            labels = pred.split('_')
            f.write(f'{labels[0]} \t {labels[1]} \t {path} \n')

