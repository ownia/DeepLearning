import numpy as np
from DeepLearningAndrewNg.NLP.week2.Emojify.emo_utils import *
import emoji
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')
    maxLen = len(max(X_train, key=len).split())
    index = 1
    print(X_train[index], label_to_emoji(Y_train[index]))
    # Overview of the Emojifier-V1
    