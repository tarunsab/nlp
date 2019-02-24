import sys
import csv
import re
import numpy
import torch
import matplotlib.pyplot as plt
from WordRNN import RNN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix

def case_normalization(msg):
    return msg.strip().lower()

def remove_delimiters(msg):
    return ' '.join(filter(None, re.split('[ ,.!]', msg)))

def pre_process_msg(msg):
    return remove_delimiters(case_normalization(msg))

def parse_data(input_path):
    with open(input_path, 'r') as fd:
        rd = csv.reader(fd, delimiter='\t', quotechar='"')
        fields = rd.next()
        data = {}
        for field in fields:
            data[field] = []
        for row in rd:
            for i, token in enumerate(row):
                data[fields[i]].append(token)
        return data

if __name__ == '__main__':
    if len(sys.argv) == 3:
        # Parse the csv files into appropriate datasets
        training_dataset = parse_data(sys.argv[1])
        testing_dataset = parse_data(sys.argv[2])

        id_values = training_dataset['id']
        tweet_data = map(pre_process_msg, training_dataset['tweet'])
        off_categories = training_dataset['subtask_a']

        break_point = len(tweet_data) // 10
        training_data = tweet_data[:-break_point]
        training_target = off_categories[:-break_point]
        validation_data = tweet_data[-break_point:]
        validation_target = off_categories[-break_point:]

        # Vectorizer that prioritises words based on frequency of occurence
        vectorizer1 = CountVectorizer (
            stop_words="english",
            preprocessor=pre_process_msg
        )

        # Vectorizer that prioritises words based on TF-IDF semantics
        vectorizer2 = TfidfVectorizer (
            stop_words="english",
            preprocessor=pre_process_msg
        )

        # Compose feature vectors from Bag of Words and vectorizer.
        # fit_transform learns the word vocabulary and transforms
        # the data into a one hot vector
        training_features = vectorizer2.fit_transform(training_data)
        validation_features = vectorizer2.transform(validation_data)

        pred = validation_target

        # Write predictions into output csv file
        csvData = []
        for i, id_val in enumerate(id_values[-break_point:]):
            csvData.append([id_val, pred[i]])
        with open('predictions.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)


        # Temporary plot of Confusion Matrix
        cm = confusion_matrix(validation_target, pred)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        tick_marks = numpy.arange(len(classNames))
        plt.xticks(tick_marks, classNames)
        plt.yticks(tick_marks, classNames)
        plt.title('Offensive or Not Offensive Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i][j]))
        plt.show()

    else:
        print("Usage: python detect_offensive.py <training_data_file> <testing_data_file>")
