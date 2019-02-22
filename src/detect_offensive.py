import sys
import csv
import re
import numpy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

def case_normalization(msg):
    return msg.strip().lower()

def get_word_sequence(msg):
    return filter(None, re.split('[ ,.!]', msg))

def remove_stopwords(msg, stopwords):
    non_stopwords = filter(lambda word : word not in stopwords, get_word_sequence(msg))
    return ' '.join(non_stopwords)

def pre_process_msg(msg):
    return ' '.join(get_word_sequence(case_normalization(msg)))

def parse_data(input_path):
    with open(input_path) as fd:
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
    if len(sys.argv) == 2:
        input_path = sys.argv[1]
        dataset = parse_data(input_path)
        tweet_data = map(pre_process_msg, dataset['tweet'])
        target_data = dataset['subtask_a']

        break_point = len(tweet_data) // 10
        training_data = tweet_data[:-break_point]
        training_target = target_data[:-break_point]
        testing_data = tweet_data[-break_point:]
        testing_target = target_data[-break_point:]

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

        # Compose feature vectors from Bag of Words and vectorizer
        training_features = vectorizer2.fit_transform(training_data)
        testing_features = vectorizer2.transform(testing_data)

        # Use support vector classification to accompany the large
        # Bag of Words feature vectors used
        model = LinearSVC()
        model.fit(training_features, training_target)
        pred = model.predict(testing_features)

        cm = confusion_matrix(testing_target, pred)

        # Temporary plot of Confusion Matrix
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
        print("Usage: python detect_offensive.py <data_file>")
