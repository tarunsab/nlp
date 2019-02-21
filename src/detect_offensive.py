import sys
import csv
from nltk.corpus import stopwords

def case_normalization(msg):
    return msg.lower()

def remove_stopwords(msg, stopwords):
    return filter(lambda word : word not in stopwords, msg.split())

def pre_process_msg(msg):
    stop_words = set(stopwords.words('english'))
    return ' '.join(remove_stopwords(case_normalization(msg), stop_words))

def parse_data(input_path):
    with open(input_path) as fd:
        rd = csv.reader(fd, delimiter='\t', quotechar='"')
        fields = rd.next()
        data = {}
        for field in fields:
            data[field] = []
        for row in rd:
            for i, cell in enumerate(row):
                data[fields[i]].append(cell)
        return data

if __name__ == '__main__':
    if len(sys.argv) == 2:
        input_path = sys.argv[1]
        dataset = parse_data(input_path)
        tweet_data = dataset['tweet']
        print(tweet_data[:10])
        print("----------------")
        tweet_data = map(pre_process_msg, tweet_data)
        print(tweet_data[:10])
    else:
        print("Usage: python detect_offensive.py <data_file>")
