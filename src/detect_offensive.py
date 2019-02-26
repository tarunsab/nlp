from RNN import RNN
import sys
import csv
import re
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def case_normalization(msg):
    return msg.strip().lower()

def remove_delimiters(msg):
    return filter(None, re.split('[ ,.!]', msg))

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

def get_word2idx(tokenized_corpus):
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
    word2idx = {w : idx + 1 for (idx, w) in enumerate(vocabulary)}
    word2idx['<pad>'] = 0
    return word2idx

def get_model_inputs(tokenized_corpus, word2idx, max_len, labels = []):
    vectorized_sents = [[word2idx[tok] for tok in sent if tok in word2idx] for sent in tokenized_corpus]
    # Create a tensor of a fixed size filled with zeroes for padding
    sent_tensor = Variable(torch.zeros((len(vectorized_sents), max_len))).long()
    sent_lengths = [len(sent) for sent in vectorized_sents]
    # Fill it with vectorized sentences
    for idx, (sent, sentlen) in enumerate(zip(vectorized_sents, sent_lengths)):
        sent_tensor[idx, :sentlen] = LongTensor(sent)
    label_tensor = FloatTensor(labels)
    return sent_tensor, label_tensor

def predict_value(input_tensor, num_of_classes):
    prediction = torch.sigmoid(model(input_tensor))
    return int(prediction.item() * num_of_classes)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python detect_offensive.py <training_data_file> <testing_data_file>")
        sys.exit(1)

    # Parse the csv files into appropriate datasets
    model_dataset = parse_data(sys.argv[1])
    testing_dataset = parse_data(sys.argv[2])

    model_corpus = map(pre_process_msg, model_dataset['tweet'])
    testing_corpus = map(pre_process_msg, testing_dataset['tweet'])

    # Generate classes for each label
    labels = model_dataset['subtask_a']
    class2label = []
    label2class = {}
    for label in labels:
        if label not in label2class:
            label2class[label] = len(class2label)
            class2label.append(label)
    classes = map(lambda label : label2class[label], labels)

    # Fix seeds for consistent results
    SEED = 234
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Split the model dataset into training data and validation data
    model_size = len(model_corpus)
    validation_size = model_size // 10
    start = random.randint(0, model_size) // 10
    end = start + validation_size
    training_corpus = model_corpus[:start] + model_corpus[end:]
    training_target = classes[:start] + classes[end:]
    validation_corpus = model_corpus[start : end] + model_corpus[:end - model_size]
    validation_target = classes[start : end] + classes[:end - model_size]

    # Other pre-processing helper variables
    USE_CUDA = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

    # Get word2idx and intialise tensors
    word2idx = get_word2idx(training_corpus)
    max_len = np.max(np.array([len(sent) for sent in training_corpus]))
    training_data_tensor, training_label_tensor = get_model_inputs(training_corpus, word2idx, max_len, training_target)
    validation_data_tensor, _ = get_model_inputs(validation_corpus, word2idx, max_len)
    testing_data_tensor, _ = get_model_inputs(testing_corpus, word2idx, max_len)

    # Hyperparameters
    EMBEDDING_SIZE = 100
    HIDDEN_LAYER_SIZE = 100
    LR = 0.01
    EPOCH = 10
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    BATCH_SIZE = 100

    # Initialise RNN model
    model = RNN(len(word2idx), EMBEDDING_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    if USE_CUDA:
        model = model.to('cuda')

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train the model
    for epoch in range(1, EPOCH + 1):
        print("On epoch " + str(epoch) + " out of " + str(EPOCH))
        total_loss = 0
        losses = []
        model.train()
        print("Starting batch loop...")
        for i in range(0, len(training_data_tensor), BATCH_SIZE):
            model.zero_grad()
            inputs = training_data_tensor[i : i + BATCH_SIZE].permute(1, 0)
            targets = training_label_tensor[i : i + BATCH_SIZE].unsqueeze(1)
            preds = model(inputs)
            loss = loss_function(preds, targets)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if i > 0 and i % 500 == 0:
                print("[%02d/%d] mean_loss : %0.2f, Perplexity : %0.2f" % (epoch, EPOCH, np.mean(losses), np.exp(np.mean(losses))))
                losses = []
        print("Finished batch loop...")

    # Test the model on the validation data
    model.eval()
    preds = []
    print("PREDICTING LABELS ON VALIDATION DATA")
    for validation_input in validation_data_tensor:
        model.zero_grad()
        predicted_class = predict_value(validation_input.unsqueeze(1), len(class2label))
        preds.append(predicted_class)
    print("COMPLETED VALIDATION DATA LABEL PREDICTION")

    # Print accuracy and macro f1 measure for validation predictions
    acc = accuracy_score(validation_target, preds)
    f1 = f1_score(validation_target, preds, average = 'macro')
    print("Accuracy: " + str(acc))
    print("Macro F1 Average: " + str(f1))

    # Plot confusion matrix of validation predictions
    cm = confusion_matrix(validation_target, preds)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    tick_marks = np.arange(len(classNames))
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

    # Predict labels for testing data
    model.eval()
    preds = []
    print("PREDICTING TESTING LABELS NOW")
    for testing_input in testing_data_tensor:
        model.zero_grad()
        predicted_label = class2label[predict_value(testing_input.unsqueeze(1), len(class2label))]
        preds.append(predicted_label)
    print("FINISHED PREDICTING TESTING LABELS")

    # Write predictions for testing data into output csv file
    csv_data = []
    id_values = testing_dataset['id']
    for i, id_val in enumerate(id_values):
        csv_data.append([id_val, preds[i]])
    with open('predictions.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
