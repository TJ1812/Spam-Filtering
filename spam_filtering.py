from os import listdir
from math import log,exp
import copy
import pandas as pd
        
'''
Naive Bayes

-First, create a hash map for our two classes : spam and ham. We traverse every word in file and add it in hash map of spam or ham. 
-Keep track of the distinct words and total words in both the maps. 


-After setting up our hash maps we are done with the training step. 
-Now we proceed to the testing.
-For each email/file, create a hash map called attribute map.

-Use the Bayesian Formula, assuming that our attributes are conditionally independent (given the class).
-Log is used to avoid underflow.

-An example for calculating log probabilities will be as follows.

p(ham|file) = p(file|ham)p(ham) 
p(file|ham) = sum(#word log((#word in ham + 1)/(distinct words + total words in ham)))

-Add one laplace smoothing is done which implies that we see each words once additionally.

'''

'''

Logistic Regression

-Take 500 most frequently appearing attributes from both ham and spam emails.
-This will be the attributes.
-Generate the data.
-Initialize weight vectors, l-rate, regularization constant, steps.
-Train Phase :- Stochastic Gradient descent with L2 Regularization.
-Test

'''


#Input :- Path to ham and spam files
#Output :- Hash maps of words for ham and spam and related parameters
def preprocess(path_ham, path_spam):
    ham_map = {}
    spam_map = {}
    total_words_in_ham = 0
    total_words_in_spam = 0
    distinct_words = 0
    files = listdir(path_ham)
    total_ham = len(files)
    for f in files:
        file = open(path_ham+'/'+f, 'r')
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                total_words_in_ham += 1
                if word not in ham_map:
                    distinct_words += 1
                    ham_map[word] = 1
                else:
                    ham_map[word] += 1
        file.close()

    files = listdir(path_spam)
    total_spam = len(files)
    for f in files:
        file = open(path_spam+'/'+f, 'r', encoding='utf8', errors='ignore')
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                total_words_in_spam += 1
                if word not in spam_map:
                    spam_map[word] = 1
                    if(word not in ham_map):
                        distinct_words += 1
                else:
                    spam_map[word] += 1
        file.close()
    
    return ham_map, spam_map, total_words_in_ham, total_words_in_spam, distinct_words, total_spam+total_ham


#Input: Path of ham ans spam files to be tested and parameters obtained from preprocessing
#Output: Accuracies
def test_naive_bayes(path_ham, path_spam, total_words_in_ham, total_words_in_spam, distinct_words):
    path_test = path_ham
    files = listdir(path_test)
    total_test_ham = len(files)
    correct_predictions_ham = 0
    for f in files:
        pred_ham = 0
        pred_spam = 0
        attribute_map = {}
        file = open(path_test+'/'+f, 'r', encoding='utf8', errors='ignore')
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                if(word not in attribute_map):
                    attribute_map[word] = 1
                else:
                    attribute_map[word] += 1
        for attr in attribute_map:
            if(attr in ham_map):
                pred_ham += attribute_map[attr]*log((ham_map[attr] + 1)/(total_words_in_ham + distinct_words))
            else:
                pred_ham += attribute_map[attr]*log((1)/(total_words_in_ham + distinct_words))
            if(attr in spam_map):
                pred_spam += attribute_map[attr]*log((spam_map[attr] + 1)/(total_words_in_spam + distinct_words))
            else:
                pred_spam += attribute_map[attr]*log((1)/(total_words_in_spam + distinct_words))
        if(pred_ham >= pred_spam):
            correct_predictions_ham += 1
        file.close()
        
    ham_accuracy = (correct_predictions_ham/total_test_ham)*100

    
    path_test = path_spam
    files = listdir(path_test)
    total_test_spam = len(files)
    correct_predictions_spam = 0
    for f in files:
        pred_ham = 0
        pred_spam = 0
        attribute_map = {}
        file = open(path_test+'/'+f, 'r', encoding='utf8', errors='ignore')
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                if(word not in attribute_map):
                    attribute_map[word] = 1
                else:
                    attribute_map[word] += 1
        for attr in attribute_map:
            if(attr in ham_map):
                pred_ham += attribute_map[attr]*log((ham_map[attr] + 1)/(total_words_in_ham + distinct_words))
            else:
                pred_ham += attribute_map[attr]*log((1)/(total_words_in_ham + distinct_words))
            if(attr in spam_map):
                pred_spam += attribute_map[attr]*log((spam_map[attr] + 1)/(total_words_in_spam + distinct_words))
            else:
                pred_spam += attribute_map[attr]*log((1)/(total_words_in_spam + distinct_words))
        if(pred_ham <= pred_spam):
            correct_predictions_spam += 1
        file.close()
    spam_accuracy = (correct_predictions_spam/total_test_spam)*100

    return ham_accuracy, spam_accuracy, ((correct_predictions_ham+correct_predictions_spam)/(total_test_ham+total_test_spam))

#Same as above function, only the stop words are removed.
def test_naive_bayes_postfiltering(path_ham, path_spam, total_words_in_ham, total_words_in_spam, distinct_words, stopwords):
    path_test = path_ham
    files = listdir(path_test)
    total_test_ham = len(files)
    correct_predictions_ham = 0
    for f in files:
        pred_ham = 0
        pred_spam = 0
        attribute_map = {}
        file = open(path_test+'/'+f, 'r', encoding='utf8', errors='ignore')
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                if(word not in attribute_map):
                    attribute_map[word] = 1
                else:
                    attribute_map[word] += 1
        
        for word in stopwords:
            if(word in attribute_map):
                attribute_map.pop(word)

        for attr in attribute_map:
            if(attr in ham_map):
                pred_ham += attribute_map[attr]*log((ham_map[attr] + 1)/(total_words_in_ham + distinct_words))
            else:
                pred_ham += attribute_map[attr]*log((1)/(total_words_in_ham + distinct_words))
            if(attr in spam_map):
                pred_spam += attribute_map[attr]*log((spam_map[attr] + 1)/(total_words_in_spam + distinct_words))
            else:
                pred_spam += attribute_map[attr]*log((1)/(total_words_in_spam + distinct_words))
        if(pred_ham >= pred_spam):
            correct_predictions_ham += 1
        file.close()
        
    ham_accuracy = (correct_predictions_ham/total_test_ham)*100

    
    path_test = path_spam
    files = listdir(path_test)
    total_test_spam = len(files)
    correct_predictions_spam = 0
    for f in files:
        pred_ham = 0
        pred_spam = 0
        attribute_map = {}
        file = open(path_test+'/'+f, 'r', encoding='utf8', errors='ignore')
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                if(word not in attribute_map):
                    attribute_map[word] = 1
                else:
                    attribute_map[word] += 1
        for attr in attribute_map:
            if(attr in ham_map):
                pred_ham += attribute_map[attr]*log((ham_map[attr] + 1)/(total_words_in_ham + distinct_words))
            else:
                pred_ham += attribute_map[attr]*log((1)/(total_words_in_ham + distinct_words))
            if(attr in spam_map):
                pred_spam += attribute_map[attr]*log((spam_map[attr] + 1)/(total_words_in_spam + distinct_words))
            else:
                pred_spam += attribute_map[attr]*log((1)/(total_words_in_spam + distinct_words))
        if(pred_ham <= pred_spam):
            correct_predictions_spam += 1
        file.close()
    spam_accuracy = (correct_predictions_spam/total_test_spam)*100

    return ham_accuracy, spam_accuracy, ((correct_predictions_ham+correct_predictions_spam)/(total_test_ham+total_test_spam))


#Generating the data from ham and spam files for logistic regression.
def generate_data_logreg(ham_map, spam_map, path_ham, path_spam):
    temp = sorted(ham_map, key=lambda x:ham_map[x], reverse=True)
    temp1 = sorted(spam_map, key=lambda x:spam_map[x], reverse=True)
    training_attr = {}
    
    for i in range(500):
        if(len(temp[i]) > 2 and ord(temp[i][0]) > 60):
            training_attr[temp[i]] = []
            
    for i in range(500):
        if(len(temp1[i]) > 2 and ord(temp1[i][0]) > 60):
            training_attr[temp1[i]] = []
    training_attr[0] = []
    
    files = listdir(path_ham)
    for f in files:
        file = open(path_ham+'/'+f, 'r')
        attribute_map = {}
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                if(word not in attribute_map):
                    attribute_map[word] = 1
                else:
                    attribute_map[word] += 1
        for data in training_attr:
            if(data == 0):
                training_attr[data].append(1)
            else:
                if(data in attribute_map):
                    training_attr[data].append(attribute_map[data])
                else:
                    training_attr[data].append(0)
    file.close()

    files = listdir(path_spam)
    for f in files:
        file = open(path_spam+'/'+f, 'r', encoding='utf8', errors='ignore')
        attribute_map = {}
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                if(word not in attribute_map):
                    attribute_map[word] = 1
                else:
                    attribute_map[word] += 1
        for data in training_attr:
            if(data == 0):
                training_attr[data].append(1)
            else:
                if(data in attribute_map):
                    training_attr[data].append(attribute_map[data])
                else:
                    training_attr[data].append(0)
        file.close()

    df = pd.DataFrame.from_dict(training_attr)
    return df, training_attr

def sigmoid(z):
    return (1/(1+exp(-z)))

#Stochastic gradient descent with L2 regularization
def train_log_reg(df, w, steps, learning_rate, regularization, total_training_examples):
    for t in range(steps):
        total = 0
        for i in range(total_training_examples):
            temp = {}
            z = 0
            
            x = df[i:i+1]
            for att in x:
                z += (float(x[att])*w[att])
            
            pred = sigmoid(z)
            error = 0
            if(i <= 340):
                error = 1 - pred
            else:
                error = 0 - pred
            total += error**2
        
            for att in w:
                temp[att] = w[att] + learning_rate*error*pred*(1-pred)*float(x[att]) - regularization*learning_rate*w[att]
            w = copy.deepcopy(temp)
        #print("step:- "+str(t)+", error:- "+str(total))
    return w

def test_log_reg(w, path_ham, path_spam):
    path_test = path_spam
    files = listdir(path_test)
    total_test_spam = len(files)
    correct_predictions_spam = 0
    for f in files:
        attribute_map = {}
        file = open(path_test+'/'+f, 'r', encoding='utf8', errors='ignore')
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                if(word in training_attr):
                    if(word not in attribute_map):
                        attribute_map[word] = 1
                    else:
                        attribute_map[word] += 1
        x = {}
        for attr in training_attr:
            if(attr in attribute_map):
                x[attr] = attribute_map[attr]
            else:
                x[attr] = 0
        total = w[0]
        for att in x:
            total += (float(x[att])*w[att])

        if((1/(1+exp(-total))) < 0.5):
            correct_predictions_spam += 1
        file.close()
    spam_accuracy = (correct_predictions_spam/total_test_spam)*100


    path_test = path_ham
    files = listdir(path_test)
    total_test_ham = len(files)
    correct_predictions_ham = 0
    for f in files:
        pred_ham = 0
        pred_spam = 0
        attribute_map = {}
        file = open(path_test+'/'+f, 'r', encoding='utf8', errors='ignore')
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            words_in_line = line.split()
            for word in words_in_line:
                if(word in training_attr):
                    if(word not in attribute_map):
                        attribute_map[word] = 1
                    else:
                        attribute_map[word] += 1
        x = {}
        for attr in training_attr:
            if(attr in attribute_map):
                x[attr] = attribute_map[attr]
            else:
                x[attr] = 0
        total = w[0]
        for att in x:
            total += (float(x[att])*w[att])
        if((1/(1+exp(-total))) >= 0.5):
            correct_predictions_ham += 1
        file.close()
    ham_accuracy = (correct_predictions_ham/total_test_ham)*100

    total_accuracy = (correct_predictions_spam + correct_predictions_ham)/(total_test_ham + total_test_spam)
    return ham_accuracy, spam_accuracy, total_accuracy

def filter_stop_words(my_map):
    f = open('stopwords.txt', 'r')
    stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = str(stopwords[i].replace('\n',''))
    filtered_map = my_map
    for word in stopwords:
        if(word in filtered_map):
            filtered_map.pop(word)
    
    return filtered_map, stopwords
 

path_ham = 'hw2_train/train/ham'
path_spam = 'hw2_train/train/spam'
path_ham_test = 'hw2_test/test/ham'
path_spam_test = 'hw2_test/test/spam'
ham_map, spam_map,total_words_in_ham, total_words_in_spam, distinct_words, total_training_examples = preprocess(path_ham, path_spam)
ham_accuracy, spam_accuracy, total_accuracy = test_naive_bayes(path_ham_test, path_spam_test, total_words_in_ham, total_words_in_spam, distinct_words)


print('-' * 100)
print('Accuracies of Naive Bayes and Logistic Regression without Filtering')
print('-' * 100)
print()
print('Naive Bayes Accuracy')
print('-' * 100)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)
print()


print('Logistic Regression')
data_logreg, training_attr = generate_data_logreg(copy.deepcopy(ham_map), copy.deepcopy(spam_map), path_ham, path_spam)

w = {}
for t in training_attr:
    w[t] = 0
w[0] = 0

steps = 10
'''
learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)
'''
learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)
'''
learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
'''
print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
'''
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)


steps = 10
learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w10_1), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w10_2), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w10_3), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w10_4), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w10_5), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)


steps = 10
learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w10_1), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w10_2), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w10_3), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w10_4), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w10_5), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)

steps = 10
learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w10_1), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w10_2), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w10_3), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w10_4), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w10_5), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)

steps = 10

learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w10_1), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w10_2), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w10_3), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w10_4), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w10_5), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)
'''

ham_map, stopwords = filter_stop_words(copy.deepcopy(ham_map))
spam_map, stopwords = filter_stop_words(copy.deepcopy(spam_map))

total_words_in_ham = 0
total_words_in_spam = 0
distinct_words = 0

for word in ham_map:
    total_words_in_ham += ham_map[word]
    distinct_words += 1
 
for word in spam_map:
    total_words_in_spam += spam_map[word]
    if(word not in ham_map):
        distinct_words += 1


ham_accuracy, spam_accuracy, total_accuracy = test_naive_bayes_postfiltering(path_ham_test, path_spam_test, total_words_in_ham, total_words_in_spam, distinct_words, stopwords)


print('-' * 100)
print('Accuracies of Naive Bayes and Logistic Regression after Filtering')
print('-' * 100)
print()
print()
print('Naive Bayes Accuracy')
print('-' * 100)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)
print()

print('Logistic Regression Accuracy')
data_logreg, training_attr = generate_data_logreg(copy.deepcopy(ham_map), copy.deepcopy(spam_map), path_ham, path_spam)

w = {}
for t in training_attr:
    w[t] = 0
w[0] = 0

steps = 10
'''
learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)
'''
learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)
'''
learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
'''
print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
'''
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)


steps = 10
learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w10_1), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w10_2), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w10_3), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w10_4), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w10_5), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)


steps = 10
learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w10_1), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w10_2), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w10_3), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w10_4), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w10_5), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)

steps = 10
learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w10_1), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w10_2), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w10_3), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w10_4), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w10_5), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)

steps = 10
learning_rate = 0.003
regularization = 0
w10_1 = train_log_reg(data_logreg, copy.deepcopy(w10_1), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.02
w10_2 = train_log_reg(data_logreg, copy.deepcopy(w10_2), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.02
w10_3 = train_log_reg(data_logreg, copy.deepcopy(w10_3), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.03
regularization = 0.002
w10_4 = train_log_reg(data_logreg, copy.deepcopy(w10_4), steps, learning_rate, regularization, total_training_examples)

learning_rate = 0.003
regularization = 0.002
w10_5 = train_log_reg(data_logreg, copy.deepcopy(w10_5), steps, learning_rate, regularization, total_training_examples)

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_1, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_2, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_3, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_4, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))

print('-' * 100)
ham_accuracy, spam_accuracy, total_accuracy = test_log_reg(w10_5, path_ham_test, path_spam_test)
print('ham accuracy:- '+str(ham_accuracy), ', spam accuracy:- '+str(spam_accuracy), ', total accuracy:- '+str(total_accuracy))
print('-' * 100)'''