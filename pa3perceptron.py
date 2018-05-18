import numpy as np
import random

# Load the data from the file and return a list of lists, with each line as a list
def load_data(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    dataList = []
    
    for line in lines:
        lineList = line.split()
        fLineList = [int(i) for i in lineList]
        dataList.append(fLineList)
    return dataList

# Returns a new list of vectors that contains only the vectors that have the accepted labels
def prune_data(data_list, accepted_labels):
    new_list = [] 
    for vector in data_list:
        if vector[-1] in accepted_labels:
            new_list.append(vector)
    return new_list


def perceptron(data_list, pos_label, neg_label, num_passes):
    w = np.zeros(len(data_list[0]) - 1, dtype=int)
    for i in range(num_passes):
        for vector in data_list:
            x = np.array(vector[:-1])
            y = 1 if vector[-1] == pos_label else -1
            # print ("Y is: ", y, "\nX is: ", x, "\nW dot X is ", np.dot(w,x))
            if y * (np.dot(w, x)) <= 0:
                w = w + y * x
    return w

def voted_perceptron(data_list, pos_label, neg_label, num_passes):
    last_vector_was_append = False
    w = np.zeros(len(data_list[0]) - 1, dtype=int)
    w_list = []
    c = 1
    for i in range(num_passes):
        for vector in data_list:
            x = np.array(vector[:-1])
            y = 1 if vector[-1] == pos_label else -1
            if y * (np.dot(w, x)) <= 0:
                w_list.append((w,c))
                w = w + y * x
                c = 1
                last_vector_was_append = True
            else:
                c += 1
                last_vector_was_append = False
    
    # Append one last time for the last w,c pair if we didn't just append
    if not last_vector_was_append:
        w_list.append((w,c))
    
    return w_list

def averaged_perceptron(data_list, pos_label, neg_label, num_passes):
    last_vector_was_append = False
    w = np.zeros(len(data_list[0]) - 1, dtype=int)
    running_sum = np.zeros(len(data_list[0]) - 1, dtype=int)
    c = 1
    for i in range(num_passes):
        for vector in data_list:
            x = np.array(vector[:-1])
            y = 1 if vector[-1] == pos_label else -1
            if y * (np.dot(w, x)) <= 0:
                running_sum = running_sum + c * w
                w = w + y * x
                c = 1
                last_vector_was_append = True
            else:
                c += 1
                last_vector_was_append = False
    
    # Add running sum one last time for the last w,c pair if we didn't just append
    if not last_vector_was_append:
        running_sum = running_sum + (c * w)
    return running_sum

def classify(classifier, vector, pos_label, neg_label):
    if (np.dot(classifier, vector)) > 0:
        return pos_label
    elif (np.dot(classifier, vector)) < 0:
        return neg_label
    else:
        random.choice([pos_label, neg_label])

def voted_classify(classifier_list, vector, pos_label, neg_label):
    sum = 0
    for w,c in classifier_list:
        sign = np.sign(np.dot(w,vector))
        sum += c * sign

    if (sum) > 0:
        return pos_label
    elif (sum) < 0:
        return neg_label
    else:
        random.choice([pos_label, neg_label])

def one_v_all_classify(classifier_list, vector, pos_label):
    pass

def calc_error(data_list, classifier, pos_label, neg_label):
    error_count = 0
    for vector in data_list:
        if classify(classifier, vector[:-1], pos_label, neg_label) != vector[-1]:
            error_count += 1
    return (error_count / len(data_list))

def voted_calc_error(data_list, classifier_list, pos_label, neg_label):
    error_count = 0
    for vector in data_list:
        if voted_classify(classifier_list, vector[:-1], pos_label, neg_label) != vector[-1]:
            error_count += 1
    return (error_count / len(data_list))

training_data = load_data("pa3train.txt")
test_data = load_data("pa3test.txt")
q1_train_data = prune_data(training_data, [1,2])
q1_test_data = prune_data(test_data, [1,2])

# Question 1 ---------------
# for num_passes in range(1, 5):
#     w = perceptron(q1_train_data, 1, 2, num_passes)
#     training_error = calc_error(q1_train_data, w, 1, 2)
#     test_error = calc_error(q1_test_data, w, 1, 2)
#     print("Training error for Regular on ", num_passes, " passes is: ", training_error)
#     print("Test error for Regular on ", num_passes, " passes is: ", test_error, "\n")

#     w_list = voted_perceptron(q1_train_data, 1, 2, num_passes)
#     voted_training_error = voted_calc_error(q1_train_data, w_list, 1, 2)
#     voted_test_error = voted_calc_error(q1_test_data, w_list, 1, 2)
#     print("Training error for Voted on ", num_passes, " passes is: ", voted_training_error)
#     print("Test error for Voted on ", num_passes, " passes is: ", voted_test_error, "\n")

#     w = averaged_perceptron(q1_train_data, 1, 2, num_passes)
#     averaged_training_error = calc_error(q1_train_data, w, 1, 2)
#     averaged_test_error = calc_error(q1_test_data, w, 1, 2)
#     print("Training error for Averaged on ", num_passes, " passes is: ", averaged_training_error)
#     print("Test error for Averaged on ", num_passes, " passes is: ", averaged_test_error, "\n")

# Question 2 ---------------
# w = averaged_perceptron(q1_train_data, 1, 2, num_passes)
# word_list = []
# for i in range(len(w)):
#     word_list.append((w[i],i+1))

# sorted_words = sorted(word_list, key=lambda x: x[0])
# print(sorted_words)

# Question 3 ----------------
C1 = perceptron(training_data, 1, -1, 1)
C2 = perceptron(training_data, 2, -1, 1)
C3 = perceptron(training_data, 3, -1, 1)
C4 = perceptron(training_data, 4, -1, 1)
C5 = perceptron(training_data, 5, -1, 1)
C6 = perceptron(training_data, 6, -1, 1)

classifier_list = [C1, C2, C3, C4, C5, C6]