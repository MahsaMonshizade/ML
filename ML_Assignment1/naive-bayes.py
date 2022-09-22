import csv
import math
#just use it for random permutation for cross validation
import random
import matplotlib.pyplot as plt

# This function gets the name of txt file and return a 2d list (each sub list in our list is a row of txt file with length 2 that first element is a sentence and a second element is lable for that sentence)
def read_file(file_name):
    arr_data = []
    with open(r'pp1data\\' + file_name, newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in filereader:
            arr_data.append(row)
    return arr_data


# This function removes punctuation from datasets
def pre_process_data(text_data):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    newData = []
    for row in text_data:
        new_string = ""
        for c in row[0]:
            if c not in punctuations:
                new_string = new_string + c
        newData.append([new_string, row[1]])
    return newData


# This function gets train data and return the number of positive sentences in the training data
def count_of_positive_lables(train_data):
    positive_count = 0
    for row in train_data:
        if(row[1] == "1"):
            positive_count += 1
    return positive_count


# This function gets train data and return the number of negative sentences in the training data
def count_of_negative_lables(train_data):
    negative_count = 0
    for row in train_data:
        if(row[1] == "0"):
            negative_count += 1
    return negative_count


# This function gets train data and return a dictionary that shows how many times each words happens in positive sentences 
def count_of_words_for_positive(train_data):
    words_count = {}
    for row in train_data:
        if row[1] == "1":
            sentence = row[0].split(" ")
            for word in sentence:
                if word in words_count:
                    words_count[word]+=1
                else:
                    words_count[word] = 1
    return words_count


# This function gets train data and return a dictionary that shows how many times each words happens in negative sentences 
def count_of_words_for_negative(train_data):
    words_count = {}
    for row in train_data:
        if row[1] == "0":
            sentence = row[0].split(" ")
            for word in sentence:
                if word in words_count:
                    words_count[word]+=1
                else:
                    words_count[word] = 1
    return words_count

# This function gets train data and return number of different words in train data
def number_of_different_words_in_total(train_data):
    count = 0
    words_arr = []
    for r in train_data:
        sentence = r[0].split(" ")
        for word in sentence:
            if word not in words_arr:
                words_arr.append(word)
                count += 1
    return count


# This function gets train data, test_data and smoothing parameter m and return the 2d prediction list that contain sentences and their predicted lable using MAP (if m = 0 therefore it is maxL)
def MAP(train_data, test_data, m):
    prediction = []
    total = number_of_different_words_in_total(train_data)
    positive_labels = count_of_positive_lables(train_data)
    negative_labels = count_of_negative_lables(train_data)
    positive_words = count_of_words_for_positive(train_data)
    negative_words = count_of_words_for_negative(train_data)
    sum_of_positive_words = 0
    for word in positive_words:
        sum_of_positive_words += positive_words[word]
    sum_of_negative_words = 0
    for word in negative_words:
        sum_of_negative_words += negative_words[word]
    
    for row in test_data:
        p_positive = math.log(positive_labels/(positive_labels+negative_labels))
        p_negative = math.log(negative_labels/(positive_labels+negative_labels))
        words = row[0].split(" ")
        for word in words:
            if m == 0 and (word not in positive_words) and (word not in negative_words):
                continue
            elif m == 0 and (word not in positive_words):
                p_positive = 0
                p_negative = 1
                break
            elif m == 0 and (word not in negative_words):
                p_positive = 1
                p_negative = 0
                break
            if word not in positive_words:
                p_positive += math.log(m/(sum_of_positive_words+(m*total)))
            if word in positive_words:
                p_positive += math.log((m+positive_words[word])/(sum_of_positive_words + (m*total)))
            if word not in negative_words:
                p_negative += math.log(m/(sum_of_negative_words + (m*total)))
            if word in negative_words:
                p_negative += math.log((m+negative_words[word])/(sum_of_negative_words + (m*total)))
        predict = "N"
        if p_positive>p_negative:
            predict = "1"
        if p_positive<p_negative:
            predict = "0"
        prediction.append([row[0], predict])
    return prediction
        

# This function gets all datas and return a dataset contains positive sentences and their lables
def positive_sentences(text_data):
    positive_sentences = []
    for r in text_data:
        if r[1] == "1":
            positive_sentences.append(r)
    return positive_sentences


# This function gets all datas and return a dataset list contains negative sentences and their lables
def negative_sentences(text_data):
    negative_sentences = []
    for r in text_data:
        if r[1] == "0":
            negative_sentences.append(r)
    return negative_sentences


# This function gets a list od data set and k (number of folds we want) and return a lsit of k folds that has same proportions for positive and negative sentences and also shuffled
def cross_validation(text_data, k):
    positiveSentences = positive_sentences(text_data)
    negativeSentences = negative_sentences(text_data)
    random.shuffle(positiveSentences)
    random.shuffle(negativeSentences)
    positive_folds = []
    negative_folds = []
    k_folds = []
    for i in range(k):
        positive_folds.append(positiveSentences[(i*(len(positiveSentences)))//k:((i+1)*(len(positiveSentences)))//k])
        negative_folds.append(negativeSentences[(i*(len(negativeSentences)))//k:((i+1)*(len(negativeSentences)))//k])
        k_folds.append(positive_folds[i]+negative_folds[i])
    return k_folds


# This function gets a list contains k different data_sets with same lenghth (output of cross_validation) and return 2 lists
# k_train_sets contains k training datases and k_test_sets contains k testing datasets
def k_train_test(k_folds):
    k_train_sets = []
    k_test_sets = []
    for k in range(len(k_folds)):
        test_set = k_folds[k]
        train_set = []
        for i in range(len(k_folds)):
            if i != k:
                train_set += k_folds[i]
        random.shuffle(train_set)
        k_train_sets.append(train_set)
        k_test_sets.append(test_set)
    return k_train_sets, k_test_sets


# This function gets a list of lables prediction and actual labels with their sentences and return accuracy of prediction
def accuracy_func(test_set, test_prediction):
    sum = 0
    for i in range(len(test_set)):
        if test_set[i][1] == test_prediction[i][1]:
            sum += 1
    accuracy = (sum/len(test_set))*100
    return accuracy


# This function gets a list of numbers and return the average of them
def k_average(data):
    sum = 0
    for num in data:
        sum += num
    return (sum/len(data))


# This function gets a list of numbers and return the sd for them
def k_standard_deviation(data):
    sum = 0
    average = k_average(data)
    for num in data:
        sum = sum + ((num-average)**2)
        sd = math.sqrt(sum/(len(data)-1))
    return sd


# This function gets k train sets in one list, k test sets in one list and m (smoothing parameter). Return average list of accuracy (for each 0.1N ... N) and sd list of accuracy (for each 0.1N ... N)
def subsamples (k_train_sets, k_test_sets, m):
    average = []
    standard_deviation = []
    for i in range(1, 11):
        accuracy_list = []
        for k in range(len(k_train_sets)):
            train_size = i*len(k_train_sets[k])//10
            test_prediction = []
            test_prediction = MAP(k_train_sets[k][0:train_size], k_test_sets[k], m)
            accuracy = accuracy_func(k_test_sets[k], test_prediction)
            accuracy_list.append(accuracy)
        average.append(k_average(accuracy_list))
        standard_deviation.append(k_standard_deviation(accuracy_list))
    
    return average, standard_deviation


# This function gets k train sets in one list, k test sets in one list and lis m (list of smoothing parameters). Return average list of accuracy (for each smothing parameter) and sd list of accuracy (for each smoothing parameter)
def samples (k_train_sets, k_test_sets, m):
    average = []
    standard_deviation = []
    for i in m:
        accuracy_list = []
        for k in range(len(k_train_sets)):
            test_prediction = []
            test_prediction = MAP(k_train_sets[k], k_test_sets[k], i)
            accuracy = accuracy_func(k_test_sets[k], test_prediction)
            accuracy_list.append(accuracy)
        average.append(k_average(accuracy_list))
        standard_deviation.append(k_standard_deviation(accuracy_list))
    
    return average, standard_deviation


def plot_part1(avg1, avg2, sd1, sd2, file_name, y_label):

    fig = plt.figure() 
    x = [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    plt.errorbar(x, avg1, yerr = sd1, label ='m = 0') 
    plt.errorbar(x, avg2, yerr = sd2, label ='m = 1') 
    
    plt.legend() 
    # naming the x axis
    plt.xlabel('train_size (portion of N)')
    # naming the y axis
    plt.ylabel(y_label)
    plt.title(file_name[:len(file_name)-4]) 
    plt.savefig(file_name)
    plt.show()


def plot_part2(m, avg, sd, file_name):
    fig = plt.figure() 
    
    plt.errorbar(m, avg, yerr = sd ) 
    
    plt.legend() 
    # naming the x axis
    plt.xlabel('smoothing parameter m')
    # naming the y axis
    plt.ylabel("accuracy")
    plt.title(file_name[:len(file_name)-4]) 
    plt.savefig(file_name)
    plt.show()
    

# Part 1

m0 = 0
m1 = 1
k = 10

file1 = "imdb_labelled.txt"
file2 = "amazon_cells_labelled.txt"
file3 = "yelp_labelled.txt"

text_data1 = read_file(file1)
text_data2 = read_file(file2)
text_data3 = read_file(file3)

k_fold1 = cross_validation(text_data1, k)
k_fold2 = cross_validation(text_data2, k)
k_fold3 = cross_validation(text_data3, k)

k_train_sets1, k_test_sets1 = k_train_test(k_fold1)
k_train_sets2, k_test_sets2 = k_train_test(k_fold2)
k_train_sets3, k_test_sets3 = k_train_test(k_fold3)

average10 , sd10 = subsamples(k_train_sets1, k_test_sets1, m0)
average20 , sd20 = subsamples(k_train_sets2, k_test_sets2, m0)
average30 , sd30 = subsamples(k_train_sets3, k_test_sets3, m0)

average11, sd11 = subsamples(k_train_sets1, k_test_sets1, m1)
average21, sd21 = subsamples(k_train_sets2, k_test_sets2, m1)
average31, sd31 = subsamples(k_train_sets3, k_test_sets3, m1)

plot_part1(average10, average11, sd10, sd11, "imdb_average_m0_1.png", "average of accuracy")
plot_part1(average20, average21, sd10, sd11, "amazon_average_m0_1.png", "average of accuracy")
plot_part1(average30, average31, sd10, sd11, "yelp_average_m0_1.png", "average of accuracy")


# Part 2 sample_size: N
m = []
for i in range(0, 10):
    m.append(i/10)
for i in range(1, 11):
    m.append(i)

average1m , sd1m = samples(k_train_sets1, k_test_sets1, m)
average2m , sd2m = samples(k_train_sets2, k_test_sets2, m)
average3m , sd3m = samples(k_train_sets3, k_test_sets3, m)

plot_part2(m, average1m, sd1m, "imdb_average_m.png")
plot_part2(m, average2m, sd2m, "amazon_average_m.png")
plot_part2(m, average3m, sd3m, "yelp_average_m.png")

### part 3 (optinal) do all part 1 and 2 with preprocesses data
# part 3.1
m0 = 0
m1 = 1
k = 10

file1 = "imdb_labelled.txt"
file2 = "amazon_cells_labelled.txt"
file3 = "yelp_labelled.txt"

text_data1 = read_file(file1)
text_data2 = read_file(file2)
text_data3 = read_file(file3)

text_data1 = pre_process_data(text_data1)
text_data2 = pre_process_data(text_data2)
text_data3 = pre_process_data(text_data3)

k_fold1 = cross_validation(text_data1, k)
k_fold2 = cross_validation(text_data2, k)
k_fold3 = cross_validation(text_data3, k)

k_train_sets1, k_test_sets1 = k_train_test(k_fold1)
k_train_sets2, k_test_sets2 = k_train_test(k_fold2)
k_train_sets3, k_test_sets3 = k_train_test(k_fold3)

average10 , sd10 = subsamples(k_train_sets1, k_test_sets1, m0)
average20 , sd20 = subsamples(k_train_sets2, k_test_sets2, m0)
average30 , sd30 = subsamples(k_train_sets3, k_test_sets3, m0)

average11, sd11 = subsamples(k_train_sets1, k_test_sets1, m1)
average21, sd21 = subsamples(k_train_sets2, k_test_sets2, m1)
average31, sd31 = subsamples(k_train_sets3, k_test_sets3, m1)

plot_part1(average10, average11, sd10, sd11, "new_imdb_average_m0_1.png", "average of accuracy")
plot_part1(average20, average21, sd10, sd11, "new_amazon_average_m0_1.png", "average of accuracy")
plot_part1(average30, average31, sd10, sd11, "new_yelp_average_m0_1.png", "average of accuracy")


# Part 3.2 sample_size: N
m = []
for i in range(0, 10):
    m.append(i/10)
for i in range(1, 11):
    m.append(i)

average1m , sd1m = samples(k_train_sets1, k_test_sets1, m)
average2m , sd2m = samples(k_train_sets2, k_test_sets2, m)
average3m , sd3m = samples(k_train_sets3, k_test_sets3, m)

plot_part2(m, average1m, sd1m, "new_imdb_average_m.png")
plot_part2(m, average2m, sd2m, "new_amazon_average_m.png")
plot_part2(m, average3m, sd3m, "new_yelp_average_m.png")




