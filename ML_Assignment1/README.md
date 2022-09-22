# Assignment 1
When you run the code, all required results in experiments will appear. 

In the following, we have each function description

\*  when it says that a function gets data as an input parameter, the data is has a list structure.

### read_file function: 
This function gets the name of txt file and return a 2d list (each sub list in our list is a row of txt file with length 2 that first element is a sentence and a second element is lable for that sentence).




```python
def read_file(file_name):
    ...
    return text_data
```
example:

file_name = "imdb_labelled.txt"

text_data = [["Not sure who was more lost - the flat characters or the audience, nearly half of whom walked out.", "0"]["...","..." ]...]


### pre_process_data function
This function removes punctuations from datasets.

```python
def pre_process_data(text_data):
    ...
    return newData
```



### count_of_positive_lables function:
This function gets train data and return the number of positive sentences in the training data
```python
def count_of_positive_lables(train_data):
    ...
    return positive_count
```

### count_of_negative_lables function
This function gets train data and return the number of negative sentences in the training data

```python
def count_of_negative_lables(train_data):
    ...
    return negative_count
```

### count_of_words_for_positive function
This function gets train data and return a dictionary that shows how many times each word appears in positive sentences 

```python
def count_of_words_for_positive(train_data):
    ... 
    return words_count
```

### count_of_words_for_negative function
This function gets train data and return a dictionary that shows how many times each word appear in negative sentences 

```python
def count_of_words_for_negative(train_data):
    ... 
    return words_count
```

### number_of_different_words_in_total function
This function gets train data and return number of different words in train data. (number od features)

```python
def number_of_different_words_in_total(train_data):
    ...
    return count
```

### MAP function
This function gets train data, test_data and smoothing parameter m and return the 2d prediction list that contain sentences and their predicted lable using MAP (if m = 0, therefore it is maxL)

```python
def MAP(train_data, test_data, m):
    ...
    return prediction
```

predition has the structure like following:

prediction = [["Not sure who was more lost - the flat characters or the audience, nearly half of whom walked out.", "0"]["...","1" ]...]


### positive_sentences function
This function gets all datas as a 2d list and return a dataset 2d list contains positive sentences and their lables.

```python
def positive_sentences(text_data):
    ...
    return positive_sentences
```

### negative_sentences function
This function gets all datas as a 2d list and return a dataset 2d list contains negative sentences and their lables.

```python
def negative_sentences(text_data):
    ...
    return negative_sentences
```

### cross_validation function
This function gets a list of data set and k (number of folds we want) and return a list of k folds that has same proportions for positive and negative sentences and also shuffled.

```python
def cross_validation(text_data, k):
    ...
    return k_folds
```

### k_train_test function
This function gets a list contains k different data_sets with same lenghth (output of cross_validation) and return 2 lists. k_train_sets contains k training datases and k_test_sets contains k testing datasets.

```python
def k_train_test(k_folds):
    ...
    return return k_train_sets, k_test_sets
```

### accuracy_func function
This function gets a list of lables prediction and actual labels with their sentences and return accuracy of prediction

```python
def accuracy_func(test_set, test_prediction):
    ...
    return accuracy
```

### k_average function
This function gets a list of numbers and return the average of them.

```python
def k_average(data):
    ...
    return (sum/len(data))
```

### k_standard_deviation function
This function gets a list of numbers and return the standard devation for them.

```python
def k_standard_deviation(data):
    ...
    return sd
```

### subsamples function
This function gets k train sets in one list, k test sets in one list and m (smoothing parameter). Return average list of accuracy (for each 0.1N ... N) and sd list of accuracy (for each 0.1N ... N).

The function used for part 1 of the Experiments.

```python
def subsamples (k_train_sets, k_test_sets, m):
    ...
    return average, standard_deviation
```

### samples function
This function gets k train sets in one list, k test sets in one list and list m (list of smoothing parameters). Return average list of accuracy (for each smothing parameter) and sd list of accuracy (for each smoothing parameter)

```python
def samples (k_train_sets, k_test_sets, m):
    ...
    return average, standard_deviation
```

### main part
in main part at the begining we have m0, m1 and k parameters. you can change their value to any value that you want.

m0 = is a smoothing parameter for part1

m1 = is another smoothing parameter for part1

k = number of folds

file1, file2 and file3 are the name of the files that we want to read and get the data using read_file function.
 first we read each file, then we make them into k_folds and then having a list of k train sets and a list of k test sets foe each of these 3 datasets.

 Then for each of these 3 datasets we calculate the average and standard devation of the accuracy one time with m0 smoothing parameter and the other time with m1 smoothig parameter. 

 at the end we plot theresults for experimetn one.
 files that contains the results for part 1:

 - amazon_average_m0_1.png
 - imdb_average_m0_1.png
 - yelp_average_m0_1.png


part 2:

for this part I use the train and test sets in part 1. This time I have a list m = [0, 0.1, 0.2 ... 0.9, 1, 2, ...10]

I get the average and sd of accuracy for each datasets and plot them as a function of m parameters.

part 3:
In this part, I did the part 1 and part 2 again but this time with pre_processed data.
The results are:

for part3.1:
- new_amazon_average_m0_1.png
- new_imdb_average_m0_1.png
- new_yelp_average_m0_1.png

for part3.2:

- new_amazon_average_m.png
- new_imdb_average_m.png
- new_yelp_average_m.png












