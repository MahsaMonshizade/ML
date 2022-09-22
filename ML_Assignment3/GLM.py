from numpy import genfromtxt, inf
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import time

# This function gets the file_name as an input and return  numpy array containing th file content
def read_file(file_name):
    my_data = genfromtxt('pp3data\\' + file_name, delimiter=',')
    return my_data


# This functions gets the train set and test set and their labels in the list and also the name of the likelihood function(we have 3 function the first one return d and r, the second oen return the predited label and the last one retun the error) and it calculates w MAP and returns the error.
def GLM(train, trainR, test, testR, myder, myder_t, myder_error, alpha):
    
    m = len(train[0])
    w = [0 for i in range(m)]  # initialize the w=0

    number_of_iteration = 100
    start = time.time()
    for i in range(100):
        d, r = myder(train, trainR, w)
        g = np.dot(train.T, d) - np.dot(alpha, w)
        h = (-1)*np.dot(np.dot(train.T, r), train) - np.dot(alpha, np.identity(m))
        new_w = w - np.dot(np.linalg.inv(h), g)
        if (np.linalg.norm(w)) != 0 and (np.linalg.norm(new_w - w))/(np.linalg.norm(w)) < 1/1000:
            w = new_w
            number_of_iteration = i
            break
        w = new_w
    finish = time.time()
    runtime = finish - start # runtime for calculating the w MAP 
    predicted_t = myder_t(test, w)
    error= myder_error(testR, predicted_t)
    return error ,number_of_iteration, runtime


# This function gets the train sts and labels and the w and return d and r for logistic likelihood
def logistic(phi, t, w):
    n = len(phi)
    r = [[0 for i in range(n)] for j in range(n)]
    a = np.dot(w, phi.T)
    y =list(map(lambda x: 1/(1 + np.exp((-1)*x)), a))
    d = t - y
    for i in range(n):
        r[i][i] = y[i]*(1-y[i])
    return d, r


# This function gets the test set and w MAP and returns list of predicted labels for logistic regression
def logistic_predict_t(phi, w):
    n = len(phi)
    predicted_t = [0 for i in range(n)]
    for i in range(n):
        a = np.dot(w, phi[i].T)
        if 1/(1 + np.exp((-1)*a)) >= 0.5:
            predicted_t[i] = 1
    return predicted_t


# This function get the True labels and precited labels list and calculate the error for our logistic regression model
def logistic_error(t, predicted_t):
    error = 0
    for i in range(len(t)):
        if t[i] != predicted_t[i]:
            error += 1
    return error/len(t)


# This function gets the train sets and labels and the w and return d and r for poisson likelihood
def poisson(phi, t, w):
    n = len(phi)
    a = [0 for i in range(n)]
    y = [0 for i in range(n)]
    d = [0 for i in range(n)]
    r = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        a[i] = np.dot(w, phi[i].T)
        y[i] = np.exp(a[i])
        d[i] = t[i] - y[i]
        r[i][i] = y[i]
    return d, r


# This function gets the test set and w MAP and returns list of predicted labels for count regression
def poisson_predict_t(phi, w):
    n = len(phi)
    predicted_t = [0 for i in range(n)]
    for i in range(n):
        a = np.dot(w, phi[i].T)
        lambdaa = np.exp(a)
        predicted_t[i] = math.floor(lambdaa)
    return predicted_t


# This function get the True labels and precited labels list and calculate the error for our count regression model
def poisson_error(t, predicted_t):
    error = 0
    for i in range(len(t)):
        error += abs(t[i] - predicted_t[i])
    return error/len(t)
    

# This function gets the train sts and labels and the w and return d and r for poisson likelihood
def ordinal(phi, t, w):
    k = 5
    s = 1
    ph = [(-1)*inf, -2, -1, 0, 1, inf]
    n = len(phi)
    a = [0 for i in range(n)]
    d = [0 for i in range(n)]
    r = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        a[i] = np.dot(w, phi[i].T)
        yit = 1/(1 + np.exp((-1)*s*(ph[int(t[i])]-a[i])))
        yit_1 = 1/(1 + np.exp((-1)*s*(ph[int(t[i-1])]-a[i])))
        d[i] = yit +yit_1 - 1
        r[i][i] = (s**2)*(yit*(1-yit)+yit_1*(1-yit_1))
    return d, r


# This function gets the test set and w MAP and returns list of predicted labels for ordinal regression
def ordinal_predict_t(phi, w):
    n = len(phi)
    s = 1
    ph = [(-1)*inf, -2, -1, 0, 1, inf]
    predicted_t = [0 for i in range(n)]
    for i in range(n):
        a = np.dot(w, phi[i].T)
        y = [0 for k in range(6)]
        p = [0 for k in range(6)]
        for j in range(6):
            y[j] = 1/(1 + np.exp((-1)*(s*(ph[j]-a))))
            if j != 0:
                p[j] = y[j] - y[j-1]
        max_value = max(p[1:])
        predicted_t[i] = p[1:].index(max_value)
    return predicted_t


# This function gets the True labels and precited labels list and calculate the error for our ordinal regression model
def ordinal_error(t, predicted_t):
    error = 0
    for i in range(len(t)):
        error += abs(t[i] - predicted_t[i])
    return error/len(t)


# This function Calculates the errors for each size 30 times and return 2d list for error(we have 30 sublist that each list has 10 error number for each portion). In addition record the number of iterations and the run time untill convergence in each run and return them.
def task(phi, t, myder, myder_t, myder_error):
    n = len(phi)
    errors = []
    numberofIterations = []
    runtimes = []
    for i in range(30):
        p = np.random.permutation(len(phi))
        phi, t = phi[p], t[p]
        train = phi[:(2*n)//3]
        trainR = t[:(2*n)//3]
        test = phi[(2*n)//3:]
        testR = t[(2*n)//3:]
        size = len(train)
        error_rate = []
        n_of_i = []
        rtime = []
        for j in range(1, 11):
            error, number_of_iteration, runtime = GLM(train[:(j * size)//10], trainR[:(j * size)//10], test, testR,  myder, myder_t, myder_error, 10)
            error_rate.append(error)
            n_of_i.append(number_of_iteration)
            rtime.append(runtime)
        errors.append(error_rate)
        numberofIterations.append(n_of_i)
        runtimes.append(rtime)
    return errors, numberofIterations, runtimes


# This function get a 2d list and reutrn a list of 10 elements that are the mean errors for each portion and also another list for SD for each portion
def mean_and_SD(errors):
    mean_list = []
    SD_list = []
    for i in range(10):
        mean_sum = 0
        for j in range(30):
            mean_sum += errors[j][i]
        mean_list.append(mean_sum/30)
        sd_sum = 0
        for j in range(30):
            sd_sum += (errors[j][i] - mean_list[i])**2
        SD_list.append(math.sqrt(sd_sum/29))
    return mean_list, SD_list


# This function get the errors list and plot the lurining curve
def plot(mean_list, SD_list, name):
    fig = plt.figure() 
    x = [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    plt.errorbar(x, mean_list, yerr = SD_list) 
    plt.xlabel("training set portion")
    plt.ylabel("error")
    plt.title(name[:len(name)-4] + " dataset")
    plt.savefig("results\\" + name)
    plt.show()


# This function belongs to Extra credit part. It use the cross validation method as model selection for alpha.
def optimize_alpha(phi, t, myder, myder_t, myder_error):
    p = np.random.permutation(len(phi))
    phi, t = phi[p], t[p]
    train = phi[:(2*n)//3]
    trainR = t[:(2*n)//3]
    test = phi[(2*n)//3:]
    testR = t[(2*n)//3:]

    length = len(train)
    alpha_error = []
    for alpha in range(1, 11):
        errors = []
        for i in range(10):
            if i == 0:
                cross_train = train[(i+1)*length//10:]
                cross_trainR = trainR[(i+1)*length//10:]
            elif i == 9:
                cross_train = train[:i*length//10]
                cross_trainR = trainR[:i*length//10]
            else:
                cross_train = np.concatenate([train[:i*length//10],train[(i+1)*length//10:]])
                cross_trainR = np.concatenate([trainR[:i*length//10],trainR[(i+1)*length//10:]])
            cross_test = train[i*length//10:(i+1)*length//10]
            cross_testR = trainR[i*length//10:(i+1)*length//10]
            error = GLM(cross_train, cross_trainR, cross_test, cross_testR, myder, myder_t, myder_error, alpha)
            errors.append(error)

        avg_error = 0
        for e in error:
            avg_error += e
        alpha_error.append(avg_error/10)
    best_alpha = 0
    best_error = inf
    for i in range(len(alpha_error)):
        if alpha_error[i]<best_error:
            best_alpha = i+1
            best_error = alpha_error[i]
    return alpha_error, best_alpha, best_error
        

datasets = ["usps", "A", "AP", "AO"]
for dataset in datasets:
    data = read_file(dataset+".csv") #read the dataset from the file and return data will be a numpy array
    n = len(data)
    new_column = [1 for i in range(n)] 
    data = np.insert(data, 0, new_column, axis=1) #add a feature fixed at one to a first column
    dataR = read_file("labels-"+dataset+".csv")
    if dataset == "A" or dataset == "usps":
        errors, numberofiterations, runtimes = task(data, dataR, logistic, logistic_predict_t, logistic_error)
    elif dataset == "AP":
        errors, numberofiterations, runtimes = task(data, dataR, poisson, poisson_predict_t, poisson_error)
    else:
        errors, numberofiterations, runtimes = task(data, dataR, ordinal, ordinal_predict_t, ordinal_error)
    errors_meanL, errors_sdL = mean_and_SD(errors)
    numberofiterations_meanL, numberofiterations_sdL = mean_and_SD(numberofiterations)
    runtimes_meanL, runtimes_sdL = mean_and_SD(runtimes)
    print(dataset)
    for j in range(1, 11):
        print("dataset portion" + str(j))
        print("number of iterations average: " + str(numberofiterations_meanL[j-1]))
        print("runtime average: " + str(runtimes_meanL[j-1]))
        print(50*"-")
    print(50*"~")

    plot(errors_meanL, errors_sdL, dataset+".png")


### Extra Credit
for dataset in datasets:
    data = read_file(dataset+".csv")
    n = len(data)
    new_column = [1 for i in range(n)]
    data = np.insert(data, 0, new_column, axis=1)
    dataR = read_file("labels-"+dataset+".csv")
    if dataset == "A" or dataset == "usps":
        alpha_error, best_alpha, best_error = optimize_alpha(data, dataR, logistic, logistic_predict_t, logistic_error)
    elif dataset == "AP":
        alpha_error, best_alpha, best_error = optimize_alpha(data, dataR, poisson, poisson_predict_t, poisson_error)
    else:
        alpha_error, best_alpha, best_error = optimize_alpha(data, dataR, ordinal, ordinal_predict_t, ordinal_error)
    
    print(dataset)
    print("error for alpha from 1 to 10: " + str(alpha_error) )
    print("best error is: " + str(best_error))
    print("best alpha is: " + str(best_alpha))
    print(50*"~")
    