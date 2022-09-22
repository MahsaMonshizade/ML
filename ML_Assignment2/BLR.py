from numpy import genfromtxt, inf
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# This function gets the name of txt file and return a 2d list
def read_file(file_name):
    my_data = genfromtxt('pp2data\\' + file_name, delimiter=',')
    return my_data

# This function gets the train dataset as a 2d list and the resultsof train dataset as a 1d list and get an integer for lmbda, return the w (model parameter)
def regularized_linear_regression(train, trainR, landa):
    m = len(train[0]) # number of features
    phi = train # phi of n*m (I know this line is not necessary I just did it for better understnding for myself :))
    t = trainR 
    w = np.dot(np.dot(np.linalg.inv(np.dot(landa, np.identity(m)) + np.dot(phi.T, phi)), phi.T), t)
    return w
    

# This function gets the train dataset as a 2d list and the resultsof train dataset as a 1d list and return the alpha and beta for task 3
def evidence_approximation(train, trainR):

    m = len(train[0]) # number of features
    n = len(train) # number of samples
    phi = train # phi of n*m
    t = trainR

    alpha = random.uniform(1, 10) # choose alpha randomley between 1 to 10
    beta = random.uniform(1,10) # choose beta randomley between 1 to 10

    while True:
        landas, v = np.linalg.eig(np.dot(np.dot(beta, phi.T), phi)) # eigen values and egen vectors of beta*phi.T*phi
        gamma = 0

        for landa in landas:
            gamma += (landa/(alpha+landa)) 

        SN_inverse = alpha*np.identity(m) + np.dot(np.dot(beta, phi.T), phi)
        mN = np.dot(np.dot(np.dot(beta, np.linalg.inv(SN_inverse)),phi.T), t)

        new_alpha = gamma/np.dot(mN.T, mN)
        new_beta = 0
        for i in range(n):
            new_beta += (t[i]-np.dot(mN.T,phi[i]))**2
        new_beta = new_beta/(n-gamma)
        new_beta = 1/new_beta

        if abs(new_alpha-alpha)<0.0001 and abs(new_beta-beta)<0.0001:
            return new_alpha, new_beta

        alpha = new_alpha
        beta = new_beta

# This function gets the test dataset as a 2d list and model parameter w and the results of test dataset as a 1d list and return the MSE
def MSE(phi, w, t):
    n = len(phi) # number of samples
    sum = 0
    for i in range(n):
        sum += (np.dot(phi[i].T, w)-t[i])**2
    return sum/n


# This function gets the list of train mse's and a list of test mse's for all landas and get landas (a list contains integer between 0 to 150) and the name of the dataset and plot for task1.
def plot_task1(train_mse, test_mse, landas, name):

    plt.plot(landas, train_mse, label='train')
    plt.plot(landas, test_mse, label='test')
    plt.legend() 
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.title(name) 
    plt.savefig('task1\\'+name+'_task1.png')
    plt.show()


# This function gets the traind datasets and its resuts and test datasets and its reasults as a seperate lists and print the best MSE and Best lambda for Task1
def task1(train, trainR, test, testR):
    
    train_mse = [] # its going to save all train mse's for all landas from 0 o 150
    test_mse = [] # its going to save all test mse's for all landas from 0 o 150
    landas = [] # its a list contatins all integers from 0 to 150
    best_mse = inf
    best_landa = 0
    for landa in range(151):
        w = regularized_linear_regression(train, trainR, landa)
        train_mse.append(MSE(train, w, trainR))
        test_mse.append(MSE(test, w, testR))
        if test_mse[landa]<best_mse:
            best_mse = test_mse[landa]
            best_landa = landa
        landas.append(landa)

    print("best MSE: " + str(best_mse))
    print("best lambda: " + str(best_landa))
    print("\n")
    
    plot_task1(train_mse, test_mse, landas, name)


# This function gets the traind datasets and its resuts and test datasets and its reasults as a seperate lists and print the best MSE and Best lambda and runtime for Task1
def task2(train, trainR, test, testR):
    
    start = time.time()

    length = len(train)
    landa_mse = []
    for landa in range(151):
        mse = []
        for i in range(10):
            if i == 0:
                cross_train = train[(i+1)*length//10:]
                cross_trainR = trainR[(i+1)*length//10:]
            elif i == 9:
                cross_train = train[:i*length//10]
                cross_trainR = trainR[:i*length//10]
            else:
                cross_train = train[:i*length//10] + train[(i+1)*length//10]
                cross_trainR = trainR[:i*length//10] + trainR[(i+1)*length//10]
            cross_test = train[i*length//10:(i+1)*length//10]
            cross_testR = trainR[i*length//10:(i+1)*length//10]
            w = regularized_linear_regression(cross_train, cross_trainR, landa)
            mse.append(MSE(cross_test, w, cross_testR))

        avg_mse = 0
        for m in mse:
            avg_mse += m
        landa_mse.append(avg_mse/10)
    best_landa = 0
    best_mse = inf
    for i in range(len(landa_mse)):
        if landa_mse[i]<best_mse:
            best_landa = i
            best_mse = landa_mse[i]
        
    w = regularized_linear_regression(train, trainR, best_landa)
    mse = MSE(test, w, testR)
    end = time.time()

    print("best MSE: " + str(mse))
    print("best lambda: " + str(best_landa))
    print(name + " runtime: " + str(end-start))
    print("\n")
        

# This function gets the traind datasets and its resuts and test datasets and its reasults as a seperate lists and print the best MSE and Best lambda and runtimeand alpha and beta for Task3
def task3(train, trainR, test, testR):
    start = time.time()

    alpha , beta = evidence_approximation(train, trainR)
        
    m = len(train[0]) # number of features
    n = len(train) # number of samples
    phi = train # phi of n*m
    t = trainR
    SN_inverse = np.dot(alpha, np.identity(m)) + np.dot(np.dot(beta, phi.T), phi)
    mN = np.dot(np.dot(np.dot(beta, np.linalg.inv(SN_inverse)),phi.T), t)
    wMAP = mN
    mse = MSE(test, wMAP,testR)
    end = time.time()

    print("best mse: " + str(mse))
    print("best landa: "+ str(alpha/beta))
    print(name + " runtime: " + str(end-start))
    print("alpha: " + str(alpha))
    print("beta: " + str(beta))
    print("\n")


if __name__ == "__main__":

    names = ["crime", "wine", "artlarge", "artsmall"]
    for name in names:
        train_name = "train-"+name+".csv"
        trainR_name = "trainR-"+name+".csv"
        test_name = "test-"+name+".csv"
        testR_name = "testR-"+name+".csv"

        train = read_file(train_name)
        trainR = read_file(trainR_name)
        test = read_file(test_name)
        testR = read_file(testR_name)

        print(name + " dataset:")
        print("\n")
        print("Task1:")
        task1(train, trainR, test, testR)
        print("Task2:")
        task2(train, trainR, test, testR)
        print("Task3:")
        task3(train, trainR, test, testR)
        print(50*"~")

    

