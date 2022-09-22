import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt, inf

# This function gets the name of txt file and return a 2d list
def read_file(file_name):
    my_data = genfromtxt('pp4data\\' + file_name, delimiter=',')
    return my_data

# This function gets the optimum alpha, optimum beta, test set, train set, train labels and linear kernel and returns the 2 lists that first one is the mean for normal dist of all predicted labesl for test and second one is their variance (for linear kernel)
def predict_t(alpha, beta, test, x, t, K):
    
    n = len(x)
    cn = np.dot((1/alpha),K) + np.dot((1/beta),np.identity(n))
    cn_inverse = np.linalg.inv(cn)

    t_mean = [0 for i in range(len(test))]
    t_var = [0 for i in range(len(test))]

    for i in range(len(test)):
        c = (1/alpha)*linear_kernel(test[i], test[i]) + (1/beta)
        v = []
        for j in range(n):
            v.append((1/alpha)*linear_kernel(x[j], test[i]))
    
        mean = np.dot(np.dot(v, cn_inverse), np.transpose(t))
        t_mean[i] = mean
        t_var[i] = c- np.matmul(np.matmul(v, cn_inverse), np.transpose(v))
    return t_mean, t_var


# This function gets two vectors and return one value based on the linear kernel formula
def linear_kernel(x1, x2):
    k = np.dot(x1, x2.T) + 1
    return k


# Plot the true function and the function using the kernel
def plot_truefunction_1D(y_predicted, name):
    x = np.arange(-3, 3, 0.01)
    y = []
    for i in x:

        if i<-1.5:
            y.append(1)
        elif i<1.5:
            y.append(np.sin(6*i))
        else:
            y.append(-1)

    plt.plot(x, y)
    plt.plot(x, y_predicted)
    plt.legend(["True Function", "Predicted Function"])
    plt.savefig("1D_visualization\\" + name)
    plt.show()

#---------------------------------------
# This function gets two vectors and s and return one value based on the RBF kernel formula
def RBF_kernel(x1, x2, s):
    sum = 0
    k = 0
    if x1.ndim == 0:
        sum = (x1 - x2)**2
        k = np.exp((-1/2)*sum/(s**2))
    else:
        for i in range(len(x1)):
            sum += (x1[i] - x2[i])**2
        k = np.exp((-1/2)*sum/(s**2))
    return k


# This function gets the optimum alpha, optimum beta, test set, train set, train labels and linear kernel and returns the 2 lists that first one is the mean for normal dist of all predicted labesl for test and second one is their variance (for RBF kernel)
def predict_t_RBF(alpha, beta, s, test, x, t):
    n = len(x)

    K = [[0 for i in range(n)] for j in range(n)]
    for k in range(n):
        for j in range(n):
            K[k][j] = RBF_kernel(x[k], x[j], s)

    cn = np.dot((1/alpha),K) + np.dot((1/beta),np.identity(n))
    cn_inverse = np.linalg.inv(cn)

    t_mean = [0 for i in range(len(test))]
    t_var = [0 for i in range(len(test))]

    for i in range(len(test)):
        c = (1/alpha)*RBF_kernel(test[i], test[i], s) + (1/beta)
        v = []
        for j in range(n):
            v.append((1/alpha)*RBF_kernel(x[j], test[i], s))
    
        mean = np.dot(np.dot(v, cn_inverse), np.transpose(t))
        t_mean[i] = mean
        t_var[i] = c- np.matmul(np.matmul(v, cn_inverse), np.transpose(v))
    return t_mean, t_var


name = "1D"

train_name = "train-"+name+".csv"
trainR_name = "trainR-"+name+".csv"
test_name = "test-"+name+".csv"
testR_name = "testR-"+name+".csv"

train = read_file(train_name)
trainR = read_file(trainR_name)
test = read_file(test_name)
testR = read_file(testR_name)

test = np.arange(-3, 3, 0.01)

n = len(train)

#-------------------------------------------- linear
K = [[0 for i in range(n)] for j in range(n)]
for k in range(n):
    for j in range(n):
        K[k][j] = linear_kernel(train[k], train[j])

l_mean, var = predict_t(7.5, 1.9, test, train, trainR, K)

plot_truefunction_1D(l_mean, "linear_kernel.png")

#--------------------------------------------- RBF

l_mean, var = predict_t_RBF(1.25836472513683, 16.673987689140233, 0.2222648616645106, test, train, trainR)

plot_truefunction_1D(l_mean, "RBF_kernel.png")



