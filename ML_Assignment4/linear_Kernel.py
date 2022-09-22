from numpy import genfromtxt, inf
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.numeric import identity
from numpy.lib.index_tricks import s_


# This function gets the name of txt file and return a 2d list
def read_file(file_name):
    my_data = genfromtxt('pp4data\\' + file_name, delimiter=',')
    return my_data


# This function gets the training and training labels and return index_list, alpha_list, beta_list and linear Kernel
def gp_regression(x, t):
    alpha = 1
    beta  = 1
    lr = 0.01

    n = len(x)
    
    a = np.log(alpha)
    b = np.log(beta)

    index_list = []
    alpha_list = []
    beta_list = []
    s_list = []

    K = [[0 for i in range(n)] for j in range(n)]
    for k in range(n):
        for j in range(n):
            K[k][j] = linear_kernel(x[k], x[j])

    
    for i in range(100):

        cn = np.dot((1/alpha),K) + np.dot((1/beta),np.identity(n))
        cn_inverse = np.linalg.inv(cn)

        logev = (-n/2)*math.log(2*math.pi) + - (1/2)*math.log(np.linalg.det(cn)) - (1/2)*np.dot(np.dot(t,cn_inverse), t.T)

        

        a = np.log(alpha)
        b = np.log(beta)

        derivative_a = derivative_wrt_alpha(alpha, K)
        der_log_ev_wrt_a = alpha*((-1/2)* np.trace(np.dot(cn_inverse, derivative_a)) + (1/2) * np.dot(np.dot(np.dot(np.dot(t, cn_inverse), derivative_a), cn_inverse), t.T))
        new_a = a + lr*der_log_ev_wrt_a
        new_alpha = np.exp(new_a)

        derivative_b = derivative_wrt_beta(beta, n)
        der_log_ev_wrt_b = beta*((-1/2)* np.trace(np.dot(cn_inverse, derivative_b)) + np.dot(np.dot(np.dot(np.dot(np.dot((1/2),t), cn_inverse), derivative_b), cn_inverse), t.T))
        new_b = b + lr*der_log_ev_wrt_b
        new_beta = np.exp(new_b)

        new_cn = np.dot((1/new_alpha),K) + np.dot((1/new_beta),np.identity(n))
        new_cn_inverse = np.linalg.inv(cn)

        new_logev = (-n/2)*math.log(2*math.pi) + - (1/2)*math.log(np.linalg.det(new_cn)) - (1/2)*np.dot(np.dot(t,new_cn_inverse), t.T)
        
        if (i%10 == 0) or (i == 99):
            index_list.append(i)
            alpha_list.append(new_alpha)
            beta_list.append(new_beta)

        if (new_logev - logev)/abs(logev) < 0.00001:
            index_list.append(i)
            alpha_list.append(new_alpha)
            beta_list.append(new_beta)
            break

        alpha = new_alpha
        beta = new_beta

    return index_list, alpha_list, beta_list, K


# This function gets two vectors and return one value based on the linear kernel formula
def linear_kernel(x1, x2):
    k = np.dot(x1, x2.T) + 1
    return k


# This function gets the alpha and kernel and return the derivative of C_N respect to alpha
def derivative_wrt_alpha(alpha, K):
    der = np.dot(-1/(alpha**2), K)
    return der 


# This function gets the beta and the number of training examples and return the derivative of C_N respect to beta
def derivative_wrt_beta(beta, n):
    der = np.dot(-1/(beta**2), np.identity(n))
    return der 


# This function gets the optimum alpha, optimum beta, test set, train set, train labels and linear kernel and returns the 2 lists that first one is the mean for normal dist of all predicted labesl for test and second one is their variance
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


# This function gets the true labels and predicted labels and calculate the MSE
def evaluation_MSE(t, t_prerdicted):
    n = len(t)
    sum = 0
    for i in range(n):
        sum += (t_prerdicted[i]- t[i])**2
    return sum/n


# This function gets the true lebels and the mean and variance for predicted labels and return the MNLL value
def evaluation_MNLL(t, t_predicted, t_var):
    n = len(t)
    sum = 0
    for i in range(n):
        sum +=ln_normal(t_predicted[i], t_var[i], t[i])
    return sum/n


# This function get the mean and the varicnce of the normal dist and a t and return the value of normal t in tht normal dist
def ln_normal(mean, var, t):
    nor = (1/np.sqrt(2*(np.pi*var))) * np.exp(-0.5*(((t-mean)**2)/var))
    return (-1) *np.log(nor)


# This function get the indexes and MNLL for the index and the name of the dataset and plot the MNLL respect to the iterations for that dataset
def plot(index, MNLL, name):
    plt.plot(index, MNLL, label='train')
    plt.xlabel("iteration numbers")
    plt.ylabel("MNLL")
    plt.title(name) 
    plt.savefig("linearKernel_results\\"+ name+'_result.png')
    plt.show()


if __name__ == "__main__":
    names = ["1D", "artsmall", "crime", "housing"]
    for name in names:
        train_name = "train-"+name+".csv"
        trainR_name = "trainR-"+name+".csv"
        test_name = "test-"+name+".csv"
        testR_name = "testR-"+name+".csv"

        train = read_file(train_name)
        trainR = read_file(trainR_name)
        test = read_file(test_name)
        testR = read_file(testR_name)

        index_list, alpha_list, beta_list, K = gp_regression(train, trainR)

        print(name + " dataset:")
        print("alpha: " + str(alpha_list[len(alpha_list)-1]))
        print("beta: " + str(beta_list[len(beta_list)-1]))

        MSE_list = []
        MNLL_list = []
        for i in range(len(index_list)):
            t_predicted, t_var = predict_t(alpha_list[i], beta_list[i], test, train, trainR, K)
            MSE_list.append(evaluation_MSE(testR, t_predicted))
            MNLL_list.append(evaluation_MNLL(testR, t_predicted, t_var))
        
        print("MSE: " + str(MSE_list[len(MSE_list)-1]))
        print("\n")
        plot(index_list, MNLL_list, name)

        


