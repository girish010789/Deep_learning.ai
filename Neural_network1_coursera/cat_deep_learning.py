import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import image
from scipy import ndimage
from calc_sigmoid import sigmoid 
from initialize_with_zeros import initialize_with_zeros
from propagate import propagate
from dataset import load_dataset
from optimize import optimize
from predict import predict
from model import model

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

##Display cat-data for different indexes (eg: index = 25 displays the 25th cat image in this dataset)
#index = 25
#plt.imshow(train_set_x_orig[index])
#print ("y = "+ str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture.")

##print number of training and test set and height and width of each image , here images are considered to be square
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

##standardize dataset by dividing each row by 255 (since each pixel value ranges from 0-255) 
train_set_x_flatten = train_set_x_flatten/255
test_set_x_flatten = test_set_x_flatten/255

##What you need to remember:
##Common steps for pre-processing a new dataset are:
##Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
##Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
##"Standardize" the data

##lets build logistic regression to determine the cat for m training examples with each example having nx*nx*3 features(X) with same number of weights(w) and using sigmoid activation function to calculate y and cost function.

##Merge everything to a model
#train_y = train_set_y.shape
d = model(train_set_x_flatten, train_set_y, test_set_x_flatten, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

##plot curves
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

#learning_rates = [0.01, 0.001, 0.0001]
#models = {}
#for i in learning_rates:
#    print ("learning rate is: " + str(i))
#    models[str(i)] = model(train_set_x_flatten, train_set_y, test_set_x_flatten, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
#    print ('\n' + "-------------------------------------------------------" + '\n')
#
#for i in learning_rates:
#    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
#
#plt.ylabel('cost')
#plt.xlabel('iterations')
#
#legend = plt.legend(loc='upper center', shadow=True)
#frame = legend.get_frame()
#frame.set_facecolor('0.90')
#plt.show()
