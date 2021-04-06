"""A 2-layer Neural Network.
Input: Different attributes of a Household as a vector
Output: The estimated number of children in that household as a real number
Architecture: 1 Hidden-Layer with the Activaction Function Leaky ReLU, Outputlayer has Dimension 1 and no Activation Function
Cost-function: Mean-Squared Error
Training/Testing: Use k-fold cross validation

The Dataset consists the following columns:
1 ID of Instance
2 Sex of the chef of the house
3 Age of this chef
4 Marital status of this chef
5 Stands for weather the spouse lives in the same household or not
6 Belonging to a special ethnicity
7 Number of children
8 Number of persons living in the household
9 Income of the household
10 Number of rooms in which the persons of the household sleep

Column 7 is used as label, a subset of the other columns is used as the different attributes of every household as input to the neural network
"""


#Packages
import numpy as np
import random
import pickle


"""Read in data set (there are m instances with n_x attributes and a label which equals the number of children)"""
f = open("data2.txt",'r').read().split('\n')

#transform data into list of lists, where every inner list is a household and its elements are the attributes of this household
data_temp = []
for instance in f:
	data_temp.append(instance.split(","))

#replace all entries "NA" with zero, such that column 5 becomes a numerical variable
for a in range(len(data_temp)):
	for b in range(len(data_temp[0])):
		if data_temp[a][b] == "NA":
			data_temp[a][b] = 0

#transform list of lists into array/matrix. It has dimension m=88713x11, so every row is a houshold
data_temp = np.array(data_temp)
data_temp = np.array([np.array(row) for row in data_temp])
#optionally set seed for better results
np.random.seed(0)
#shuffle data
np.random.shuffle(data_temp)

#choose columns (attributes of households) that should be considered, create matriz and transpose it
X_data = data_temp[:,[2,3,4,5,8,9,10]].T #choose columns that shall be considered			
X_data = np.vectorize(float)(X_data)
#standardize
X_mean = np.mean(X_data, axis=1).reshape(X_data.shape[0],1)


X_std = np.std(X_data, axis=1).reshape(X_data.shape[0],1)
X_data = (X_data - X_mean) / X_std
#so X-data is (n_x,m)-matrix, where every column is a vector containing all attributes of one instance

#choose column 7, containing the number of children of an household which will be our label. Create matriz and transpose it
Y_data =data_temp[:,[7]].T
Y_data = np.vectorize(float)(Y_data)
#standardize
Y_mean = np.mean(Y_data, axis=1) 
Y_std = np.std(Y_data, axis=1)
Y_data =  (Y_data - Y_mean) / Y_std
#Y_data is (1,m)-matrix containing the labels (number of children) of the instances/households

#(n_x+1,m)-matrix where every column is a vector containing all attributes and the label of one instance
Data = np.append(X_data, Y_data, axis=0)



#Dimensions
n_x = len(X_data)
m = len(X_data[0])

#Activation_function
def leaky_relu(Z):
	A = np.maximum(Z,Z*0.01)
	cache = Z
	return (A,cache)

#Derivative of leaky_relu
def leaky_relu_backwards(Z):
	def der_elementwise(Z):
		if Z < 0:
			Z *= 0.01
		return(Z)
	return(np.vectorize(der_elementwise)(Z))

#Initialize parameters w and b
def init_parameters(n_h):
	W1 = np.random.randn(n_h,n_x) * 0.1
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(1,n_h) *0.01
	b2 = 0

	parameters = {"W1" : W1, "b1" : b1, "W2" : W2, "b2" : b2}

	return parameters

def linear_forward(A,W,b):
	Z = np.dot(W,A) + b
	cache = (A,W,b)
	return (Z,cache)

def activation_forward(A_prev,W,b,activation):
	Z, linear_cache = linear_forward(A_prev,W,b)
	if activation == "leaky_relu":
		A, activation_cache = leaky_relu(Z)
	elif activation == "":
		A, activation_cache = Z, Z
	cache = (linear_cache, activation_cache)
	return(A,cache)

#forward-propagation
def two_layer_forward(X, parameters):
	caches = []
	
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	A1, cache = activation_forward(X,W1,b1,"leaky_relu")
	caches.append(cache)

	W2 = parameters["W2"]
	b2 = parameters["b2"]
	A2, cache = activation_forward(A1,W2,b2,"")
	caches.append(cache)

	Y_hat = A2
	return(Y_hat,caches)

#Cost-functions
def cost_func(Y,Y_hat):
	m = len(Y_hat[0])
	cost = 1/m * np.sum(np.power(Y-Y_hat,2))
	return(cost)

def linear_backward(dZ,cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]
	dW = 1/m* np.dot(dZ,A_prev.T)
	db = 1/m * np.sum(dZ, axis=1, keepdims = True)
	dA_prev = np.dot(W.T,dZ)
	return(dA_prev, dW, db)

def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache
	if activation == "leaky_relu":
		dZ = dA * leaky_relu_backwards(activation_cache)
	elif activation == "":
		dZ = dA * 1
	dA_prev, dW, db = linear_backward(dZ,linear_cache)
	return(dA_prev, dW, db)	

#backward-propagation
def two_layer_backward(A2,Y, caches):
	grads = {}
	dA2 = -2/m * (Y-A2)

	dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA2, caches[1], "")
	grads["dA1"] = dA_prev_temp
	grads["dW2"] = dW_temp
	grads["db2"] = db_temp

	dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA1"], caches[0], "leaky_relu")
	grads["dW1"] = dW_temp
	grads["db1"] = db_temp

	return grads

def update_parameters(parameters,grads,learning_rate):
	parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
	parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
	parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
	parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]
	return parameters

#find out how many instances were actually classified correctly
def accuracy(Y,Y_hat):
	correct = 0
	wrong = 0
	#destandardize data
	Y_hat_orig = np.round((Y_hat * Y_std) + Y_mean)
	Y_orig = (Y * Y_std) + Y_mean
	Y_orig = np.vectorize(int)(Y_orig)
	
	for dif in np.squeeze(Y_hat_orig - Y_orig):
		if dif == 0:
			correct += 1
		else:
			wrong += 1
	print("Accuracy is " + str(round((correct*100)/(correct+wrong),2))+"%")

#read in former saved parameters
def read_parameters():
	f = open("parameters.txt", "rb")
	parameters = pickle.load(f)
	f.close()
	return parameters

#save the obtained parameters in a separate text-file
def save_parameters(parameters):
	f = open("parameters.txt", "wb")
	pickle.dump(parameters,f)
	f.close()

#combine everything to the neural network
def neural_net(X,Y,n_h,learning_rate,epochs, cost_goal = 0, print_cost = False, read = False, save = False):

	#start either with former saved parameters or initialize them
	if read:
		parameters = read_parameters()
	else:
		parameters = init_parameters(n_h)
		
	for epoch in range(epochs+1):
		#compute predicted labels in every iteration and store cache for backward-propagation
		Y_hat,caches = two_layer_forward(X, parameters)
	
		#show costs to control convergence
		cost = cost_func(Y,Y_hat)
		if print_cost == True:
			if epoch % 10 == 0:
				print("epoch = " + str(epoch), "cost = " + str(cost))						

		#save gradients for gradient-descent
		grads = two_layer_backward(Y_hat,Y,caches)

		#update parameters via gradient descent
		parameters = update_parameters(parameters,grads,learning_rate)
		#stop and safe the parameters if the desired cost is obtained. The network can afterwards be called again to continue training with a different learning rate 
		if cost < cost_goal:
			save = True
			break
		
		#linearly update learning rate
		learning_rate = learning_rate - learning_rate/epochs

	#save final predicted labels and computed cost
	Y_hat,caches = two_layer_forward(X, parameters) 		
	final_cost = cost_func(Y,Y_hat)
	
	print("final cost is " + str(final_cost))									
	accuracy(Y,Y_hat)															
	
	#save obtained parameters in different text-file
	if save:
		save_parameters(parameters)
	
	return (parameters)

#k-fold
def fold_data(i,k):
	#get the indices of the columns for the training-set and the test-set		 					
	training_ind = []
	test_ind = []
	#take every k-th index, starting from the i-th
	#print("m =" +str(m))
	for j in range(m):								
		if j % k == i:						 
			test_ind.append(j)
		else:
			training_ind.append(j)

	#split Dataset into training-data and test-data
	training_data = Data[:, training_ind]
	test_data = Data[:, test_ind]

	X_train = training_data[:-1]
	Y_train = training_data[-1:]
	
	X_test = test_data[:-1]
	Y_test = test_data[-1:]
	
	return(X_train,Y_train,X_test,Y_test)

#Use k-fold cross validation to get a mean test_error and get the parameters using entire dataset
def cross_validate(k,n_h,learning_rate,epochs, mean_cost_goal = 0, print_cost = False, read = False, save = False):
	#store all costs	
	costs = []
	#repeat neural net training k-times, always using different test-set												
	for i in range(k):
		print(str(i+1) + " of " +str(k))
		#split Dataset into training-data and test-data
		X_train,Y_train,X_test,Y_test = fold_data(i,k)
		#train model
		parameters = neural_net(X_train,Y_train,n_h,learning_rate,epochs, print_cost = print_cost, read = read)
		#test gained parameters
		Y_hat = two_layer_forward(X_test,parameters)[0] 		
		#store final cost
		cost = cost_func(Y_test,Y_hat)
		print("test-cost = " + str(cost))
		accuracy(Y_test,Y_hat)
		print("")
		costs.append(cost)

	#calculate mean_cost of all regressions
	mean_cost = np.mean(costs)
	#safe ressults
	if mean_cost < mean_cost_goal:
		save = True
	#now compute parameters with entire dataset
	parameters = neural_net(X_data,Y_data,n_h,learning_rate,epochs, print_cost = print_cost, read = read, save = save)
	#show results
	print("mean-cost = "+str(mean_cost))
	return(parameters)

#Alternative use of k-fold Cross-Validation: Return parameters of set with best test-costs:
def alt_cross_validate(k,n_h,learning_rate,epochs, least_cost_goal = 0, print_cost = False, read = False, save = False):
	#store costs and their belonging parameters
	output = []
	#repeat neural net training k-times, always using different test-set											
	for i in range(k):
		print(str(i+1) + " of " +str(k))
		#split Dataset into training-data and test-data
		X_train,Y_train,X_test,Y_test = fold_data(i,k)
		#train model
		parameters = neural_net(X_train,Y_train,n_h,learning_rate,epochs, print_cost = print_cost, read = read)
		#test gained parameters
		Y_hat = two_layer_forward(X_test,parameters)[0]
		#store final cost and the belonging parameters
		cost = cost_func(Y_test,Y_hat)
		print("test-cost = " + str(cost))
		accuracy(Y_test,Y_hat)
		print("")
		output.append((cost,parameters))

	#choose the parameters which produce the least cost
	cost,parameters = min(output)
	#save parameters
	if cost < least_cost_goal:
		save = True
	if save:
		save_parameters(parameters)
	#show results								
	print("least cost = "+str(cost))
	return(parameters)



import sys
genero = sys.argv[1]
edad = sys.argv[2]
estadocivil = sys.argv[3]
a = sys.argv[4]
personas = sys.argv[5] # numero de personas que viven en el hogar
ingresos = sys.argv[6] #ingresos del hogar
cuartos = sys.argv[7] #cuartos



# 1 Sex of the chef of the house
# 2 Age of this chef
# 3 Marital status of this chef
# 4 Stands for weather the spouse lives in the same household or not
# 5 Number of persons living in the household
# 6 Income of the household
# 7 Number of rooms in which the persons of the household sleep

#one can import the following function from a different file to use the neural network and the obtained parameters from this file to make a prediction on new date
def label_new_data(X1, X2, X3, X4, X5, X6, X7):
	X = np.array([X1, X2, X3, X4, X5, X6, X7])
	#standardize X
	X = (X - X_mean) / X_std
	#call learned parameters
	parameters = read_parameters()
	#predict label with neural network architecture
	Y_hat = two_layer_forward(X,parameters)[0]
	#destandardize label
	Y_hat = Y_hat * Y_std + Y_mean
	#round label to integer
	#Y_hat =  round(Y_hat[0][0])
	Y_hat =  Y_hat[0][0]

	return Y_hat

print(label_new_data(int(genero), int(edad), int(estadocivil), int(a), int(personas), int(ingresos), int(cuartos)))
