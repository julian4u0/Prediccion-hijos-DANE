"""A 2-layer Neural Network.
Input: Different attributes of a Household as a vector
Output: The estimated number of children in that household as a real number
Architecture: 1 Hidden-Layer with the Activaction Function Leaky ReLU, Outputlayer has Dimension 1 and no Activation Function
Cost-function: Mean-Squared Error
Training/Testing: 60.000 Training Samples, 28713 Test Samples

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

#replace categorical variables with integer numbers
ethnicities = []
for a in range(len(data_temp)):
	
	if data_temp[a][5] == "NA":
		data_temp[a][5] = 0

	ethn = data_temp[a][6]
	if ethn not in ethnicities:
		ethnicities.append(ethn)
	data_temp[a][6] = ethnicities.index(ethn)

#transform list of lists into array/matrix. It has dimension m=88713x11, so every row is a houshold
data_temp = np.array(data_temp)
data_temp = np.array([np.array(row) for row in data_temp])
#optionally set seed for better results
np.random.seed(0)
#shuffle data
np.random.shuffle(data_temp)

#choose columns (attributes of households) that should be considered, create matriz and transpose it
X_data = data_temp[:,[2,3,4,5,6,9,10]].T #choose columns that shall be considered		
X_data = np.vectorize(float)(X_data)
#split into training and test set, 60.000 instances go into the training set, the rest into the test set
X_train = X_data[:,:60000]
X_test = X_data[:,60000:]
#standardize
X_mean = np.mean(X_train, axis=1).reshape(X_data.shape[0],1)
X_std = np.std(X_train, axis=1).reshape(X_data.shape[0],1)
X_data = (X_data - X_mean) / X_std
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
#so X-data is (n_x,m)-matrix, where every column is a vector containing all attributes of one instance

#choose column 7, containing the number of children of an household which will be our label. Create matriz and transpose it
Y_data = data_temp[:,[7]].T
Y_data = np.vectorize(float)(Y_data)
#split into training and test set, 60.000 instances go into the training set, the rest into the test set
Y_train = Y_data[:,:60000]
Y_test = Y_data[:,60000:]
#standardize
Y_mean = np.mean(Y_train, axis=1) 
Y_std = np.std(Y_train, axis=1)
Y_data = (Y_data - X_mean) / X_std
Y_train =  (Y_train - Y_mean) / Y_std
Y_test =  (Y_test - Y_mean) / Y_std
#Y_data is (1,m)-matrix containing the labels (number of children) of the instances/households

#(n_x+1,m)-matrix where every column is a vector containing all attributes and the label of one instance
Data = np.append(X_data, Y_data, axis=0)



#Dimensions
n_x = len(X_train)
m = len(X_train[0])

#Activation_function
def leaky_relu(Z):
	A = np.maximum(Z,Z*0.01)
	cache = Z
	return (A,cache)

#Derivative of leaky_relu
def leaky_relu_backwards(Z):
	def der_elementwise(Z):
		if Z < 0:
			return (0.01)
		else:
			return (1)
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
	#print("A1 = " + str(A1))
	caches.append(cache)

	W2 = parameters["W2"]
	b2 = parameters["b2"]
	A2, cache = activation_forward(A1,W2,b2,"")
	#print("A2 = " + str(A2))
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
		#print("dA check = " + str(dA))
		#print("dZ = " +str(dZ))
	elif activation == "":
		dZ = dA * 1
	dA_prev, dW, db = linear_backward(dZ,linear_cache)
	return(dA_prev, dW, db)	

#backward-propagation
def two_layer_backward(A2,Y, caches):
	grads = {}
	dA2 = (-2/m) * (Y-A2)
	#print("dA2 = " + str(dA2))

	dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA2, caches[1], "")
	grads["dA1"] = dA_prev_temp
	grads["dW2"] = dW_temp
	grads["db2"] = db_temp
	#print("dW2 = " + str(grads["dW2"]))
	#print("")
	#print("db2 = " + str(grads["db2"]))

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

#check if gradients where computed correctly
def gradient_checking(X,Y,parameters, grads):
	
	epsilon = 10**(-7)
	
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]

	W1_flat = np.matrix.flatten(np.copy(W1))
	W2_flat = np.matrix.flatten(np.copy(W2))
	b1_flat = np.matrix.flatten(np.copy(b1))
	b2_flat = np.matrix.flatten(np.copy(b2))	
	parameter = np.hstack((W1_flat,b1_flat,W2_flat,b2_flat))

	dW1 = np.matrix.flatten(np.copy(grads["dW1"]))
	dW2 = np.matrix.flatten(np.copy(grads["dW2"]))
	db1 = np.matrix.flatten(np.copy(grads["db1"]))
	db2 = np.matrix.flatten(np.copy(grads["db2"]))
	gradient = np.hstack((dW1,db1,dW2,db2))


	d_approx = np.zeros(len(gradient))
	index_add = 0

	dict_entries = ["W1","b1","W2","b2"]

	for entry in dict_entries:
		param = parameters[entry]
		for i in range(len(param)):
			for j in range(len(param[0])):
				param_plus = np.copy(param)
				param_plus[i][j] = param[i][j] + epsilon
				parameters_plus = dict(parameters)
				parameters_plus[entry] = param_plus
				
				param_minus = np.copy(param)
				param_minus[i][j] = param[i][j] - epsilon
				parameters_minus = dict(parameters)
				parameters_minus[entry] = param_minus
				
				Y_hat_plus,__ = two_layer_forward(X, parameters_plus)
				cost_plus = cost_func(Y,Y_hat_plus)
				
				Y_hat_minus,__ = two_layer_forward(X, parameters_minus)
				cost_minus = cost_func(Y,Y_hat_minus)
				
				d_approx[index_add + i*len(param[0])+j] = (cost_plus - cost_minus) / (2*epsilon)

		index_add += len(param) * len(param[0]) 

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
		#print("cost = " + str(cost))
		if print_cost == True:
			if epoch % 10 == 0:
				print("epoch = " + str(epoch), "cost = " + str(cost))						

		#save gradients for gradient-descent
		grads = two_layer_backward(Y_hat,Y,caches)

		#if needed: check gradients
		#if epoch == 2:
			#gradient_checking(X,Y,parameters, grads)

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

#Test learned parameters on test set
def test_model(X,Y):
	#gain learned parameters
	parameters = read_parameters()
	#run neural entwork to make prediction
	Y_hat,__ = two_layer_forward(X, parameters)
	#calculate costs
	cost = cost_func(Y,Y_hat)

	print(cost)
	return (cost)

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

#one can import the following function from a different file to use the neural network and the obtained parameters from this file to make a prediction on new date
def label_new_data(X1, X2, X3, X4, X5, X6, X7):
	#reshape X
	X = np.array([X1, X2, X3, X4, X5, X6, X7])
	X = X.reshape((7,1))
	#categorize Ethnicity
	X[4,0] = ethnicities.index(X[3,0])
	#standardize X
	X = (X - X_mean) / X_std
	#call learned parameters
	parameters = read_parameters()
	#predict label with neural network architecture
	Y_hat = two_layer_forward(X,parameters)[0]
	#destandardize label
	Y_hat = Y_hat * Y_std + Y_mean
	#round label to integer
	Y_hat =  round(Y_hat[0][0])
	#Y_hat =  Y_hat[0][0]

	return Y_hat

print(label_new_data(int(genero), int(edad), int(estadocivil), int(a), int(personas), int(ingresos), int(cuartos)))