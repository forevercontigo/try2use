from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import xlrd,xlwt
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import use_batch
def data_in_one(inputdata):
	data=[]
	for lst in inputdata:
		min_number = min(lst)
		max_number = max(lst)
		tmp = []
		for  i in lst:
			new = (i - min_number)/(max_number-min_number)
			tmp.append(new)
		data.append(tmp)
	return data

def get_Batch(data, label, batch_size):
	input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32 ) 
	x_batch, y_batch = tf.train.batch(input_queue,\
	 batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
	return x_batch, y_batch

def xls_convert():
	print('xls_convert')
	X = []
	original_data = xlrd.open_workbook('new.xls')
	table = original_data.sheets()[0]
	rows = table.nrows
	#First col is label and the rest 33 col are training data	
	#for test,because the my computer performance,I only use 10 rows 
	#rows = 200
	Y = table.col_values(2)[:rows]
	mean = np.mean(Y)
	print('mean',mean)
	label = []
	for i in Y:
		if i>mean:
			n = 1.0
			label.append(n)
		else:
			n = 0.0
			label.append(n)
	#print(label)
	label = np.array(label).reshape(rows,1)
	label = label.astype(np.float32)
	tmp=[]
	for row in range(rows):
		tmp=table.row_values(row)[0:2]
		tmp.extend(table.row_values(row)[3:])
		X.append(tmp)
	return X,label,rows
def main():
	print('main')
	X,Y,samples = xls_convert()
	X = data_in_one(X)
	print('sample number: ',samples)
	print(Y)
	X_shape = np.array(X).shape
	print('X shape',X_shape)
	Y_shape = np.array(Y).shape
	print('Y shape',Y_shape)
#*************choose train and test data*****************
	PERCENTAGE = 0.8
	divide = int(X_shape[0]*PERCENTAGE)
	print(divide)
	train_x = X[0:divide]
	train_y = Y[0:divide]
	test_x = X[divide:]
	test_y = Y[divide:]
#**************divide data into batch* ************
	batch_size = 100
	x_batch,y_batch=get_Batch(train_x,train_y,batch_size)
	
#number of training example
	x_row = batch_size
	#number of features 
	x_col = X_shape[1]
	x = tf.placeholder(tf.float32,shape = [None,x_col])
	y = tf.placeholder(tf.float32,shape = [None,1])
	#Nh=Ns/(α∗(Ni+No))
	#Ni = number of input neurons.
	#No = number of output neurons.
	#Ns = number of samples in training data set.
	#α = an arbitrary scaling factor usually 2-10
	hidden_size_1 = 5
	hidden_size_2 = 5
	hidden_size_3 = 5
	hidden_size_4 = 5
	hidden_size_5 = 5
	output_size = batch_size
#***********input layer*****************
	W1 = tf.Variable(tf.random_normal(shape = [x_col,hidden_size_1]))
	b1 = tf.Variable(tf.random_normal(shape = [hidden_size_1]))
	h1 = tf.nn.relu(tf.matmul(x,W1)+b1)
#***********hidden layer 1 *****************
	W2 = tf.Variable(tf.random_normal(shape = [hidden_size_1,hidden_size_2]))
	b2 = tf.Variable(tf.random_normal(shape = [hidden_size_2]))
	h2 = tf.nn.relu(tf.matmul(h1,W2)+b2)
#************hidden layer 2*****************
	W3 = tf.Variable(tf.random_normal(shape = [hidden_size_2,hidden_size_3]))
	b3 = tf.Variable(tf.random_normal(shape = [hidden_size_3]))
	h3 = tf.nn.relu(tf.matmul(h2,W3)+b3)
#************hidden layer 3*****************
	W4 = tf.Variable(tf.random_normal(shape = [hidden_size_3,hidden_size_4]))
	b4 = tf.Variable(tf.random_normal(shape = [hidden_size_4]))
	h4 = tf.nn.relu(tf.matmul(h3,W4)+b4)
#************hidden layer 4*****************
	W5 = tf.Variable(tf.random_normal(shape = [hidden_size_4,hidden_size_5]))
	b5 = tf.Variable(tf.random_normal(shape = [hidden_size_5]))
	h5 = tf.nn.relu(tf.matmul(h4,W5)+b5)
#************hidden layer 5*****************
	W6 = tf.Variable(tf.random_normal(shape = [hidden_size_5,output_size]))
	b6 = tf.Variable(tf.random_normal(shape = [output_size]))
	h6 = tf.nn.relu(tf.matmul(h5,W6)+b6)
#***********output layer*****************
	W7 = tf.Variable(tf.random_normal(shape = [output_size,1]))
	b7= tf.Variable(tf.random_normal(shape = [1]))
	logits = tf.nn.relu(tf.matmul(h6,W7)+b7)
	pred = tf.argmax( tf.nn.softmax(logits))
#regularization,avoid over-fit
	regularization = tf.nn.l2_loss(h1)\
					+ tf.nn.l2_loss(h2) \
					+ tf.nn.l2_loss(h3) \
					+ tf.nn.l2_loss(logits)

	beta = 0.001 # L2 正则化系数
	learning_rate = 0.2 # 学习速率
	loss = tf.reduce_mean(\
			tf.nn.softmax_cross_entropy_with_logits(\
				labels=y, logits=logits) + \
			beta * regularization) 
	optimizer = tf.train.AdamOptimizer(\
				learning_rate).minimize(loss)
	
	init_global=tf.global_variables_initializer()
	init_local=tf.local_variables_initializer()
	loss_total=[]
#**************accuracy***************
	predictions_correct = tf.cast(tf.equal(logits,y), tf.float32)
	accuracy = tf.reduce_mean(predictions_correct)
	with tf.Session() as sess:
		sess.run(init_global)
		sess.run(init_local)
# 开启协调器
		coord = tf.train.Coordinator()
# 使用start_queue_runners 启动队列填充
		threads = tf.train.start_queue_runners(sess,coord)
		print('************training**********')
		epoch = 0
		total_accuracy=[]
		total_train_accuracy=[]
		try:
			while not coord.should_stop():
				data,label = sess.run([x_batch,y_batch])
				_,l,prediction,train_accuracy,logit_result=\
										sess.run([optimizer,loss,pred,accuracy,logits],feed_dict={x:data,y:label})
				correct_prediction = tf.equal(label,prediction)
				count = 0
				
				test_accuracy = sess.run(accuracy, \
					feed_dict={x:test_x,y:test_y})
				total_accuracy.append(test_accuracy)
				total_train_accuracy.append(train_accuracy)
#***************  test  ************************
				epoch = epoch + 1
				loss_total.append(l)
		except tf.errors.OutOfRangeError:
			print('training finish!')
		finally:
			coord.request_stop()
		print('loss',loss_total)
		print('accuracy',total_accuracy)
		print('train accuracy',total_train_accuracy)
		plt.plot(total_train_accuracy)
		plt.show()
				
main()

print("asdsaddsa")
