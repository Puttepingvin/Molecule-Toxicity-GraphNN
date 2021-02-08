import numpy as np
import tensorflow as tf

print("--------Start of File---------")

#Data generator
def getDatasets(data_info_filename, data_adjacency_filename, data_features_filename,data_labels_filename):
	dataset_info = tf.data.experimental.CsvDataset(
	    data_info_filename,
	    [tf.int32, 
		tf.int32, 
		],
		header=True
	)
	dataset_adjacency = tf.data.experimental.CsvDataset(
	    data_adjacency_filename,
	    np.repeat(tf.float32, 132),
	).batch(132)
	dataset_nodes = tf.data.experimental.CsvDataset(
	    data_features_filename,
	    [tf.int32],
		header=True,
		select_cols= [1]
	).batch(132)
	dataset_labels = tf.data.experimental.CsvDataset(
	    data_labels_filename,
	    np.repeat(tf.int32, 1),
		header=True,
		select_cols= [1]
	)
	return((dataset_info, dataset_adjacency,dataset_nodes,dataset_labels))


num_atomtypes = 43
num_nodes = 132

dataset_info, dataset_adjacency,dataset_nodes,dataset_labels = getDatasets(
	data_info_filename = "train_data/train_graph_size.csv",
	data_adjacency_filename = "train_data/train_graphs.csv",
	data_features_filename = "train_data/train_nodes.csv",
	data_labels_filename = "train_data/train_labels.csv"
	)
dataset_info_testing, dataset_adjacency_testing,dataset_nodes_testing,dataset_labels_testing = getDatasets(
	data_info_filename = "test_data/test_graph_size.csv",
	data_adjacency_filename = "test_data/test_graphs.csv",
	data_features_filename = "test_data/test_nodes.csv",
	data_labels_filename = "test_data/test_labels.csv"
	)


dataset_training = tf.data.Dataset.zip((dataset_labels, dataset_nodes, dataset_adjacency))
dataset_testing = tf.data.Dataset.zip((dataset_labels_testing, dataset_nodes_testing, dataset_adjacency_testing)).repeat()

#Separate the positive and negative data
def filter_fn_pos(l,n,a):
  return tf.math.equal(l[0], 1)
def filter_fn_neg(l,n,a):
  return tf.math.equal(l[0], 0)
negative_ds = dataset_training.filter(filter_fn_neg).repeat()
positive_ds = dataset_training.filter(filter_fn_pos).repeat()
dataset_training = tf.data.experimental.sample_from_datasets([negative_ds, positive_ds], [0.1, 0.1]).shuffle(500)
iter_data = dataset_training.make_one_shot_iterator()
iter_test = dataset_testing.make_one_shot_iterator()



#Model design
size_hidden1 = 3;
size_hidden2 = 1;
drop_rate = 0.1;

W1_placeholder = np.random.rand(num_atomtypes, size_hidden1)
W2_placeholder = np.random.rand(size_hidden1, size_hidden2)
W3_placeholder = np.random.rand(1,num_nodes);
I_placeholder = np.eye(num_nodes);

H2 = tf.placeholder(tf.float32, [num_nodes, size_hidden2])

W1 = tf.Variable(W1_placeholder, dtype=tf.float32)
W2 = tf.Variable(W2_placeholder, dtype=tf.float32)
W3 = tf.Variable(W3_placeholder, dtype=tf.float32)

I = tf.constant(I_placeholder, dtype=tf.float32);

def makeguess(data):
	A = data[2]
	features = data[1]
	label = data[0]

	onedX = features[0];
	X = tf.one_hot(
	    onedX-1,
	    num_atomtypes,
	    axis=1,
	    dtype=tf.float32
	)

	Ahat = A + I;
	H0 = X;

	conns = tf.reduce_sum(Ahat,1);
	Dhat = tf.linalg.diag(tf.math.divide(tf.math.divide(conns,conns),conns))

	L = tf.matmul(tf.matmul(Dhat,Ahat),Dhat)

	H1 = tf.nn.relu(tf.matmul(tf.matmul(L,H0),W1))
	H2 = tf.nn.relu(tf.matmul(tf.matmul(L,H1),W2))
	return (tf.reshape(tf.math.tanh(tf.matmul(W3, H2)), []), label)

guess_train, label_train = makeguess(iter_data.get_next())
guess_test, label_test = makeguess(iter_test.get_next())



#Loss and optimizer definition
loss =  (label_train - guess_train)*(label_train - guess_train)
loss += 9*label_train*loss;
train_step = tf.train.AdamOptimizer(0.01).minimize(loss	)


#Main loop
with tf.compat.v1.Session() as sess:
	init = tf.compat.v1.global_variables_initializer()
	sess.run(init)
	for epoch_num in range(10): #Theyre not really epochs
		pls = np.array([]);
		alls = np.array([]);
		numcorr = 0
		j = 0;
		for i in range(1000):
			t,g,l,ls = sess.run((train_step, guess_train, label_train[0],loss))
			alls = np.concatenate((alls,ls));
			numcorr += (l-round(g)) == 0 
			if l == 1:
				j = j + 1;
				pls = np.concatenate((pls,ls));
			if i%500 == 0:
				print(i)
		print("Loss Trining", np.sum(alls)/(i+1))
		print("Loss Training Positive", np.sum(pls)/j)
		print("Corr% Training",numcorr/i*100)
		alls = np.array([]);
		pls = np.array([]);
		numcorr = 0
		numcorrpos = 0
		j = 0
		for i in range(2808):
			g,l = sess.run((guess_test, label_test[0]))
			numcorr += l == round(g)
			if l == 1:
				j = j + 1;
				numcorrpos += l == round(g)
		print("Corr% Testing", numcorr/(i+1)*100)
		print("j:", j)
		print("Corr% Testing positive", numcorrpos/max(j,1)*100)
		print("Test Score: ", numcorrpos/max(j,1)*50 + (numcorr - numcorrpos)/(i+1 - j)*50) #Data is inherently imbalanced but we want accuracy on positive and negative data to be about equal
		print("epoch:", epoch_num)
	saver = tf.train.Saver()
	saver.save(sess, './my_model', global_step = 1)
print("---------End of File----------")