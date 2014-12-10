from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer, SigmoidLayer


"""
A simple neural net implementation using PyBrain.

See http://www.pybrain.org/docs/index.html?#installation for
details on installing PyBrain.
"""


def read_data(filename):
	"""
	See http://www.pybrain.org/docs/api/datasets/classificationdataset.html

	Reads a (naive) csv file of data and converts it into
	a ClassificationDataSet. 'Naive' in this case means
	the data can be parsed by splitting on commas - i.e.,
	no quotations or escapes. I picked this file format
	because it should be trivial to convert all our data into it.

	Raises an exception when an IO error occurs.

	Parameters:
	  filename - The name of the file containing the data.
	"""
	data_file = open(filename, "r")
	data_lines = [line.split(',') for line in data_file.readlines()]
	data_file.close()

	features = [[float(f) for f in line[0:-1]] for line in data_lines]
	classes = [[int(line[-1])] for line in data_lines]
        # Workaround to make classifications zero-based
        class_min = min([c[0] for c in classes])
        for i in range(len(classes)):
                classes[i][0] -= class_min

	data_set = ClassificationDataSet(len(features[0]))
	for feature_vector, classification in zip(features, classes):
		data_set.addSample(feature_vector, classification)

	return data_set

def train(data):
	"""
	See http://www.pybrain.org/docs/tutorial/fnn.html

	Returns a neural network trained on the test data.

	Parameters:
	  data - A ClassificationDataSet for training.
	         Should not include the test data.
	"""
	network = buildNetwork(
		# This is where we specify the architecture of
		# the network.  We can play around with different
		# parameters here.
		# http://www.pybrain.org/docs/api/tools.html
		data.indim, 5, data.outdim,
		hiddenclass=SigmoidLayer,
		outclass=SoftmaxLayer
	)

	# We can fiddle around with this guy's options as well.
	# http://www.pybrain.org/docs/api/supervised/trainers.html
	trainer = BackpropTrainer(network, dataset=data)
	trainer.trainUntilConvergence(maxEpochs=20)

	return network

