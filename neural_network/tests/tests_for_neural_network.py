import unittest
import numpy as np
from deeplearningudcty.neural_network.code.neural_network import NeuralNetwork

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])





class TestMethods(unittest.TestCase):
	
	
	##########
	# Unit tests for network functionality
	##########
	
	def test_train(self):
		# Test that weights are updated correctly on training
		network = NeuralNetwork(3, 2, 1, 0.5)
		network.weights_input_to_hidden = test_w_i_h.copy()
		network.weights_hidden_to_output = test_w_h_o.copy()
		
		network.train(inputs, targets)
		
		self.assertTrue(np.allclose(network.weights_hidden_to_output,
		                            np.array([[0.37275328],
		                                      [-0.03172939]])))
		
		
		self.assertTrue(np.allclose(network.weights_input_to_hidden,
		                            np.array([[0.10562014, -0.20185996],
		                                      [0.39775194, 0.50074398],
		                                      [-0.29887597, 0.19962801]])))
	
	def test_run(self):
		# Test correctness of run method
		network = NeuralNetwork(3, 2, 1, 0.5)
		network.weights_input_to_hidden = test_w_i_h.copy()
		network.weights_hidden_to_output = test_w_h_o.copy()
		
		self.assertTrue(np.allclose(network.run(inputs), 0.09998924))


suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)