import unittest
import back_propagation
from perceptron import Perceptron
from functions import p_sigmoid, p_sigmoidDerivative

class TestColumn(unittest.TestCase):
    def test_column_creation(self):
        nodes = [1,2,3]
        column = back_propagation.Column(nodes)
        self.assertEqual(column.get_nodes(), [1,2,3])

    def test_input_setup(self):
        nodes = [1,2,3]
        column = back_propagation.Column(nodes)
        column.set_inputs([[1,2,3], [2,3,4]])
        self.assertEqual(column.get_inputs(), [[1,2,3], [2,3,4]])

    def test_column_linking(self):
        nodes = [1,2,3]
        second_nodes = [3,4,5]
        first_column = back_propagation.Column(nodes)
        second_column = back_propagation.Column(second_nodes)
        first_column.set_next(second_column)
        next_nodes = first_column.get_next().get_nodes()
        self.assertEqual(next_nodes, [3,4,5])

    def test_input_nodes_are_passed_forward(self):
        nodes = [4,2,3]
        second_nodes = [3,4,5]
        first_column = back_propagation.Column(nodes)
        second_column = back_propagation.Column(second_nodes)
        first_column.set_next(second_column)
        first_column.forward(back_propagation.pass_forward)
        self.assertEqual(second_column.get_inputs(), [4,2,3,1])

class TestUpdateLayers(unittest.TestCase):
    def test_output_layer(self):
        expected = -0.1480512964

        output_layer = back_propagation.Column([Perceptron(p_sigmoid, weights=[-0.3, 0.1, 0.8], id=7)])
        output_layer.set_inputs([0.7109495026,0.6224593312,1])
        error = -0.6567763068
        alpha = 2.0

        back_propagation.update_output_layer(output_layer, error, alpha)
        actual = round(output_layer.get_deltas()[0], 10)
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()