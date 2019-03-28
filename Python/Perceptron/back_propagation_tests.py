import unittest
import back_propagation

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


if __name__ == '__main__':
    unittest.main()