import unittest
import perceptron

class TestStringMethods(unittest.TestCase):
    def test_cyka(self):
        p1 = perceptron.Perceptron(activation=4)
        self.assertEqual(p1.get_weights(), [0,0,0])

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
