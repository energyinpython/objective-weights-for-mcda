from objective_weights_mcda.mcda_methods import VIKOR
from objective_weights_mcda.additions import rank_preferences
from objective_weights_mcda.mcda_methods import COPRAS
from objective_weights_mcda.mcda_methods import ARAS

import unittest
import numpy as np

# Test for VIKOR method
class Test_VIKOR(unittest.TestCase):

    def test_vikor(self):
        """Test based on paper Papathanasiou, J., & Ploskas, N. (2018). Vikor. In Multiple Criteria Decision Aid 
        (pp. 31-55). Springer, Cham."""

        matrix = np.array([[8, 7, 2, 1],
        [5, 3, 7, 5],
        [7, 5, 6, 4],
        [9, 9, 7, 3],
        [11, 10, 3, 7],
        [6, 9, 5, 4]])

        weights = np.array([0.4, 0.3, 0.1, 0.2])

        types = np.array([1, 1, 1, 1])

        method = VIKOR(v = 0.625)
        test_result = method(matrix, weights, types)
        real_result = np.array([0.640, 1.000, 0.693, 0.271, 0.000, 0.694])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for COPRAS method
class Test_COPRAS(unittest.TestCase):

    def test_copras(self):
        """Goswami, S., & Mitra, S. (2020). Selecting the best mobile model by applying 
        AHP-COPRAS and AHP-ARAS decision making methodology. International Journal of Data and 
        Network Science, 4(1), 27-42."""

        matrix = np.array([[80, 16, 2, 5],
        [110, 32, 2, 9],
        [130, 64, 4, 9],
        [185, 64, 4, 1],
        [135, 64, 3, 4],
        [140, 32, 3, 5],
        [185, 64, 6, 7],
        [110, 16, 3, 3],
        [120, 16, 4, 3],
        [340, 128, 6, 5]])

        weights = np.array([0.60338, 0.13639, 0.19567, 0.06456])

        types = np.array([-1, 1, 1, 1])

        method = COPRAS()
        test_result = method(matrix, weights, types)
        real_result = np.array([1, 0.85262, 0.91930, 0.68523, 0.80515, 0.72587, 0.83436, 0.79758, 0.79097, 0.79533])
        self.assertEqual(list(np.round(test_result, 5)), list(real_result))


# Test for ARAS method
class Test_ARAS(unittest.TestCase):

    def test_aras(self):
        """Goswami, S., & Mitra, S. (2020). Selecting the best mobile model by applying 
        AHP-COPRAS and AHP-ARAS decision making methodology. International Journal of Data and 
        Network Science, 4(1), 27-42."""

        matrix = np.array([[80, 16, 2, 5],
        [110, 32, 2, 9],
        [130, 64, 4, 9],
        [185, 64, 4, 1],
        [135, 64, 3, 4],
        [140, 32, 3, 5],
        [185, 64, 6, 7],
        [110, 16, 3, 3],
        [120, 16, 4, 3],
        [340, 128, 6, 5]])

        weights = np.array([0.60338, 0.13639, 0.19567, 0.06456])

        types = np.array([-1, 1, 1, 1])

        method = ARAS()
        test_result = method(matrix, weights, types)
        print(test_result)
        real_result = np.array([0.68915, 0.58525, 0.62793, 0.46666, 0.54924, 0.49801, 0.56959, 0.54950, 0.54505, 0.53549])
        self.assertEqual(list(np.round(test_result, 5)), list(real_result))


# Test for rank preferences
class Test_Rank_preferences(unittest.TestCase):

    def test_rank_preferences(self):
        """Test based on paper Papathanasiou, J., & Ploskas, N. (2018). Vikor. In Multiple Criteria Decision Aid 
        (pp. 31-55). Springer, Cham."""

        pref = np.array([0.640, 1.000, 0.693, 0.271, 0.000, 0.694])
        test_result =rank_preferences(pref , reverse = False)
        real_result = np.array([3, 6, 4, 2, 1, 5])
        self.assertEqual(list(test_result), list(real_result))


def main():
    test_vikor = Test_VIKOR()
    test_vikor.test_vikor()

    test_rank_preferences = Test_Rank_preferences()
    test_rank_preferences.test_rank_preferences()

    test_copras = Test_COPRAS()
    test_copras.test_copras()

    test_aras = Test_ARAS()
    test_aras.test_aras()


if __name__ == '__main__':
    main()