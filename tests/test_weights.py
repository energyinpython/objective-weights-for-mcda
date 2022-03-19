import unittest
import numpy as np
from objective_weights_mcda import normalizations as norms
from objective_weights_mcda import weighting_methods as mcda_weights


# Test for CRITIC weighting
class Test_CRITIC(unittest.TestCase):

    def test_critic(self):
        """Test based on paper Tuş, A., & Aytaç Adalı, E. (2019). The new combination with CRITIC and WASPAS methods 
        for the time and attendance software selection problem. Opsearch, 56(2), 528-538."""

        matrix = np.array([[5000, 3, 3, 4, 3, 2],
        [680, 5, 3, 2, 2, 1],
        [2000, 3, 2, 3, 4, 3],
        [600, 4, 3, 1, 2, 2],
        [800, 2, 4, 3, 3, 4]])

        types = np.array([-1, 1, 1, 1, 1, 1])

        test_result = mcda_weights.critic_weighting(matrix, types)
        real_result = np.array([0.157, 0.249, 0.168, 0.121, 0.154, 0.151])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for MEREC weighting
class Test_MEREC(unittest.TestCase):

    def test_merec(self):
        """Test based on paper Keshavarz-Ghorabaee, M., Amiri, M., Zavadskas, E. K., Turskis, Z., & Antucheviciene, 
        J. (2021). Determination of objective weights using a new method based on the removal 
        effects of criteria (MEREC). Symmetry, 13(4), 525."""

        matrix = np.array([[450, 8000, 54, 145],
        [10, 9100, 2, 160],
        [100, 8200, 31, 153],
        [220, 9300, 1, 162],
        [5, 8400, 23, 158]])

        types = np.array([1, 1, -1, -1])

        test_result = mcda_weights.merec_weighting(matrix, types)
        real_result = np.array([0.5752, 0.0141, 0.4016, 0.0091])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for Entropy weighting
class Test_Entropy(unittest.TestCase):

    def test_Entropy(self):
        """Test based on paper Xu, X. (2004). A note on the subjective and objective integrated approach to 
        determine attribute weights. European Journal of Operational Research, 156(2), 
        530-532."""

        matrix = np.array([[30, 30, 38, 29],
        [19, 54, 86, 29],
        [19, 15, 85, 28.9],
        [68, 70, 60, 29]])

        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.entropy_weighting(matrix, types)
        real_result = np.array([0.4630, 0.3992, 0.1378, 0.0000])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))

    def test_Entropy2(self):
        """Test based on paper Zavadskas, E. K., & Podvezko, V. (2016). Integrated determination of objective 
        criteria weights in MCDM. International Journal of Information Technology & Decision 
        Making, 15(02), 267-283."""

        matrix = np.array([[3.0, 100, 10, 7],
        [2.5, 80, 8, 5],
        [1.8, 50, 20, 11],
        [2.2, 70, 12, 9]])

        types = np.array([-1, 1, -1, 1])

        test_result = mcda_weights.entropy_weighting(matrix, types)
        real_result = np.array([0.1146, 0.1981, 0.4185, 0.2689])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))
        

# Test for CILOS weighting
class Test_CILOS(unittest.TestCase):

    def test_cilos(self):
        """Test based on paper Alinezhad, A., & Khalili, J. (2019). New methods and applications in multiple 
        attribute decision making (MADM) (Vol. 277). Cham: Springer."""

        matrix = np.array([[3, 100, 10, 7],
        [2.500, 80, 8, 5],
        [1.800, 50, 20, 11],
        [2.200, 70, 12, 9]])

        types = np.array([-1, 1, -1, 1])

        test_result = mcda_weights.cilos_weighting(matrix, types)
        real_result = np.array([0.334, 0.220, 0.196, 0.250])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


    def test_cilos2(self):
        """Test based on paper Zavadskas, E. K., & Podvezko, V. (2016). Integrated determination of objective 
        criteria weights in MCDM. International Journal of Information Technology & Decision 
        Making, 15(02), 267-283."""

        matrix = np.array([[0.6, 100, 0.8, 7],
        [0.72, 80, 1, 5],
        [1, 50, 0.4, 11],
        [0.818, 70, 0.667, 9]])

        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.cilos_weighting(matrix, types)
        real_result = np.array([0.3343, 0.2199, 0.1957, 0.2501])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for IDOCRIW weighting
class Test_IDOCRIW(unittest.TestCase):

    def test_idocriw(self):
        """Test based on paper Zavadskas, E. K., & Podvezko, V. (2016). Integrated determination of objective 
        criteria weights in MCDM. International Journal of Information Technology & Decision 
        Making, 15(02), 267-283."""

        matrix = np.array([[3.0, 100, 10, 7],
        [2.5, 80, 8, 5],
        [1.8, 50, 20, 11],
        [2.2, 70, 12, 9]])

        types = np.array([-1, 1, -1, 1])

        test_result = mcda_weights.idocriw_weighting(matrix, types)
        real_result = np.array([0.1658, 0.1886, 0.35455, 0.2911])
        self.assertEqual(list(np.round(test_result, 3)), list(np.round(real_result, 3)))


# Test for Angle weighting
class Test_Angle(unittest.TestCase):

    def test_angle(self):
        """Test based on paper Shuai, D., Zongzhun, Z., Yongji, W., & Lei, L. (2012, May). A new angular method to 
        determine the objective weights. In 2012 24th Chinese Control and Decision Conference 
        (CCDC) (pp. 3889-3892). IEEE."""

        matrix = np.array([[30, 30, 38, 29],
        [19, 54, 86, 29],
        [19, 15, 85, 28.9],
        [68, 70, 60, 29]])

        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.angle_weighting(matrix, types)
        real_result = np.array([0.4150, 0.3612, 0.2227, 0.0012])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for Coefficient of Variation weighting
class Test_Coeff_var(unittest.TestCase):

    def test_coeff_var(self):
        """Test based on paper Shuai, D., Zongzhun, Z., Yongji, W., & Lei, L. (2012, May). A new angular method to 
        determine the objective weights. In 2012 24th Chinese Control and Decision Conference 
        (CCDC) (pp. 3889-3892). IEEE."""

        matrix = np.array([[30, 30, 38, 29],
        [19, 54, 86, 29],
        [19, 15, 85, 28.9],
        [68, 70, 60, 29]])
        
        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.coeff_var_weighting(matrix, types)
        real_result = np.array([0.4258, 0.3610, 0.2121, 0.0011])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for Standard Deviation weighting
class Test_STD(unittest.TestCase):

    def test_std(self):
        """Test based on paper Sałabun, W., Wątróbski, J., & Shekhovtsov, A. (2020). Are mcda methods benchmarkable? 
        a comparative study of topsis, vikor, copras, and promethee ii methods. Symmetry, 12(9), 
        1549."""

        matrix = np.array([[0.619, 0.449, 0.447],
        [0.862, 0.466, 0.006],
        [0.458, 0.698, 0.771],
        [0.777, 0.631, 0.491],
        [0.567, 0.992, 0.968]])
        
        types = np.array([1, 1, 1])

        test_result = mcda_weights.std_weighting(matrix, types)
        real_result = np.array([0.217, 0.294, 0.488])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))
        


def main():
    test_critic = Test_CRITIC()
    test_critic.test_critic()

    test_merec = Test_MEREC()
    test_merec.test_merec()

    test_entropy = Test_Entropy()
    test_entropy.test_Entropy()
    test_entropy.test_Entropy2()

    test_cilos = Test_CILOS()
    test_cilos.test_cilos()
    test_cilos.test_cilos2()

    test_idocriw = Test_IDOCRIW()
    test_idocriw.test_idocriw()

    test_angle = Test_Angle()
    test_angle.test_angle()

    test_coeff_var = Test_Coeff_var()
    test_coeff_var.test_coeff_var()

    test_std = Test_STD()
    test_std.test_std()
    

if __name__ == '__main__':
    main()

