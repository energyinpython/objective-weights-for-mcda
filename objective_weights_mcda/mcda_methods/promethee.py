import itertools
import numpy as np
from .mcda_method import MCDA_method


class PROMETHEE_II(MCDA_method):
    def __init__(self):
        """
        Create the PROMETHEE II method object
        """
        pass

    def __call__(self, matrix, weights, types, preference_functions, p, q):
        PROMETHEE_II._verify_input_data(matrix, weights, types)
        if len(preference_functions) != np.shape(matrix)[1]:
            raise ValueError('The list of preference functions must be equal in length to the number of criteria')
        if len(p) != np.shape(matrix)[1]:
            raise ValueError('The length of the vector p must be equal to the number of criteria')
        if len(q) != np.shape(matrix)[1]:
            raise ValueError('The length of the vector q must be equal to the number of criteria')

        return PROMETHEE_II._promethee_II(self, matrix, weights, types, preference_functions, p, q)


    # preference function type 1 (Usual criterion) requires no parameters
    # alternatives are indifferent only if they are equal to each other
    # otherwise there is a strong preference for one of them
    def _usual_function(self, d, p, q):
        if d <= 0:
            return 0
        else:
            return 1

    # preference function type 2 (U-shape criterion) requires indifference threshold (q)
    def _ushape_function(self, d, p, q):
        if d <= q:
            return 0
        else:
            return 1

    # preference function type 3 (V-shape criterion) requires threshold of absolute preference (p) 
    def _vshape_function(self, d, p, q):
        if d <= 0:
            return 0
        elif 0 <= d <= p:
            return d / p
        elif d > p:
            return 1

    # preference function type 4 (Level criterion) requires both preference and indifference thresholds (p and q)
    def _level_function(self, d, p, q):
        if d <= q:
            return 0
        elif q <= d <= p:
            return 0.5
        elif d > p:
            return 1

    # preference function type 5 (V-shape eith indifference criterion also known as linear)
    # requires both preference and indifference thresholds (p and q)
    def _linear_function(self, d, p, q):
        if d <= q:
            return 0
        elif q <= d <= p:
            return (d - q) / (p - q)
        elif d > p:
            return 1

    @staticmethod
    def _promethee_II(self, matrix, weights, types, preference_functions, p, q):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Criteria weights. The sum of weights must be equal to 1.
            types: ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.
            preference_functions : list
                List with methods containing preference functions for calculating the
                preference degree for each criterion.
            p : ndarray
                Vector with values representing the threshold of absolute preference.
            q : ndarray
                Vector with values representing the threshold of indifference.

        Returns
        --------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ----------
        >>> promethee_II = PROMETHEE_II()
        >>> preference_functions = [promethee_II._linear_function for pf in range(len(weights))]
        >>> u = np.sqrt(np.sum(np.square(np.mean(matrix, axis = 0) - matrix), axis = 0) / matrix.shape[0])
        >>> p = 2 * u
        >>> q = 0.5 * u
        >>> pref = promethee_II(matrix, weights, types, preference_functions, p, q)
        >>> rank = rank_preferences(pref, reverse = True)
        """

        m, n = matrix.shape

        H = np.zeros((m, m))

        for i, k, j in itertools.product(range(m), range(m), range(n)):
            H[i, k] += preference_functions[j](types[j] * (matrix[i, j] - matrix[k, j]), p[j], q[j]) * weights[j]

        phi_plus = np.sum(H, axis = 1) / (m - 1)
        phi_min = np.sum(H, axis = 0) / (m - 1)

        phi_net = phi_plus - phi_min
        return phi_net