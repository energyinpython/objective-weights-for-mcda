Usage
=====

.. _installation:

Installation
------------

To use objective_weights_mcda, first install it using pip:

.. code-block:: python

	pip install objective-weights-mcda

Importing methods from objective-weights-mcda package
-------------------------------------

Import MCDA methods from module `mcda_methods`:

.. code-block:: python

	from objective_weights_mcda.mcda_methods import VIKOR

Import weighting methods from module `weighting_methods`:

.. code-block:: python

	from objective_weights_mcda import weighting_methods as mcda_weights

Import normalization methods from module `normalizations`:

.. code-block:: python

	from objective_weights_mcda import normalizations as norm_methods

Import correlation coefficient from module `correlations`:

.. code-block:: python

	from objective_weights_mcda import correlations as corrs

Import method for ranking alternatives according to prefernce values from module `additions`:

.. code-block:: python

	from objective_weights_mcda.additions import rank_preferences



Usage examples
----------------------


The VIKOR method
__________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in ascending order according to preference values.

.. code-block:: python

	import numpy as np
	from objective_weights_mcda.mcda_methods import VIKOR
	from objective_weights_mcda.additions import rank_preferences

	# provide decision matrix in array numpy.darray
	matrix = np.array([[8, 7, 2, 1],
	[5, 3, 7, 5],
	[7, 5, 6, 4],
	[9, 9, 7, 3],
	[11, 10, 3, 7],
	[6, 9, 5, 4]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.4, 0.3, 0.1, 0.2])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([1, 1, 1, 1])

	# Create the VIKOR method object providing v parameter. The default v parameter is set to 0.5, so if you do not provide it, v will be equal to 0.5.
	vikor = VIKOR(v = 0.625)

	# Calculate the VIKOR preference values of alternatives
	pref = vikor(matrix, weights, types)

	# Generate ranking of alternatives by sorting alternatives ascendingly according to the VIKOR algorithm (reverse = False means sorting in ascending order) according to preference values
	rank = rank_preferences(pref, reverse = False)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.6399 1.     0.6929 0.2714 0.     0.6939]
	Ranking:  [3 6 4 2 1 5]
	

Correlation coefficents
__________________________

Spearman correlation coefficient

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of correlation coefficient between two vectors

.. code-block:: python

	import numpy as np
	from objective_weights_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `spearman` coefficient
	coeff = corrs.spearman(R, Q)
	print('Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Spearman coeff:  0.9

	
	
Weighted Spearman correlation coefficient

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of correlation coefficient between two vectors

.. code-block:: python

	import numpy as np
	from objective_weights_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `weighted_spearman` coefficient
	coeff = corrs.weighted_spearman(R, Q)
	print('Weighted Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Weighted Spearman coeff:  0.8833

	
	
Pearson correlation coefficient

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of correlation coefficient between two vectors

.. code-block:: python

	import numpy as np
	from objective_weights_mcda import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `pearson_coeff` coefficient
	coeff = corrs.pearson_coeff(R, Q)
	print('Pearson coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Pearson coeff:  0.9
	
	
	
Methods for criteria weights determination
___________________________________________

Entropy weighting method

Parameters
	X : ndarray
		Decision matrix with performance values of m alternatives and n criteria
Returns
	ndarray
		vector of criteria weights. Profit criteria are represented by 1 and cost by -1.
		
.. code-block:: python

	import numpy as np
	from objective_weights_mcda import weighting_methods as mcda_weights

	matrix = np.array([[30, 30, 38, 29],
	[19, 54, 86, 29],
	[19, 15, 85, 28.9],
	[68, 70, 60, 29]])
	
	weights = mcda_weights.entropy_weighting(matrix)
	
	print('Entropy weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Entropy weights:  [0.463  0.3992 0.1378 0.    ]
	

CRITIC weighting method

Parameters
	X : ndarray
		Decision matrix with performance values of m alternatives and n criteria
	types : ndarray
		Vector of criteria types. Profit criteria are represented by 1 and cost by -1.
Returns
	ndarray
		Vector of criteria weights
		
.. code-block:: python

	import numpy as np
	from objective_weights_mcda import weighting_methods as mcda_weights

	matrix = np.array([[5000, 3, 3, 4, 3, 2],
	[680, 5, 3, 2, 2, 1],
	[2000, 3, 2, 3, 4, 3],
	[600, 4, 3, 1, 2, 2],
	[800, 2, 4, 3, 3, 4]])
	
	weights = mcda_weights.critic_weighting(matrix)
	
	print('CRITIC weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	CRITIC weights:  [0.157  0.2495 0.1677 0.1211 0.1541 0.1506]


Standard deviation weighting method

Parameters
	X : ndarray
		Decision matrix with performance values of m alternatives and n criteria
	types : ndarray
		Vector of criteria types. Profit criteria are represented by 1 and cost by -1.
Returns
	ndarray
		Vector of criteria weights
		
.. code-block:: python

	import numpy as np
	from objective_weights_mcda import weighting_methods as mcda_weights

	matrix = np.array([[0.619, 0.449, 0.447],
	[0.862, 0.466, 0.006],
	[0.458, 0.698, 0.771],
	[0.777, 0.631, 0.491],
	[0.567, 0.992, 0.968]])
	
	weights = mcda_weights.std_weighting(matrix)
	
	print('Standard deviation weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Standard deviation weights:  [0.2173 0.2945 0.4882]
	
	
	
Normalization methods
______________________

Here is an example of `vector_normalization` usage. Other normalizations provided in module `normalizations`, namely `minmax_normalization`, `max_normalization`,
`sum_normalization`, `linear_normalization` are used in analogous way.


Vector normalization

Parameters
	X : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	types : ndarray
		Criteria types. Profit criteria are represented by 1 and cost by -1.
Returns
	ndarray
		Normalized decision matrix

.. code-block:: python

	matrix = np.array([[8, 7, 2, 1],
    [5, 3, 7, 5],
    [7, 5, 6, 4],
    [9, 9, 7, 3],
    [11, 10, 3, 7],
    [6, 9, 5, 4]])

    types = np.array([1, 1, 1, 1])

    norm_matrix = norms.vector_normalization(matrix, types)
    print('Normalized matrix: ', np.round(norm_matrix, 4))
	
Output

.. code-block:: console

	Normalized matrix:  [[0.4126 0.3769 0.1525 0.0928]
	 [0.2579 0.1615 0.5337 0.4642]
	 [0.361  0.2692 0.4575 0.3714]
	 [0.4641 0.4845 0.5337 0.2785]
	 [0.5673 0.5384 0.2287 0.6499]
	 [0.3094 0.4845 0.3812 0.3714]]
