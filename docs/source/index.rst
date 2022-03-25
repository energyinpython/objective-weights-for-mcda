Welcome to objective_weights_mcda documentation!
===================================

This is Python 3 library dedicated for multi-criteria decision analysis with criteria weights determined by objective weighting methods.
This library includes:

- The VIKOR method `VIKOR`

- Weighting methods for determination criteria weights:

	- `equal_weighting` (Equal weighting method)
	- `entropy_weighting` (Entropy weighting method)
	- `std_weighting` (Standard deviation weighting method)
	- `critic_weighting` (CRITIC weighting method)
	- `gini_weighting` (Gini coefficient-based weighting method)
	- `merec_weighting` (MEREC weighting method)
	- `stat_var_weighting` (Statistical variance weighting method)
	- `cilos_weighting` (CILOS weighting method)
	- `idocriw_weighting` (IDOCRIW weighting method)
	- `angle_weighting` (Angle weighting method)
	- `coeff_var_weighting` (Coefficient of variation weighting method)
	
- Correlation coefficients:

	- `spearman` (Spearman rank correlation coefficient)
	- `weighted_spearman` (Weighted Spearman rank correlation coefficient)
	- `pearson_coeff` (Pearson correlation coefficient)
	
- Methods for normalization of decision matrix:

	- `linear_normalization` (Linear normalization)
	- `minmax_normalization` (Minimum-Maximum normalization)
	- `max_normalization` (Maximum normalization)
	- `sum_normalization` (Sum normalization)
	- `vector_normalization` (Vector normalization)
	
- additions:

	- `rank_preferences` (Method for ordering alternatives according to their preference values obtained with MCDA methods)
	
Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
