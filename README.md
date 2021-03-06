# objective-weights-for-mcda

This is Python 3 library dedicated for multi-criteria decision analysis with criteria weights determined by objective weighting methods.
The documentation is provided [here](https://objective-weights-for-mcda.readthedocs.io/en/latest/)

This library is a project that is currently under active development. Recently other MCDA methods have been added, namely COPRAS, ARAS, 
and PROMETHEE II. More methods will be added soon. Other interesting libraries by this author include pyrepo-mcda (focused on MCDA methods 
based on reference objects and distance metrics for MCDA methods), objective-weighting (focused on objective criteria weighting methods for MCDA), 
and distance-metrics-mcda (focused on distance metrics for MCDA).

# Installation
Downloading and installation of `objective-weights-mcda` package can be done with using pip

```
pip install objective-weights-mcda
```

# Methods
`mcda_methods` includes:
- `vikor` with VIKOR method

Other modules include:
- `additions` include `rank_preference` method for ranking alternatives according to MCDA score

- `correlations` include: 
	- Spearman rank correlation coefficient `spearman`, 
	- Weighted Spearman rank correlation coefficient `weighted_spearman`,
	- Pearson correlation coefficient `pearson_coeff`
	
- `normalizations` with methods for decision matrix normalization:
	- `linear_normalization` - Linear normalization,
	- `minmax_normalization` - Minimum- Maximum normalization,
	- `max_normalization` - Maximum normalization,
	- `sum_normalization` - Sum normalization,
	- `vector_normalization` - Vector normalization
	
- `weighting_methods` include 11 objective weighting methods for determination of criteria weights (significance) without decision-maker involvement:
	- `equal_weighting` - Equal weighting method
	- `entropy_weighting` - Entropy weighting method
	- `std_weighting` - Standard deviation weighting method
	- `critic_weighting` - CRITIC weighting method
	- `gini_weighting` - Gini coefficient-based weighting method
	- `merec_weighting` - MEREC weighting method
	- `stat_var_weighting` - Statistical variance weighting method
	- `cilos_weighting` - CILOS weighting method
	- `idocriw_weighting` - IDOCRIW weighting method
	- `angle_weighting` - Angle weighting method
	- `coeff_var_weighting` - Coefficient of variation weighting method
	
Examples of usage of `objective_weights_mcda` are provided on [GitHub](https://github.com/energyinpython/objective-weights-for-mcda) in [examples](https://github.com/energyinpython/objective-weights-for-mcda/tree/main/examples)

## License
This package called `objective-weights-mcda` was created by Aleksandra B??czkiewicz. It is licensed under the terms of the MIT license.

## Note
This project is under active development.
