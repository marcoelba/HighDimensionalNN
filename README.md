# Neural Network for High-Dimensional longitudinal data

This codebase contains some routines and modules to analyse longitudinal data, possibly high-dimensional, here intended as having a number if input features bigger than the sample size.

#### The classic approach to repeated measurements
Classic statistical approaches for the analysis of longitudinal data, and in general repeated mesurements, make use of mixed models. Although mixed models can be flexible enough to include non-linear terms, the choice of the non-linear components is not straightforward, and hard to know in advance, especially in explorative studies.\
Moreover, when high-dimensional covariates are involved, variable selection (or shrinkage) must be included in the model, and not many options are available for such combination.

### Over-parametrized Neural Networks

Out aim is to provide an alternative approach which relies on using over-parametrized Neural Networks (NN). Over-parametrized NN have been studied in the literature, showing how the massive amount of parameters (or weights in NN terminology) can help the model generalize beyond the training data, therefore avoiding overfitting. Although no theorethical results exist concerning the ability of such models to prevent overfitting, results on simulations seem to support this claims.

### Building the model
Any model can be made of single module blocks, for example:

<p align='center'><a href='https://github.com/marcoelba/HighDimensionalNN/graphs_vae_attention'><img src='.github/full_blocks.pdf' width='800'></a></p>
