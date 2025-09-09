# Neural Network for High-Dimensional longitudinal data

This codebase contains some routines and modules to analyse longitudinal data, possibly high-dimensional, here intended as having a number if input features bigger than the sample size.

#### The classic approach to repeated measurements
Classic statistical approaches for the analysis of longitudinal data, and in general repeated mesurements, make use of mixed models. Although mixed models can be flexible enough to include non-linear terms, the choice of the non-linear components is not straightforward, and hard to know in advance, especially in explorative studies.\
Moreover, when high-dimensional covariates are involved, variable selection (or shrinkage) must be included in the model, and not many options are available for such combination.

### Over-parametrized Neural Networks

Out aim is to provide an alternative approach which relies on using over-parametrized Neural Networks (NN). Over-parametrized NN have been studied in the literature, showing how the massive amount of parameters (or weights in NN terminology) can help the model generalize beyond the training data, therefore avoiding overfitting. Although no theorethical results exist concerning the ability of such models to prevent overfitting, results on simulations seem to support this claims.

### Building the model
Any model can be made of single module blocks, for example:

![Plot SVG](graphs_vae_attention/full_blocks.svg)
*Figure: Example of longitudinal model*

Here we use a Variational Autoencoder (VAE) to estimate a lower-dimensional latent representation of the high-dimensional input features, for example genomics data. The output of the VAE is then pre-processed before being fed into a Transformer block. We use a Transformer, and more specifically we make use of the self-attention dot-product layer, to extract information about the temporal dynamic of the problem. Whether we have only baseline features, or dynamic features, we want to understand how the input affects the evolution of the outcome $y$ over time. The attention mechanism is perfect for this purpose.

### Explaining the model predictions
NN are commonly regarder as black-box models that cannot be explained, this is especially true from the point of view of classic statistical approach to data analysis. However, this is hardly true nowdays, given the availability of tools and techniques that have been developed to dissect a NN model and understand how predictions are obtained.\
In our specific appliactions we can investigate how the latent space is affected by input features and we can understand how the generative process work, for example using perturbations. We also provide a custom class for the Transformer, in order to allow the extraction of the attention weights, which can be used to asses the time dependence. SHAP, the most common feature explainability tool, can be effectively used to understand which features affect the final predictions, as well as middle layers. Moreover, specific to our architecture, the VAE reconstruction space provide useful information about feature importance, and effectively acts as a shrinkage layer for covariates that are not relevant.
