# MNIST Examples of Scalar, Diagonal, and Fourier Diffusion Models

This repository contains examples of training diffusion models on the MNIST Dataset.

The purpose of this repository is to provide a simple, easy-to-understand example of how to train diffusion models using a small dataset. It should be possible to train these models on a high-quality CPU without the need for a GPU. The code is heavily commented and should be easy to follow along.

In addition to standard score-based diffusion models, this work introduces a new method to control the drift and noise schedules with matrix operators $H(t)$, the signal transfer matrix which returns forward process mean when applied to the ground truth image and $\Sigma(t)$, the noise covariance matrix. 


## Theoretical Methods

The full description of this method is described in [Fourier Diffusion Models: A Method to Control MTF and NPS in Score-Based Stochastic Image Generation](https://arxiv.org/abs/2303.13285).

We consider a forward stochastic process defined by:

$$
p(x_t | x_0) = \mathcal{N}(x_t ; H(t) x_0, \Sigma(t))
$$

where $x_0$ is the ground truth image and $x_t$ is the degraded image at time $t$. The signal transfer matrix $H(t)$ is a linear operator that returns the forward process mean when applied to the ground truth image, and $\Sigma(t)$ is a symmetric positive-semidefinite covariance matrix which defines the noise magnitude/correlations in the forward process.

In this work, we consider $H(t)$ and $\Sigma(t)$ as scalar matrices (scalar value times identity), diagonal matrices (element-wise multiplication by a vector), and Fourier matrices (circulant matrices diagonalized by the discrete Fourier transform). 

We can show that this forward process is defined by the following stochastic differential equation:

$$
dx_t = H'(t) H^{-1}(t) dt + (\Sigma'(t) - 2 H'(t) H^{-1}(t)\Sigma(t))^{\frac{1}{2}} dw_t
$$

where $w_t$ is a standard Brownian motion. The time-reversed process is given by the following stochastic differential equation:

$$
dx_t = [-H'(t) H^{-1}(t)  - (\Sigma'(t) + 2 H'(t) H^{-1}(t)\Sigma(t)) \nabla \log{p(x_t)}] dt + (\Sigma'(t) + 2 H'(t) H^{-1}(t)\Sigma(t))^{1/2} dw_t
$$

where $\nabla \log{p(x_t)}$ is the score function or the gradient of the log-prior evaluated at $x_t$. For deep learning applications, this score is estimated using a neural network.


For the special case of scalar diffusion models, we show this general formula reduces to the method presented in [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456). We can use the following parameterization for scalar diffusion models without loss of generality:

$$
H(t) = e^{-\frac{1}{2}\int_0^t \beta(s) ds} I
$$

$$
\Sigma(t) = \sigma^2(t) I
$$

resulting in the following stochastic differential equation:

$$
dx_t = -\frac{1}{2}\beta(t) x_t dt + \sqrt{\beta(t)\sigma^2(t) - \frac{d}{dt}\sigma^2(t)} dw_t
$$

which corresponds to the variance-preserving (VP) or variance-exploding (VE) diffusion models proposed in the aforementioned score-based SDE paper 

## Installation

Follow instructions to install pytorch and torchvision from [pytorch.org](https://pytorch.org/). CPU-only should work, but GPU is recommended.

You should also install matplotlib using `pip install matplotlib` from the command line.

After installing the dependencies, clone this repository using `git clone https://github.com/tivnanmatt/mnist-diffusion` from the command line. Then activate the environment you installed pytorch in and you should be able to run the examples.

## Diffusion Source Code

`diffusion.py` contains the source code for the diffusion model. Here is a list of the classes it contains:

- `SymmetricMatrix`: A class for representing symmetric matrices. This is used to represent the signal transfer matrix $H(t)$ and the noise covariance matrix $\Sigma(t)$. It is an abstract base class that requires the child class to implement methods for matrix-vector multiplication, matrix-matrix multiplication, matrix inverse, and matrix square root.
- `ScalarMatrix`: A child class of `SymmetricMatrix` that represents a scalar matrix. Or a scalar times the identity matrix.
- `DiagonalMatrix`: A child class of `SymmetricMatrix` that represents a diagonal matrix. This is equivalent to element-wise multiplication by a vector. It is defined by a vector the same size as the input argument that defines the diagonal. 
- `FourierMatrix`: A child class of `SymmetricMatrix` that represents a circulant matrix that is diagonalized by the discrete Fourier transform. It uses the fast Fourier transform to compute matrix-vector and matrix-matrix multiplication. It is defined by the fourier transfer coefficients. 

- `DiffusionModel`: An abstract base class that represents a diffusion model. It is assumed that the forward process is defined by two time-dependent matrix-valued functions that return the signal transfer matrix $H(t)$ and noise covariance matrix $\Sigma(t)$. The child class must implement $H(t)$ and $\Sigma(t)$ as well as their time derivatives, $H'(t)$ and $\Sigma'(t)$. It also must implement a method to sample from the diffusion model at the final time step $t=1$. The only input to initialize the class is the score estimator. This is a function that takes in an image as well as the current time step and returns the score function or the gradient of the log-prior. This class implements methods to sample from the forward stochastic process, the unconditional reverse process, and the conditional reverse process. It also implements a method to compute the log-likelihood of an image under the model.

- `ScoreEstimator`: An abstract base class that represents a score estimator. It is assumed that the score estimator is a neural network that takes in an image and returns the score function or the gradient of the log-prior. The child class must implement the neural network as well as a method to compute the log-prior. This class implements a method to compute the log-likelihood of an image under the model.

- `UnconditionalScoreEstimator`: A child class of `ScoreEstimator` that represents a score estimator for the unconditional reverse process. It is initialized with a neural network that takes in an image and returns the score function or the gradient of the log-prior. It implements a method to compute the log-likelihood of an image under the model.

- `ConditionalScoreEstimator`: A child class of `ScoreEstimator` that represents a score estimator for the conditional reverse process. It is initialized with a neural network that takes in an image and returns the score function or the gradient of the log-prior. It implements a method to compute the log-likelihood of an image under the model.

- `ScalarDiffusionModel`: A child class of `DiffusionModel` that represents a diffusion model with scalar $H(t)$ and $\Sigma(t)$. It is defined by scalar-valued functions to define the signal scale function and noise variance function as well as their time derivatives. 

- `ScalarDiffusionModel_VariancePreserving`: A child class of `ScalarDiffusionModel` that represents a diffusion model with scalar $H(t)$ and $\Sigma(t)$ that converges to zero-mean identity covariance noise. It is parameterized by a single scalar-valued function $\bar{\alpha}(t)$ that defines the signal scale function and its time derivative. The signal magnitude is defined by $\sqrt{\bar{\alpha}(t)}$ and the noise variance is defined by $1 - \bar{\alpha}(t)$.

- `ScalarDiffusionModel_VariancePreserving_LinearSchedule`: A child class of `ScalarDiffusionModel_VariancePreserving` for the special case where $\bar{\alpha}(t)$ is a linear function of time. 

- `DiagonalDiffusionModel`: A child class of `DiffusionModel` that represents a diffusion model with diagonal $H(t)$ and $\Sigma(t)$. It is defined by vector-valued functions to define the signal scale function and noise variance function as well as their time derivatives. It allows for pixel-dependent diffusion schedules which can be useful for tasks such as in-painting.

- `FourierDiffusionModel`: A child class of `DiffusionModel` that represents a diffusion model with circulant $H(t)$ and $\Sigma(t)$. It is defined by vector-valued functions to define the signal scale function and noise variance function as well as their time derivatives. It allows for spatial-frequency-dependent diffusion schedules which can be useful for tasks such as image restoration. This method allows for continuous probability flow from ground-truth images to degraded images with a specified MTF and NPS. This process requires fewer time steps to invert than conditional guidance of scalar diffusion models. The details of the method are described in this paper [Fourier Diffusion Models: A Method to Control MTF and NPS in Score-Based Stochastic Image Generation](https://arxiv.org/abs/2303.13285).

## MNIST Utilities

`mnist_utils.py` contains utilities for data loaders, score estimators, time encoders, position encoders, and bayesian classifiers specific to the MNIST dataset.

## Examples

`example_diffusion_scalar.py` contains an example of training a scalar diffusion model on the MNIST dataset. It uses a linear schedule for $\bar{\alpha}(t)$.

`example_diffusion_diagonal.py` contains an example of training a diagonal diffusion model on the MNIST dataset. It uses pixel-dependent diffusion schedules for an in-painting task.

`example_diffusion_fourier.py` contains an example of training a Fourier diffusion model on the MNIST dataset. It uses spatial-frequency-dependent diffusion schedules for an image restoration task.

`example_diffusion_classifier_guided.py` contains an example of training a Bayesian classifier on the MNIST dataset, and then using the classifier to guide the diffusion model. It uses a linear schedule for $\bar{\alpha}(t)$.
