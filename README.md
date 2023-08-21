# MNIST Examples of Scalar, Diagonal, and Fourier Diffusion Models

This repository contains examples of training diffusion models on the MNIST Dataset.

The purpose of this repository is to provide a simple, easy-to-understand example of how to train diffusion models using a small dataset. It should be possible to train these models on a high-quality CPU without the need for a GPU. The code is heavily commented and should be easy to follow along.

In addition to standard score-based diffusion models, this work introduces a new method to control the drift and noise schedules with matrix operators $**H**(t)$, the signal transfer matrix which returns forward process mean when applied to the ground truth image and $\Sigma(t)$, the noise covariance matrix. This method is described in the paper [Fourier Diffusion Models: A Method to Control MTF and NPS in Score-Based Stochastic Image Generation](https://arxiv.org/abs/2303.13285).

## Installation

Follow instructions to install pytorch and torchvision from [pytorch.org](https://pytorch.org/). CPU-only should work, but GPU is recommended.

You should also install matplotlib using `pip install matplotlib` from the command line.

After installing the dependencies, clone this repository using `git clone https://github.com/tivnanmatt/mnist-diffusion` from the command line. Then activate the environment you installed pytorch in and you should be able to run the examples.

## Diffusion Source Code

`diffusion.py` contains the source code for the diffusion model. Here is a list of the classes it contains:

- `SymmetricMatrix`: A class for representing symmetric matrices. This is used to represent the signal transfer matrix $**H**(t)$ and the noise covariance matrix $\Sigma(t)$. It is an abstract base class that requires the child class to implement methods for matrix-vector multiplication, matrix-matrix multiplication, matrix inverse, and matrix square root.
- `ScalarMatrix`: A child class of `SymmetricMatrix` that represents a scalar matrix. Or a scalar times the identity matrix.
- `DiagonalMatrix`: A child class of `SymmetricMatrix` that represents a diagonal matrix. This is equivalent to element-wise multiplication by a vector. It is defined by a vector the same size as the input argument that defines the diagonal. 
- `FourierMatrix`: A child class of `SymmetricMatrix` that represents a circulant matrix that is diagonalized by the discrete Fourier transform. It uses the fast Fourier transform to compute matrix-vector and matrix-matrix multiplication. It is defined by the fourier transfer coefficients. 

- `DiffusionModel`: An abstract base class that represents a diffusion model. It is assumed that the forward process is defined by two time-dependent matrix-valued functions that return the signal transfer matrix $**H**(t)$ and noise covariance matrix $\Sigma(t)$. The child class must implement $**H**(t)$ and $\Sigma(t)$ as well as their time derivatives, $**H**'(t)$ and $\Sigma'(t)$. It also must implement a method to sample from the diffusion model at the final time step $t=1$. The only input to initialize the class is the score estimator. This is a function that takes in an image as well as the current time step and returns the score function or the gradient of the log-prior. This class implements methods to sample from the forward stochastic process, the unconditional reverse process, and the conditional reverse process. It also implements a method to compute the log-likelihood of an image under the model.

- `ScoreEstimator`: An abstract base class that represents a score estimator. It is assumed that the score estimator is a neural network that takes in an image and returns the score function or the gradient of the log-prior. The child class must implement the neural network as well as a method to compute the log-prior. This class implements a method to compute the log-likelihood of an image under the model.

- `UnconditionalScoreEstimator`: A child class of `ScoreEstimator` that represents a score estimator for the unconditional reverse process. It is initialized with a neural network that takes in an image and returns the score function or the gradient of the log-prior. It implements a method to compute the log-likelihood of an image under the model.

- `ConditionalScoreEstimator`: A child class of `ScoreEstimator` that represents a score estimator for the conditional reverse process. It is initialized with a neural network that takes in an image and returns the score function or the gradient of the log-prior. It implements a method to compute the log-likelihood of an image under the model.

- `ScalarDiffusionModel`: A child class of `DiffusionModel` that represents a diffusion model with scalar $**H**(t)$ and $\Sigma(t)$. It is defined by scalar-valued functions to define the signal scale function and noise variance function as well as their time derivatives. 

- `ScalarDiffusionModel_VariancePreserving`: A child class of `ScalarDiffusionModel` that represents a diffusion model with scalar $**H**(t)$ and $\Sigma(t)$ that converges to zero-mean identity covariance noise. It is parameterized by a single scalar-valued function $\bar{\alpha}(t)$ that defines the signal scale function and its time derivative. The signal magnitude is defined by $\sqrt{\bar{\alpha}(t)}$ and the noise variance is defined by $1 - \bar{\alpha}(t)$.

- `ScalarDiffusionModel_VariancePreserving_LinearSchedule`: A child class of `ScalarDiffusionModel_VariancePreserving` for the special case where $\bar{\alpha}(t)$ is a linear function of time. 

- `DiagonalDiffusionModel`: A child class of `DiffusionModel` that represents a diffusion model with diagonal $**H**(t)$ and $\Sigma(t)$. It is defined by vector-valued functions to define the signal scale function and noise variance function as well as their time derivatives. It allows for pixel-dependent diffusion schedules which can be useful for tasks such as in-painting.

- `FourierDiffusionModel`: A child class of `DiffusionModel` that represents a diffusion model with circulant $**H**(t)$ and $\Sigma(t)$. It is defined by vector-valued functions to define the signal scale function and noise variance function as well as their time derivatives. It allows for spatial-frequency-dependent diffusion schedules which can be useful for tasks such as image restoration. This method allows for continuous probability flow from ground-truth images to degraded images with a specified MTF and NPS. This process requires fewer time steps to invert than conditional guidance of scalar diffusion models. The details of the method are described in this paper [Fourier Diffusion Models: A Method to Control MTF and NPS in Score-Based Stochastic Image Generation](https://arxiv.org/abs/2303.13285).

## MNIST Utilities

`mnist.py` contains utilities for loading the MNIST dataset and plotting images. It also contains a class `MNISTScoreEstimator` that is a child class of `ScoreEstimator` that represents a score estimator for the MNIST dataset. It is initialized with a neural network that takes in an image and returns the score function or the gradient of the log-prior. It implements a method to compute the log-likelihood of an image under the model.

## Examples

`example_diffusion_scalar.py` contains an example of training a scalar diffusion model on the MNIST dataset. It uses a linear schedule for $\bar{\alpha}(t)$.

`example_diffusion_diagonal.py` contains an example of training a diagonal diffusion model on the MNIST dataset. It uses pixel-dependent diffusion schedules for an in-painting task.

`example_diffusion_fourier.py` contains an example of training a Fourier diffusion model on the MNIST dataset. It uses spatial-frequency-dependent diffusion schedules for an image restoration task.

`example_diffusion_classifier_guided.py` contains an example of training a Bayesian classifier on the MNIST dataset, and then using the classifier to guide the diffusion model. It uses a linear schedule for $\bar{\alpha}(t)$.
