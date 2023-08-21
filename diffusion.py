import torch 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SymmetricMatrix(torch.nn.Module):
    def __init__(self):
        super(SymmetricMatrix, self).__init__()
        return
    def forward(self, x):
        return self.matvec(x)
    def matvec(self, x):
        raise NotImplementedError("Must implement matvec(x) in subclass")
    def sqrt(self):
        raise NotImplementedError("Must implement sqrt() in subclass")
    def inv(self):
        raise NotImplementedError("Must implement inv() in subclass")
    def mat_add(self, M):
        raise NotImplementedError("Must implement mat_add(M in subclass")
    def mat_sub(self, M):
        raise NotImplementedError("Must implement mat_sub(M) in subclass")
    def mat_mul(self, M):
        raise NotImplementedError("Must implement mat_mul(M) in subclass")
    def __add__(self, M):
        return self.mat_add(M)
    def __sub__(self, M):
        return self.mat_sub(M)
    def __mul__(self, x):
        return self.matvec(x)
    def __matmul__(self, M):
        return self.mat_mul(M)




class ScalarMatrix(SymmetricMatrix):
    def __init__(self, scalar):
        super(ScalarMatrix, self).__init__()
        assert isinstance(scalar, torch.Tensor), "scalar must be a tensor"
        assert torch.isfinite(scalar).all(), "scalar must be finite"
        # assert (scalar >= 0).all(), "scalar must be non-negative"
        self.scalar = scalar
        return
    def matvec(self, x):
        scalar = self.scalar
        while len(scalar.shape) < len(x.shape):
            scalar = scalar.unsqueeze(-1)
        return scalar * x
    def sqrt(self):
        return ScalarMatrix(torch.sqrt(self.scalar))
    def inv(self):
        return ScalarMatrix(1.0/self.scalar)
    def mat_add(self, M):
        assert isinstance(M, ScalarMatrix), "M must be a ScalarMatrix"
        return ScalarMatrix(self.scalar + M.scalar)
    def mat_sub(self, M):
        assert isinstance(M, ScalarMatrix), "M must be a ScalarMatrix"
        return ScalarMatrix(self.scalar - M.scalar)
    def mat_mul(self, M):
        assert isinstance(M, ScalarMatrix), "M must be a ScalarMatrix"
        return ScalarMatrix(self.scalar * M.scalar)





class DiagonalMatrix(SymmetricMatrix):
    def __init__(self, diagonal):
        super(DiagonalMatrix, self).__init__()
        assert isinstance(diagonal, torch.Tensor), "diagonal must be a tensor"
        # assert len(diagonal.shape) == 1, "diagonal must be a vector"
        assert torch.isfinite(diagonal).all(), "diagonal must be finite"
        # assert (diagonal >= 0).all(), "diagonal must be non-negative"
        self.diagonal = diagonal
        return
    def matvec(self, x):
        return self.diagonal * x
    def sqrt(self):
        return DiagonalMatrix(torch.sqrt(self.diagonal))
    def inv(self):
        return DiagonalMatrix(1.0/self.diagonal)
    def mat_add(self, M):
        assert isinstance(M, DiagonalMatrix), "M must be a DiagonalMatrix"
        return DiagonalMatrix(self.diagonal + M.diagonal)
    def mat_sub(self, M):
        assert isinstance(M, DiagonalMatrix), "M must be a DiagonalMatrix"
        return DiagonalMatrix(self.diagonal - M.diagonal)
    def mat_mul(self, M):
        assert isinstance(M, DiagonalMatrix), "M must be a DiagonalMatrix"
        return DiagonalMatrix(self.diagonal * M.diagonal)



class FourierMatrix(SymmetricMatrix):
    def __init__(self, fourier_transfer_function):
        super(FourierMatrix, self).__init__()
        assert isinstance(fourier_transfer_function, torch.Tensor), "fourier_transfer_function must be a tensor"
        # assert fourier_transfer_function.shape[-1] == 2, "fourier_transfer_function must be a complex tensor"
        assert torch.isfinite(fourier_transfer_function).all(), "fourier_transfer_function must be finite"
        self.fourier_transfer_function = fourier_transfer_function
        return
    def matvec(self, x):
        return torch.fft.ifft2(self.fourier_transfer_function*torch.fft.fft2(x, dim=(-2,-1)), dim=(-2,-1)).real
    def sqrt(self):
        return FourierMatrix(torch.sqrt(self.fourier_transfer_function))
    def inv(self):
        return FourierMatrix(1.0/self.fourier_transfer_function)
    def mat_add(self, M):
        assert isinstance(M, FourierMatrix), "M must be a FourierMatrix"
        return FourierMatrix(self.fourier_transfer_function + M.fourier_transfer_function)
    def mat_sub(self, M):
        assert isinstance(M, FourierMatrix), "M must be a FourierMatrix"
        return FourierMatrix(self.fourier_transfer_function - M.fourier_transfer_function)
    def mat_mul(self, M):
        assert isinstance(M, FourierMatrix), "M must be a FourierMatrix"
        return FourierMatrix(self.fourier_transfer_function * M.fourier_transfer_function)




class DiffusionModel(torch.nn.Module):
    def __init__(self,
                score_estimator):        
        """
        score_estimator:  a torch.nn.Module that takes as input either (x_t,t) or (x_t,y,t) and outputs a score estimate of the same shape as x_t
        
        """

        super(DiffusionModel, self).__init__()

        self.score_estimator = score_estimator
        
        # self.H = H
        # self.H_prime = H_prime
        # self.Sigma = Sigma
        # self.Sigma_prime = Sigma_prime
        # self.sample_x_T = sample_x_T

        return


    # these are the functions that must be implemented in the subclass
    
    def H(self, t):
        # input is a (batch_size) tensor of times between 0 and 1
        # output is a SymmetricMatrix with the same batch size, representing the signal transfer matrix
        raise NotImplementedError("Must implement H(t) in subclass")
    
    def H_prime(self, t):
        # input is a (batch_size) tensor of times between 0 and 1
        # output is a SymmetricMatrix with the same batch size, representing the time derivative of the signal transfer matrix
        raise NotImplementedError("Must implement H_prime(t) in subclass")
    
    def Sigma(self, t):
        # input is a (batch_size) tensor of times between 0 and 1
        # output is a SymmetricMatrix with the same batch size, representing the noise covariance matrix
        raise NotImplementedError("Must implement Sigma(t) in subclass")
    
    def Sigma_prime(self, t):
        # input is a (batch_size) tensor of times between 0 and 1
        # output is a SymmetricMatrix with the same batch size, representing the time derivative of the noise covariance matrix
        raise NotImplementedError("Must implement Sigma_prime(t) in subclass")
    
    def sample_x_T(self, batch_size=None, y=None):
        # input is either a (batch_size) tensor of times between 0 and 1 or a (batch_size, y_dim) tensor of y values
        # output is a (batch_size, x_dim) tensor of samples from either p(x_T) or p(x_T | y)
        raise NotImplementedError("Must implement sample_x_T(batch_size, y) in subclass")

    def forward(    self, 
                    x_t, 
                    t,
                    y=None, 
                    ):
        if y is None:
            # unconditional score estimation
            score_estimate = self.score_estimator(x_t, t)
        else:
            # conditional score estimation
            score_estimate = self.score_estimator(x_t, y, t)
        return score_estimate
    
    def sample_x_t_given_x_0_and_t_and_epsilon( self,
                                                x_0,
                                                t,
                                                epsilon):
        
        # epsilon is assumed to be a tensor of random noise, normal distribution, mean=0, std=1

        # signal transfer matrix
        H_t = self.H(t)           
        
        # noise covariance matrix
        Sigma_t = self.Sigma(t) 
        
        # sample from p(x_t | x_0) = N(H_t * x_0, Sigma_t)  
        x_t = H_t * x_0 + Sigma_t.sqrt() * epsilon

        return x_t
    
    def sample_x_t_given_x_0_and_t( self,
                                    x_0,
                                    t):
        
        # sample random noise, normal distribution, mean=0, std=1
        epsilon = torch.randn_like(x_0)

        # sample from p(x_t | x_0) = N(H_t * x_0, Sigma_t)
        x_t = self.sample_x_t_given_x_0_and_t_and_epsilon(x_0, t, epsilon)

        return x_t
    
    def sample_x_t_plus_tau_given_x_t_and_t_and_tau(  self,
                                                    x_t,
                                                    t,
                                                    tau):
        # signal transfer matrix at time t
        H_t = self.H(t)

        # noise covariance matrix at time t
        Sigma_t = self.Sigma(t)

        # signal transfer matrix at time t+dt
        H_t_plus_tau = self.H(t+tau)

        # noise covariance matrix at time t+dt
        Sigma_t_plus_tau = self.Sigma(t+tau)

        # sample from p(x_t+tau | x_t) = N(M1 * x_t, M2)
        M1 = H_t_plus_tau @ H_t.inv()
        M2 = Sigma_t_plus_tau - M1 @ M1 @ Sigma_t
        x_t_plus_tau = M1 * x_t + M2.sqrt() * torch.randn_like(x_t)

        return x_t_plus_tau
    
    def sample_x_t_plus_dt_given_x_t_and_t_and_dt(  self,
                                                    x_t,
                                                    t,
                                                    dt):
        
        # this assumed dt is small, so we can use a first order approximation
        assert dt > 0, "dt must be positive"

        # signal transfer matrix at time t
        H_t = self.H(t)

        # noise covariance matrix at time t
        Sigma_t = self.Sigma(t)

        # time derivative of signal transfer matrix at time t
        H_prime_t = self.H_prime(t)

        # time derivative of noise covariance matrix at time t
        Sigma_prime_t = self.Sigma_prime(t)

        # define the coefficients of the stochastic differential equation
        F = H_prime_t @ H_t.inv()
        f = F * x_t
        G2 = Sigma_prime_t - H_prime_t @ H_t.inv() @ Sigma_t - H_t.inv() @ H_prime_t @ Sigma_t
        G = G2.sqrt()

        # sample from p(x_t+dt | x_t) = N(x_t + f dt, G @ G dt)
        x_t_plus_dt = x_t + f*dt + G*torch.randn_like(x_t)*torch.sqrt(dt)

        return x_t_plus_dt
    
    def sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_score_estimate( self,
                                                                    x_t,
                                                                    t,
                                                                    dt,
                                                                    score_estimate):
        
        # this assumed dt is small, so we can use a first order approximation
        assert dt > 0, "dt must be positive"

        # this method applies the Andersen formula for the reverse-time stochastic differential equation

        # signal transfer matrix at time t
        H_t = self.H(t)

        # noise covariance matrix at time t
        Sigma_t = self.Sigma(t)

        # time derivative of signal transfer matrix at time t
        H_prime_t = self.H_prime(t)

        # time derivative of noise covariance matrix at time t
        Sigma_prime_t = self.Sigma_prime(t)

        # define the coefficients of the stochastic differential equation
        F = H_prime_t @ H_t.inv()
        f = F * x_t
        G2 = Sigma_prime_t - H_prime_t @ H_t.inv() @ Sigma_t - H_t.inv() @ H_prime_t @ Sigma_t
        G = G2.sqrt()

        # sample from p(x_t-dt | x_t) = N(x_t - [f  - G @ G * score_estimate] dt, G @ G dt)
        x_t_minus_dt = x_t - f*dt + G2*score_estimate*dt + G*torch.randn_like(x_t)*torch.sqrt(torch.tensor(dt))

        return x_t_minus_dt
    
    def sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_epsilon_estimate( self,
                                                                        x_t,
                                                                        t,
                                                                        dt,
                                                                        epsilon_estimate):
        
        # noise covariance matrix at time t
        Sigma_t = self.Sigma(t)

        # compute the score estimate
        score_estimate = Sigma_t.inv().sqrt() * epsilon_estimate * -1

        # sample from p(x_t-dt | x_t) = N(x_t - [f  - G @ G * score_estimate] dt, G @ G dt)
        x_t_minus_dt = self.sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_score_estimate(x_t, t, dt, score_estimate)

        return x_t_minus_dt
    
    def sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_x_0_estimate( self,
                                                                    x_t,
                                                                    t,
                                                                    dt,
                                                                    x_0_estimate):
        
        # signal transfer matrix at time t
        H_t = self.H(t)

        # noise covariance matrix at time t
        Sigma_t = self.Sigma(t)

        # compute the score estimate
        score_estimate = Sigma_t.inv() * (x_t - H_t * x_0_estimate) * -1

        # sample from p(x_t-dt | x_t) = N(x_t - [f  - G @ G * score_estimate] dt, G @ G dt)
        x_t_minus_dt = self.sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_score_estimate(x_t, t, dt, score_estimate)

        return x_t_minus_dt
    
    def sample_x_t_minus_dt_given_x_t_and_y_and_t_and_dt_and_score_estimate( self,
                                                                    x_t,
                                                                    t,
                                                                    dt,
                                                                    score_estimate,
                                                                    y,
                                                                    log_likelihood):
        
        # this assumed dt is small, so we can use a first order approximation
        assert dt > 0, "dt must be positive"

        # this method applies the Andersen formula for the reverse-time stochastic differential equation

        # signal transfer matrix at time t
        H_t = self.H(t)

        # noise covariance matrix at time t
        Sigma_t = self.Sigma(t)

        # time derivative of signal transfer matrix at time t
        H_prime_t = self.H_prime(t)

        # time derivative of noise covariance matrix at time t
        Sigma_prime_t = self.Sigma_prime(t)

        # define the coefficients of the stochastic differential equation
        F = H_prime_t @ H_t.inv()
        f = F * x_t
        G2 = Sigma_prime_t - H_prime_t @ H_t.inv() @ Sigma_t - H_t.inv() @ H_prime_t @ Sigma_t
        G = G2.sqrt()

        score_prior = score_estimate.clone()

        # compute the likelihood score
        
        # x_t.requires_grad = True  # Ens/ure that x_t requires gradient computation
        x_t = x_t.detach().clone()
        x_t.requires_grad = True
        outputs = log_likelihood(x_t, y, t)  # Forward pass through the model
        outputs.backward(torch.ones_like(outputs))  # Compute gradient w.r.t. each batch element
        score_likelihood = x_t.grad  # Extract the gradients


        # apply Bayes rule to get the posterior score
        score_posterior = score_prior + score_likelihood

        # sample from p(x_t-dt | x_t) = N(x_t - [f  - G @ G * score_estimate] dt, G @ G dt)
        x_t_minus_dt = x_t - f*dt + G2*score_posterior*dt + G*torch.randn_like(x_t)*torch.sqrt(torch.tensor(dt))

        return x_t_minus_dt
    
    def compute_loss( self, 
                    x_0_batch, 
                    y_batch=None,
                    loss_name=None):
        
        # if loss_name is not provided, use the default loss function
        if loss_name is None:
            loss_name = 'jensen-fisher'

        # assert that the loss function name is valid
        if loss_name not in ['elbo', 'jensen-fisher']:
            raise ValueError("loss must be either 'elbo' or 'jensen-fisher'")

        # sample random times between 0 and 1
        #   (different times for each batch element)
        t = torch.rand((x_0_batch.shape[0])).to(device)
        
        # sample random noise, normal distribution, mean=0, std=1
        epsilon_target = torch.randn_like(x_0_batch).to(device)

        # noise covariance matrix at time t
        Sigma_t = self.Sigma(t)

        # compute the target score
        score_target = Sigma_t.inv().sqrt() * epsilon_target * -1
        
        # sample from p(x_t | x_0) = N(H_t * x_0, Sigma_t)
        x_t = self.sample_x_t_given_x_0_and_t_and_epsilon(x_0_batch, t, epsilon_target)
        
        # now estimate epsilon given x_t, y, t
        if y_batch is not None:
            score_estimate = self.score_estimator(x_t, y_batch, t)
        else:
            score_estimate = self.score_estimator(x_t, t)

        # compute the loss
        if loss_name == 'jensen-fisher':
            loss = torch.mean((score_target - score_estimate)**2)
        elif loss_name == 'elbo':
            epsilon_estimate = Sigma_t.sqrt() * score_estimate * -1
            loss = torch.mean((epsilon_target - epsilon_estimate)**2)

        return loss
    
    def forward_process(self, 
                        x_0, 
                        num_steps=128,
                        t_stop=1.0,
                        returnFullProcess=True):

        # determine the batch size
        batch_size = x_0.shape[0]

        # start time is always 0.0
        t = torch.zeros((batch_size)).float().to(device)

        # dt is t_stop/num_steps
        dt = t_stop/num_steps

        # initialize x_t to the forward process input x_0
        x_t = x_0

        # if returnFullProcess, initialize a tensor to hold all the x_t's
        if returnFullProcess:
            x_t_all = torch.zeros((num_steps, batch_size, 1, x_0.shape[-2], x_0.shape[-1])).to(device)

        for i in range(num_steps):

            # sample next time
            x_t = self.sample_x_t_plus_tau_given_x_t_and_t_and_tau(x_t, t, dt)

            # update t
            t = t + dt
            
            # if returnFullProcess, add to the list
            if returnFullProcess:
                x_t_all[i] = x_t 

        # if returnFullProcess, return the full process, otherwise just return the final x_t
        if returnFullProcess:
            return x_t_all
        else:
            return x_t
        
    def reverse_process(self, 
                        y=None,
                        x_T=None,
                        batch_size=1,
                        t_stop=0.0,
                        num_steps=128,
                        returnFullProcess=True,
                        verbose=False,
                        log_likelihood=None
                        ):
        
        # determine the batch size
        if y is not None:
            batch_size = y.shape[0]
            if x_T is not None:
                assert batch_size == x_T.shape[0], "y and x_T must have the same batch size"
        elif x_T is not None:
            batch_size = x_T.shape[0]
        elif batch_size is None:
            raise ValueError("Must provide either y, x_T, or batch_size")
        
        # start time is always 1.0
        t = torch.ones((batch_size)).float().to(device)

        # dt is (1-t_stop)/num_steps
        dt = (1.0-t_stop)/num_steps
        
        # initialize x_t to the reverse process input x_T
        if x_T is not None:
            x_t = x_T.clone()
        else:
            x_t = self.sample_x_T(batch_size=batch_size, y=y)

        if returnFullProcess:
            x_t_all = torch.zeros((num_steps, batch_size, 1, x_t.shape[-2], x_t.shape[-1])).to(device)
        for i in range(num_steps):

            if verbose:
                print('reverse process step ', i+1, ' of ', num_steps)

            # set up the score estimator for evaluation
            self.score_estimator.eval()
            # make a prediction
            with torch.no_grad():
                if isinstance(self.score_estimator, ConditionalScoreEstimator):
                    score_estimate = self.score_estimator(x_t, y, t)
                elif isinstance(self.score_estimator, UnconditionalScoreEstimator):
                    score_estimate = self.score_estimator(x_t, t)
                else:
                    raise ValueError("score_estimator must be an instance of either ConditionalScoreEstimator or UnconditionalScoreEstimator")
            # set the model back to training mode
            self.score_estimator.train()

            self.sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_score_estimate(x_t, t, dt, score_estimate)


            # if a model of log p(y | x_t) is provided, use it to improve the score estimate
            if log_likelihood is not None:
                # sample from p(x_t-dt | x_t) = N(x_t - [f  - G @ G * (score_prior + score_likelihood)] dt, G @ G dt)
                x_t = self.sample_x_t_minus_dt_given_x_t_and_y_and_t_and_dt_and_score_estimate(x_t, t, dt, score_estimate, y, log_likelihood)
            else:
                # sample from p(x_t-dt | x_t) = N(x_t - [f  - G @ G * score_estimate] dt, G @ G dt)
                x_t = self.sample_x_t_minus_dt_given_x_t_and_t_and_dt_and_score_estimate(x_t, t, dt, score_estimate)

            # update t
            t = t - dt
            
            # if returnFullProcess, add to the list
            if returnFullProcess:
                x_t_all[i] = x_t.detach().clone()

        # if returnFullProcess, return the full process, otherwise just return the final x_t
        if returnFullProcess:
            return x_t_all
        else:
            return x_t


class UnconditionalScoreEstimator(torch.nn.Module):
    def __init__(self):
        super(UnconditionalScoreEstimator, self).__init__()
        return
    def forward(self, x_t, t):
        raise NotImplementedError("Must implement forward(x_t, t) in subclass")
    
class ConditionalScoreEstimator(torch.nn.Module):
    def __init__(self):
        super(ConditionalScoreEstimator, self).__init__()
        return
    def forward(self, x_t, y, t):
        raise NotImplementedError("Must implement forward(x_t, y, t) in subclass")






class ScalarDiffusionModel(DiffusionModel):
    def __init__(self,  signal_scale_func,
                        noise_variance_func,
                        sample_x_T_func=None,
                         **kwargs):
        
        super(ScalarDiffusionModel, self).__init__(**kwargs)

        self.signal_scale_func = signal_scale_func
        self.noise_variance_func = noise_variance_func
        self.sample_x_T_func = sample_x_T_func

        return
    
    def H(self, t):
        return ScalarMatrix(self.signal_scale_func(t))
    
    def H_prime(self, t):
        jacobians = [torch.autograd.functional.jacobian(self.signal_scale_func, t[i].unsqueeze(0)) for i in range(t.shape[0])]
        return ScalarMatrix(torch.stack(jacobians).squeeze())

    def Sigma(self, t):
        return ScalarMatrix(self.noise_variance_func(t))
    
    def Sigma_prime(self, t):
        jacobians = [torch.autograd.functional.jacobian(self.noise_variance_func, t[i].unsqueeze(0)) for i in range(t.shape[0])]
        return ScalarMatrix(torch.stack(jacobians).squeeze())

    def sample_x_T(self, batch_size=None, y=None):
        if self.sample_x_T_func is None:
            return torch.randn((batch_size, 1))
        else:
            return self.sample_x_T_func(batch_size=batch_size, y=y)
        


class ScalarDiffusionModel_VariancePreserving(ScalarDiffusionModel):
    def __init__(self,  alpha_bar_func,
                        sample_x_T_func=None,
                         **kwargs):
        
        # for the variance preserving model, the signal scale is defined as sqrt(alpha_bar_t)
        def signal_scale_func(t):
            return torch.sqrt(alpha_bar_func(t))

        # for the variance preserving model, the noise variance is defined as (1 - alpha_bar_t**2)
        def noise_variance_func(t):
            return 1.0 - alpha_bar_func(t)
        
        super(ScalarDiffusionModel_VariancePreserving, self).__init__(signal_scale_func=signal_scale_func,
                                                                        noise_variance_func=noise_variance_func,
                                                                        sample_x_T_func=sample_x_T_func,
                                                                        **kwargs)
        return



class ScalarDiffusionModel_VariancePreserving_LinearSchedule(ScalarDiffusionModel_VariancePreserving):
    def __init__(self, **kwargs):
        
        # define the linear schedule function
        def alpha_bar_func(t):
            return (1.0 - 0.999*t)
        
        super(ScalarDiffusionModel_VariancePreserving_LinearSchedule, self).__init__(alpha_bar_func=alpha_bar_func,
                                                                                        **kwargs)
        return


class DiagonalDiffusionModel(DiffusionModel):
    def __init__(self,  signal_scale_func,
                        noise_variance_func,
                        signal_scale_derivative_func=None,
                        noise_variance_derivative_func=None,
                        sample_x_T_func=None,
                         **kwargs):
        
        super(DiagonalDiffusionModel, self).__init__(**kwargs)

        self.signal_scale_func = signal_scale_func
        self.noise_variance_func = noise_variance_func
        self.signal_scale_derivative_func = signal_scale_derivative_func
        self.noise_variance_derivative_func = noise_variance_derivative_func
        self.sample_x_T_func = sample_x_T_func

        return
    
    def H(self, t):
        return DiagonalMatrix(self.signal_scale_func(t))
    
    def H_prime(self, t):
        if self.signal_scale_derivative_func is None:
            jacobians = [torch.autograd.functional.jacobian(self.signal_scale_func, t[i].unsqueeze(0)).squeeze(-1)for i in range(t.shape[0])]
            return DiagonalMatrix(torch.stack(jacobians).squeeze(1))
        else:
            return DiagonalMatrix(self.signal_scale_derivative_func(t))
        
    def Sigma(self, t):
        return DiagonalMatrix(self.noise_variance_func(t))
    
    def Sigma_prime(self, t):
        if self.noise_variance_derivative_func is None:
            jacobians = [torch.autograd.functional.jacobian(self.noise_variance_func, t[i].unsqueeze(0)).squeeze(-1) for i in range(t.shape[0])]
            return DiagonalMatrix(torch.stack(jacobians).squeeze(1))
        else:
            return DiagonalMatrix(self.noise_variance_derivative_func(t))
        
    def sample_x_T(self, batch_size=None, y=None):
        if self.sample_x_T_func is None:
            return torch.randn((batch_size, 1))
        else:
            return self.sample_x_T_func(batch_size=batch_size, y=y)




class FourierDiffusionModel(DiffusionModel):
    def __init__(self,  
                 modulation_transfer_function_func,
                 noise_power_spectrum_func,
                 modulation_transfer_function_derivative_func=None,
                 noise_power_spectrum_derivative_func=None,
                 sample_x_T_func=None,
                 **kwargs):
        
        super(FourierDiffusionModel, self).__init__(**kwargs)

        self.modulation_transfer_function_func = modulation_transfer_function_func
        self.noise_power_spectrum_func = noise_power_spectrum_func
        self.modulation_transfer_function_derivative_func = modulation_transfer_function_derivative_func
        self.noise_power_spectrum_derivative_func = noise_power_spectrum_derivative_func
        self.sample_x_T_func = sample_x_T_func

        return
    
    def H(self, t):
        return FourierMatrix(self.modulation_transfer_function_func(t))
    
    def H_prime(self, t):
        if self.modulation_transfer_function_derivative_func is None:
            jacobians = [torch.autograd.functional.jacobian(self.modulation_transfer_function_func, t[i].unsqueeze(0)).squeeze(-1) for i in range(t.shape[0])]
            return FourierMatrix(torch.stack(jacobians).squeeze(1))
        else:
            return FourierMatrix(self.modulation_transfer_function_derivative_func(t))
        
    def Sigma(self, t):
        return FourierMatrix(self.noise_power_spectrum_func(t))
    
    def Sigma_prime(self, t):
        if self.noise_power_spectrum_derivative_func is None:
            jacobians = [torch.autograd.functional.jacobian(self.noise_power_spectrum_func, t[i].unsqueeze(0)).squeeze(-1) for i in range(t.shape[0])]
            return FourierMatrix(torch.stack(jacobians).squeeze(1))
        else:
            return FourierMatrix(self.noise_power_spectrum_derivative_func(t))
        
    def sample_x_T(self, batch_size=None, y=None):
        if self.sample_x_T_func is None:
            return torch.randn((batch_size, 1))
        else:
            return self.sample_x_T_func(batch_size=batch_size, y=y)
# list of classes in this file:
__all__ = ['SymmetricMatrix',
            'ScalarMatrix',
            'DiagonalMatrix',
            'FourierMatrix',
            'DiffusionModel',
            'UnconditionalScoreEstimator',
            'ConditionalScoreEstimator',
            'ScalarDiffusionModel',
            'ScalarDiffusionModel_VariancePreserving',
            'ScalarDiffusionModel_VariancePreserving_LinearSchedule',
            'DiagonalDiffusionModel',
            'FourierDiffusionModel',
            ]
