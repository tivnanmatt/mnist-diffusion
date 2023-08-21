# load the mnist dataset and train a diffusion model on it

import torch
import matplotlib.pyplot as plt
import os

from diffusion import DiagonalDiffusionModel
from mnist_utils import ConditionalScoreEstimator, UNet, TimeEncoder, PositionEncoder, mnist_train_loader, mnist_test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# define parameters
# --------------------------
verbose=True
loadPreviousWeights=True
runTraining=True
runTesting=False
runReverseProcess=True
# --------------------------

if __name__ == '__main__':

    time_encoder = TimeEncoder(out_channels=32).to(device)
    position_encoder = PositionEncoder(out_channels=32).to(device)
    denoiser = UNet(in_channels=66, out_channels=1, num_base_filters=32).to(device)
    
    score_estimator = ConditionalScoreEstimator(denoiser, time_encoder, position_encoder).to(device)

    idx_inpainting = torch.zeros(1,1,28, 28).to(device).to(torch.bool)
    idx_inpainting[:,:,7:21, 7:21] = True

    def sample_x_T_func(batch_size=None, y=None):
        if y is None:
            raise ValueError('y must be provided')
        _idx_inpainting = idx_inpainting.repeat(y.shape[0], 1, 1, 1)
        x_T = y.clone()
        x_T[_idx_inpainting] = torch.randn(y[_idx_inpainting].shape).to(device)
        return x_T
 
    def signal_scale_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
        scale = torch.ones(t.shape[0], 1, 28, 28).to(device)
        _idx_inpainting = idx_inpainting.repeat(_t.shape[0], 1, 1, 1)
        scale[_idx_inpainting] = (torch.exp(-5*_t*_t))[_idx_inpainting]
        return scale
    
    def noise_variance_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
        variance = torch.zeros(t.shape[0], 1, 28, 28).to(device) + 1e-10
        _idx_inpainting = idx_inpainting.repeat(_t.shape[0], 1, 1, 1)
        variance[_idx_inpainting] += (1 - torch.exp(-10*_t*_t))[_idx_inpainting]
        return variance
    
    def signal_scale_derivative_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
        derivative = torch.zeros(t.shape[0], 1, 28, 28).to(device) - 1e-10
        _idx_inpainting = idx_inpainting.repeat(t.shape[0], 1, 1, 1)
        derivative[_idx_inpainting] += (-10*_t*torch.exp(-5*_t*_t))[_idx_inpainting]
        return derivative
    
    def noise_variance_derivative_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
        derivative = torch.zeros(t.shape[0], 1, 28, 28).to(device) + 1e-10
        _idx_inpainting = idx_inpainting.repeat(t.shape[0], 1, 1, 1)
        derivative[_idx_inpainting] += (20*_t*torch.exp(-10*_t*_t))[_idx_inpainting]
        return derivative


    diffusion_model = DiagonalDiffusionModel(signal_scale_func=signal_scale_func,
                                            noise_variance_func=noise_variance_func,
                                            signal_scale_derivative_func=signal_scale_derivative_func,
                                            noise_variance_derivative_func=noise_variance_derivative_func,
                                            score_estimator=score_estimator,
                                            sample_x_T_func=sample_x_T_func).to(device)

    if loadPreviousWeights:
        if os.path.exists('./data/weights/diffusion_diagonal.pt'):
            diffusion_model.load_state_dict(torch.load('./data/weights/diffusion_diagonal.pt'))
            print('Loaded weights from ./data/weights/diffusion_diagonal.pt')

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)

    for epoch in range(100):

        # run the training loop
        if runTraining:
            for i, (x_0_batch, _) in enumerate(mnist_train_loader):
                x_0_batch = x_0_batch.to(device)
                y_batch = x_0_batch.clone()
                y_batch[idx_inpainting.repeat(x_0_batch.shape[0], 1, 1, 1)] = 0.0
                loss = diffusion_model.compute_loss(x_0_batch, y_batch, loss_name='elbo')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose:
                    print(f'Epoch {epoch+1}, iteration {i+1}, loss {loss.item()}')
        
        # run the test loop
        if runTesting:
            for i, (x_0_batch, _) in enumerate(mnist_test_loader):
                x_0_batch = x_0_batch.to(device)
                y_batch = x_0_batch.clone()
                y_batch[idx_inpainting.repeat(x_0_batch.shape[0], 1, 1, 1)] = 0.0
                loss = diffusion_model.compute_loss(x_0_batch, y_batch, loss_name='elbo')
                if verbose:
                    print(f'Epoch {epoch}, iteration {i}, test loss {loss.item()}')
        
        
        if runReverseProcess:

            for i, (x_0_batch, _) in enumerate(mnist_test_loader):
                x_0_batch = x_0_batch.to(device)
                y_batch = x_0_batch.clone()
                y_batch[idx_inpainting.repeat(x_0_batch.shape[0], 1, 1, 1)] = 0.0
                break

            inds = torch.randperm(y_batch.shape[0])
            y_batch = y_batch[inds]

            # run the reverse process loop
            x_t_all = diffusion_model.reverse_process(
                            y=y_batch[0:4],
                            t_stop=0.0,
                            num_steps=1024,
                            returnFullProcess=True
                            )

            # plot the results as an animation
            from matplotlib.animation import FuncAnimation

            fig, ax = plt.subplots(2, 2)
            ims = []
            for i in range(4):
                im = ax[i//2, i%2].imshow(x_t_all[0,i, 0, :, :], animated=True)
                im.set_clim(-1, 1)
                ims.append([im])
            
            def updatefig(frame):
                print('Animating frame ', frame)
                for i in range(4):
                    if frame < 64:
                        ims[i][0].set_array(x_t_all[frame*16 + 15,i, 0, :, :])
                return [im[0] for im in ims]
            
            ani = FuncAnimation(fig, updatefig, frames=range(64), interval=50, blit=True)

            ani.save(f'./data/animations/diffusion_diagonal.mp4', fps=15)

        torch.save(diffusion_model.state_dict(), './data/weights/diffusion_diagonal.pt')


        