# load the mnist dataset and train a diffusion model on it

import torch
import matplotlib.pyplot as plt
import os

from diffusion import FourierDiffusionModel
from mnist_utils import UnconditionalScoreEstimator, UNet, TimeEncoder, PositionEncoder, mnist_train_loader, mnist_test_loader

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
    denoiser = UNet(in_channels=65, out_channels=1, num_base_filters=32).to(device)
    
    score_estimator = UnconditionalScoreEstimator(denoiser, time_encoder, position_encoder).to(device)

    # now do it with torch
    def gaussian_blur_fourier_transfer_function(fwhm, size):
        """Generate a Gaussian blur transfer function with a given FWHM and size."""
        sigma = fwhm / (2.0 * torch.sqrt(2.0 * torch.log(torch.tensor(2.0))))
        xGrid = torch.linspace(-size // 2, size // 2, steps=size).to(device)
        yGrid = torch.linspace(-size // 2, size // 2, steps=size).to(device)
        xGrid, yGrid = torch.meshgrid(xGrid, yGrid)
        rGrid = torch.sqrt(xGrid**2 + yGrid**2)
        y = torch.exp(-rGrid**2 / (2 * sigma**2))
        y /= y.sum()  # Normalize
        y = torch.fft.fft2(y)
        y = torch.abs(y)
        return y
    
    fourier_transfer_function_LPF = gaussian_blur_fourier_transfer_function(6.0, 28)
    fourier_transfer_function_BPF = gaussian_blur_fourier_transfer_function(3.0, 28) - fourier_transfer_function_LPF
    fourier_transfer_function_HPF = torch.ones(28, 28).to(device) - fourier_transfer_function_BPF - fourier_transfer_function_LPF
    
    fourier_transfer_function_LPF = fourier_transfer_function_LPF.unsqueeze(0).unsqueeze(0)
    fourier_transfer_function_BPF = fourier_transfer_function_BPF.unsqueeze(0).unsqueeze(0)
    fourier_transfer_function_HPF = fourier_transfer_function_HPF.unsqueeze(0).unsqueeze(0)

    def modulation_transfer_function_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) * torch.exp(-5*_t*_t)
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) * torch.exp(-7*_t*_t)
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) * torch.exp(-9*_t*_t)
        return LPF + BPF + HPF

    def modulation_transfer_function_derivative_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) * (-10*_t * torch.exp(-5*_t*_t))
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) * (-14*_t * torch.exp(-7*_t*_t))
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) * (-18*_t * torch.exp(-9*_t*_t))
        return LPF + BPF + HPF + 1e-10

    def noise_power_spectrum_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) * (1.0 - torch.exp(-10*_t*_t))
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) * (1.0 - torch.exp(-14*_t*_t))
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) * (1.0 - torch.exp(-18*_t*_t))
        return LPF + BPF + HPF + 1e-10

    def noise_power_spectrum_derivative_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) *  (20*_t * torch.exp(-10*_t*_t))
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) *  (28*_t * torch.exp(-14*_t*_t))
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) *  (36*_t * torch.exp(-18*_t*_t))
        return LPF + BPF + HPF + 1e-10


    # Create the FourierDiffusionModel using the functions we've defined
    diffusion_model = FourierDiffusionModel(
        score_estimator=score_estimator,
        modulation_transfer_function_func=modulation_transfer_function_func,
        noise_power_spectrum_func=noise_power_spectrum_func,
        modulation_transfer_function_derivative_func=modulation_transfer_function_derivative_func,
        noise_power_spectrum_derivative_func=noise_power_spectrum_derivative_func
    )

    if loadPreviousWeights:
        if os.path.exists('./data/weights/diffusion_fourier_unconditional.pt'):
            diffusion_model.load_state_dict(torch.load('./data/weights/diffusion_fourier_unconditional.pt'))
            print('Loaded weights from ./data/weights/diffusion_fourier_unconditional.pt')

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)

    for epoch in range(100):

        # run the training loop
        if runTraining:
            for i, (x_0_batch, _) in enumerate(mnist_train_loader):
                x_0_batch = x_0_batch.to(device)
                loss = diffusion_model.compute_loss(x_0_batch, loss_name='elbo')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose:
                    print(f'Epoch {epoch+1}, iteration {i+1}, loss {loss.item()}')

        # run the test loop
        if runTesting:
            for i, (x_0_batch, _) in enumerate(mnist_test_loader):
                x_0_batch = x_0_batch.to(device)
                diffusion_model.eval()
                with torch.no_grad():
                    loss = diffusion_model.compute_loss(x_0_batch, loss_name='elbo')
                diffusion_model.train()
                if verbose:
                    print(f'Epoch {epoch}, iteration {i}, test loss {loss.item()}')
        
        
        if runReverseProcess:

            # for i, (x_0_batch, _) in enumerate(mnist_test_loader):
            #     x_0_batch = x_0_batch.to(device)
            #     break

            # inds = torch.randperm(x_0_batch.shape[0])
            # x_0_batch = x_0_batch[inds]

            # x_T = diffusion_model.sample_x_t_given_x_0_and_t(x_0_batch, t=torch.ones(x_0_batch.shape[0]).to(device))

            x_T = torch.randn_like((4,1,28,28)).to(device)

            # run the reverse process loop
            x_t_all = diffusion_model.reverse_process(
                            x_T=x_T[0:4],
                            t_stop=0.0,
                            num_steps=1024,
                            returnFullProcess=True,
                            verbose=True
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

            ani.save(f'./data/animations/diffusion_fourier_unconditional.mp4', fps=15)

        torch.save(diffusion_model.state_dict(), './data/weights/diffusion_fourier_unconditional.pt')