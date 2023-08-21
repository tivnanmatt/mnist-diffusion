# load the mnist dataset and train a diffusion model on it

import torch
import matplotlib.pyplot as plt
import os

from diffusion import ScalarDiffusionModel_VariancePreserving_LinearSchedule
from diffusion import UnconditionalScoreEstimator
from mnist_utils import UNet, TimeEncoder, PositionEncoder, mnist_train_loader, mnist_test_loader

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

    class ScoreEstimator(UnconditionalScoreEstimator):
        def __init__(self, denoiser, time_encoder, position_encoder):
            super().__init__()
            self.denoiser = denoiser
            self.time_encoder = time_encoder
            self.position_encoder = position_encoder

        def forward(self, x_t, t):
            t_enc = self.time_encoder(t.unsqueeze(1))
            pos_enc = self.position_encoder().repeat(x_t.shape[0], 1, 1, 1)
            denoiser_input = torch.cat((x_t, t_enc, pos_enc), dim=1)
            return self.denoiser(denoiser_input)
        
    score_estimator = ScoreEstimator(denoiser, time_encoder, position_encoder).to(device)
    
    def sample_x_T_func(batch_size=None, y=None):
        if batch_size is None:
            raise ValueError('batch_size must be provided')
        return torch.randn(batch_size, 1, 28, 28).to(device)

    diffusion_model = ScalarDiffusionModel_VariancePreserving_LinearSchedule(score_estimator=score_estimator,
                                                                             sample_x_T_func=sample_x_T_func).to(device)


    if loadPreviousWeights:
        if os.path.exists('./data/weights/diffusion_scalar.pt'):
            diffusion_model.load_state_dict(torch.load('./data/weights/diffusion_scalar.pt'))
            print('Loaded weights from ./data/weights/diffusion_scalar.pt')

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
                loss = diffusion_model.compute_loss(x_0_batch, loss_name='elbo')
                if verbose:
                    print(f'Epoch {epoch}, iteration {i}, test loss {loss.item()}')
        
        if runReverseProcess:

            # run the reverse process loop
            x_t_all = diffusion_model.reverse_process(
                            batch_size=4,
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

            ani.save(f'./data/animations/diffusion_scalar.mp4', fps=15)

        torch.save(diffusion_model.state_dict(), './data/weights/diffusion_scalar.pt')


        