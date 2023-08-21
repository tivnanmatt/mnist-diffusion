# load the mnist dataset and train a diffusion model on it

import torch
import matplotlib.pyplot as plt
import os

from diffusion import ScalarDiffusionModel_VariancePreserving_LinearSchedule
from diffusion import UnconditionalScoreEstimator
from mnist_utils import UNet, TimeEncoder, PositionEncoder, BayesianClassifier

from mnist_utils import mnist_train_loader, mnist_test_loader

import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# --------------------------
# define parameters
# --------------------------
verbose=True
loadPreviousWeights=True
runTraining=False
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
        return torch.rand(batch_size, 1, 28, 28).to(device)

    diffusion_model = ScalarDiffusionModel_VariancePreserving_LinearSchedule(score_estimator=score_estimator,
                                                                             sample_x_T_func=sample_x_T_func).to(device)
    
    if loadPreviousWeights:
        if os.path.exists('./data/weights/diffusion_scalar.pt'):
            diffusion_model.load_state_dict(torch.load('./data/weights/diffusion_scalar.pt'))
            print('Loaded weights from ./data/weights/diffusion_scalar.pt')

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False, num_workers=2)

    time_encoder_classifier = TimeEncoder(out_channels=32).to(device)
    classifier = BayesianClassifier(time_encoder_classifier, input_channels=33, output_channels=10).to(device)

    # Load classifier weights if they exist
    classifier_weights_path = './data/weights/classifier.pt'
    if os.path.exists(classifier_weights_path):
        classifier.load_state_dict(torch.load(classifier_weights_path))
        print(f'Loaded classifier weights from {classifier_weights_path}')

    # Define other components as in the original script
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

    # Load diffusion model weights
    diffusion_weights_path = './data/weights/diffusion_scalar.pt'
    if os.path.exists(diffusion_weights_path):
        diffusion_model.load_state_dict(torch.load(diffusion_weights_path))
        print(f'Loaded diffusion model weights from {diffusion_weights_path}')

    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    for epoch in range(100):
        # Train the classifier
        if runTraining:
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                t_batch = torch.rand(x_batch.size(0)).to(device)
                x_t = diffusion_model.sample_x_t_given_x_0_and_t(x_0=x_batch, t=t_batch)
                optimizer_classifier.zero_grad()
                outputs = classifier(x_t,t_batch)
                # loss = torch.mean((outputs - torch.nn.functional.one_hot(y_batch, num_classes=10))**2)
                loss_fn = torch.nn.NLLLoss()
                loss = loss_fn(torch.log(outputs), y_batch)
                loss.backward()
                optimizer_classifier.step()
                print(f'Epoch {epoch+1}, iteration {i+1}, classifier loss {loss.item()}')

        
        # Rest of the reverse process code remains the same...

        # Save classifier weights
        torch.save(classifier.state_dict(), classifier_weights_path)

        if runReverseProcess:

            # Classifier-guided reverse process
            y_test = torch.tensor([i for i in range(10)]).view(-1).to(device)  # 10 samples for each y in [0, 1, ..., 9]

            def log_likelihood(x_t, y, t):
                classifier.eval()
                pdf_pred = classifier(x_t, t)
                classifier.train()
                return torch.sum(y * torch.log(pdf_pred), dim=1)
            
            x_t_all = diffusion_model.reverse_process(
                                    batch_size=10,  # Since we have 10 samples for each of 10 classes
                                    t_stop=0.0,
                                    num_steps=4096,
                                    returnFullProcess=True,
                                    y=torch.nn.functional.one_hot(y_test, num_classes=10),
                                    log_likelihood=log_likelihood,
                                    verbose=True
                                    )

            # Plot the results as an animation
            from matplotlib.animation import FuncAnimation

            fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # Adjusted figure size for a 2x5 subplot grid
            ims = []

            for i in range(10):
                ax = axes[i // 5, i % 5]
                im = ax.imshow(x_t_all[0, i, 0, :, :].cpu(), animated=True)
                im.set_clim(-1, 1)
                ims.append([im])
                
                # Setting ticks and labels
                ax.set_xticks([5, 10, 15, 20, 25])
                ax.set_yticks([5, 10, 15, 20, 25])
                if i % 2 == 1:  # If it's the bottom row
                    ax.set_xticklabels([5, 10, 15, 20, 25])
                else:
                    ax.set_xticklabels([])
                ax.set_title(f"p(x_t | y={i})")

            def updatefig(frame):
                print('Animating frame ', frame)
                for i in range(10):
                    if frame < 64:
                        ims[i][0].set_array(x_t_all[frame*64 + 63, i, 0, :, :].cpu())
                return [im[0] for im in ims]
            
            ani = FuncAnimation(fig, updatefig, frames=range(64), interval=50, blit=True)

            ani.save(f'./data/animations/diffusion_classifier_guided.mp4', fps=15)