
from cement import Controller, ex

import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from .dataset_rad_energy import Dataset
from tensorboardX import SummaryWriter
from nf4ip.ext.vae.mlpvae import MLPVAE, Reversible, Config
from nf4ip.ext.inn.models.inn_model import InnModel
from pathlib import Path
from nf4ip.ext.vae.LWFA_trainer_vae import LWFA_Trainer
from nf4ip.core.config import handle_config

class LWFA(Controller):

    class Meta:
        label = 'model'
        stacked_on = 'base'
        stacked_type = 'nested'

    def _default(self):
        self._parser.print_help()

    @ex(
        help='train the lwfa model'
    )
    def train(self):

        # Dataset
        norm = ['log', 'max', 'drop 0']
        angle = 14
        train_set = Dataset(angle=angle, norm=norm, balance=0, data_sets=[2, 3])
        val_set = Dataset(angle=angle, norm=norm, balance=0, data_sets=[1])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.app.config.get('nf4ip', 'batch_size'), shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=5, shuffle=False, drop_last=True)

        def swap(x, y, **kwargs):
            return y, x

        self.app.filter.register('train_input', swap, 100)
        self.app.filter.register('val_input', swap, 100)

        def handleOutput(output, **kwargs):
            return torch.nn.Sigmoid()(output)

        self.app.filter.register('train_forward_output', handleOutput)
        self.app.filter.register('train_backward_output', handleOutput)
        self.app.filter.register('train_backward_rand_output', handleOutput)
        self.app.filter.register('val_backward_output', handleOutput)

        self.app.hook.register('post_validate', self.validationImage)

        self.inn = InnModel(self.app, self.app.device, train_loader, val_loader)

        self.inn.train()

        exp = 0
        # Save model
        model_name_path = Path('inn_best_trial_5e-6_2_3_{}_base_test'.format(exp))
        torch.save(self.inn.model.state_dict(), model_name_path)
        
        """
        hParams = {'ndim_y': ndim_y,
                   'ndim_x': ndim_x,
                   'ndim_z': ndim_z,
                   'ndim_tot': ndim_tot,
                   'num_blocks': num_blocks,
                   'batch_size': batch_size,
                   'feature': feature,
                   'lr': lr,
                   'epochs': epochs,
                   'loss': loss}

        metric = {'Loss': loss}

        self.log_writer.add_hparams(hParams, metric)
        """

    def printEpoch(self, i_epoch, loss, **kwargs):
        self.app.log.info(str(i_epoch) + " " + str(loss), )

    def printValidation(self, loss, **kwargs):
        """"""
        self.app.log.info('validation loss: {}'.format(loss.data.item()), )

    def validationImage(self, model, i_epoch, n_epochs, x_samps, y_samps, **kwargs):
        # Compute current output for Validation data x <- y,z
        if hasattr(self.app, 'tensorboard') and i_epoch % 30 == 0 or i_epoch == n_epochs:
            y_samps_t = y_samps
            y = {}
            stack = np.zeros(y_samps.shape[0])
            # Perform x <- y,z with sample_rate diffrent z
            for i in range(0, 1000):
                y_samps = y_samps_t
                y_samps = torch.cat([torch.randn(x_samps.shape[0], self.inn.ndim_z).to(self.app.device),
                                     self.inn.zeros_noise_scale * torch.zeros(x_samps.shape[0],
                                                                              self.inn.ndim_total - self.inn.ndim_y - self.inn.ndim_z).to(
                                         self.app.device),
                                     y_samps], dim=1)
                y_samps = y_samps.to(self.app.device)
                rev_x = model.model(y_samps, rev=True)
                rev_x = torch.nn.Sigmoid()(rev_x)
                rev_x = rev_x.detach().cpu().numpy()
                y[i] = rev_x
                stack = np.vstack((stack, rev_x[:, 0]))
            stack = stack[1:]

            # Use the standart deviation to get the uncertainty
            std = []
            for i in range(stack.shape[1]):
                std.append(np.std(stack[:, i]))

            # Mean the outcome to get the most proable x
            out_mean = np.mean(stack, axis=0)

            # Plot current output
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.errorbar(np.arange(out_mean.shape[0]), out_mean, yerr=std, label='Net output', fmt='.', c='orange',
                        elinewidth=0.5)

            gt = x_samps.detach().cpu().numpy()
            ax.scatter(list(range(0, len(gt))), gt, label='Ground truth', s=6)
            # ax.vlines(230,0,1)
            plt.xlabel('Timesteps')
            plt.ylabel('Injection certainty')
            plt.title('Invertible network output on validation data')
            plt.legend()

            self.app.tensorboard.log_writer.add_figure(str(i_epoch), fig, i_epoch)
            plt.close(fig)

    @ex(
        help='train the vae model'
    )
    def train_vae(self):
        torch.manual_seed(1234)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Dataset
        norm = ['log', 'max', 'drop 0']
        angle = 14
        train_set = Dataset(angle=angle, norm=norm, balance=0, data_sets=[2, 3])
        val_set = Dataset(angle=angle, norm=norm, balance=0, data_sets=[1])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.app.config.get('nf4ip', 'batch_size'), shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=5, shuffle=False, drop_last=True)

        # VAE Network
        mlp_config = {
            'hidden_size': 100,
            'num_hidden': 8,
            'activation': torch.relu
        }
        config = Config(28, 1, 512)
        size = 34
        layers = 6
        lay = []
        for i in range(0, layers):
            lay.append(Reversible("conv", 3, ind=size, outd=size, pad=1))

        cnn_layers = [
            Reversible("conv", 3, ind=1, outd=size, input_layer=True),
            *lay,
            Reversible("pool", 3, ind=size, outd=size),
            Reversible("conv", 3, ind=size, outd=size),
            Reversible("pool", 3, ind=size, outd=size),
            Reversible("conv", 3, ind=size, outd=size),
            Reversible("pool", 3, ind=size, outd=size),
            Reversible("conv", 3, ind=size, outd=size)
        ]
        base_vae = MLPVAE(config, mlp_config, cnn_layers).float().to(self.app.device)
        trainer = LWFA_Trainer(base_vae, train_loader, val_loader)

        epochs = 15000
        lr = 5e-6
        gamma = 10
        beta = 0.005194530884589891
        l = trainer.train_VAE(lr, epochs, beta, gamma, send=False)

        #trainer.train_VAE(0.0005, 200, 1, 1)
        #trainer.train_MLP(0.001, 300)
