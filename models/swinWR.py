import sys
from pathlib import Path
from turtle import forward
sys.path.append(Path(__file__).parents[1].absolute().as_posix())
import os

import numpy as np
from PIL import Image

from swinir.models.network_swinir import SwinIR
from wrmodel import WRmodel
from train_set import ChunkedWatermarkedSet
import torch
from torch import nn
from torch import optim


class SwinWR(WRmodel):
    def __init__(self, image_size=(128, 128), train_last_layer_only=True, load_path=None):
        super().__init__(image_size)
        self._model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=self.image_size[0],
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )

        if load_path is None:
            self._model.load_state_dict(
                torch.utils.model_zoo.load_url(
                    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
                )["params"]
            )

        if train_last_layer_only:
            for _, param in self._model.named_parameters():
                param.requires_grad = False

            self._model.conv_last.weight.requires_grad = True
            self._model.conv_last.bias.requires_grad = True

        # define optimizer and loss func
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001, weight_decay=0.0001)
        self._lossfn = nn.L1Loss()
    
        # define device to run on
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._devicde)
        print('Running model on ' + self._device)


    def _decode_output(self, x):
        x = x.cpu().detach().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        return (x * 255).astype(np.uint8)

    def _encode_input(self, x):
        if len(x.shape) == 3:
            x = np.expand_dims(x, 0)
        x = np.transpose(x, (0, 3, 1, 2)) / 255  # convert to channels first
        return torch.from_numpy(x).float().to("cpu")

    def __call__(self, img):
        return self.predict(img)

    def forward_pass(self, x):
        return self._model(x)
    
    def save(self, path):
        torch.save(self._model.state_dict(), path)

    def load(self, path):
        self._model.load_state_dict(
            torch.load(path)
        )

    def train_epoch(self, trainset: ChunkedWatermarkedSet, epoch=0, verbose=True):
        epoch_loss = 0
        for i, (x, y_hat) in enumerate(trainset):
            # zero the parameter gradients
            self._optimizer.zero_grad()

            # forward + backward + optimize
            y = self.forward_pass(x)
            loss = self._lossfn(y, y_hat)
            loss.backward()
            self._optimizer.step()

            # add loss to return
            epoch_loss += loss.item()

            # print statistics
            if verbose:
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        
        return epoch_loss / len(trainset)

    def test(self, testset: ChunkedWatermarkedSet):
        with torch.no_grad():
            return np.mean(
                self._lossfn(self.forward_pass(x), y_hat).item() for x, y_hat in testset
            )

    def train(self, n_epochs=-1, val_stop=5, save_path='.', save_every=-1):
        # load train and validtion sets
        train_set = ChunkedWatermarkedSet()
        validation_set = ChunkedWatermarkedSet() # important todo: change this with the validation set

        epoch = 0
        train_losses = []
        val_losses = []
        while n_epochs < 0 or epoch < n_epochs:
            # train the model one epoch
            train_loss = self.train_epoch(train_set, epoch)

            # test on the validation set
            val_loss = self.test(validation_set)

            # print epoch summary
            print(f'Epoch {epoch} summary:\nTrain loss: {train_loss}\nValidation loss: {val_loss}\n')

            # log losses
            val_losses.append(val_loss)
            train_losses.append(train_loss)

            # save model if necessary
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self.save(os.path.join(save_path, f'ckpt_{epoch}.pth'))

            # if validation loss hasn't improved in val_loss epochs, stop training
            if val_stop > 0 and len(val_losses) >= val_stop and \
                np.mean(val_losses[-val_stop + 1:]) > val_losses[-val_stop]:
                print(f'Validation loss hasn\'t improved in {val_stop} epochs. Stopping training...')
                break

            epoch += 1
        
        return train_losses, val_losses



if __name__ == "__main__":
    with torch.no_grad():
        model = SwinWR()
        Image.fromarray(model("../resources/dataset/input/0c3ee986fa326b1a_7.jpg")).show()
