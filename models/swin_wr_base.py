from pathlib import Path
import os

import numpy as np
from torch.optim import Optimizer
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn
from datasets.chunked_watermarked_set import ChunkedWatermarkedSet, DataSetType

from preprocessing import formatter as processing


class SwinWRBase(nn.Module):
    _lossfn: nn.Module
    _optimizer: Optimizer
    _model: nn.Module

    def __init__(self, image_size=(128, 128)):
        super().__init__()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

    def train_epoch(self, dataloader: DataLoader, epoch=0, log_every=-1):
        epoch_loss = 0
        running_loss = 0
        for i, (x, y_hat) in enumerate(iter(dataloader)):
            # zero the parameter gradients
            self._optimizer.zero_grad()

            # forward + backward + optimize
            y = self(x)
            loss = self._lossfn(y, y_hat)
            loss.backward()
            self._optimizer.step()

            # add loss to return
            epoch_loss += loss.item()

            # print statistics
            if log_every > 0:
                running_loss += loss.item()
                if (i + 1) % log_every == 0:
                    print(
                        f"[{epoch + 1}, {i + 1:5d}] "
                        "loss: {running_loss / log_every:.3f}"
                    )
                    running_loss = 0

        return epoch_loss / len(dataloader)

    def train(
        self,
        n_epochs=-1,
        val_stop=5,
        save_path=".",
        save_every=-1,
        data_shuffle=True,
        data_num_workers=0,
        batch_size=16,
    ):
        # load train and validation sets
        train_data_loader = DataLoader(
            ChunkedWatermarkedSet(
                data_set_type=DataSetType.Training, device=self._device
            ),
            batch_size=batch_size,
            shuffle=data_shuffle,
            num_workers=data_num_workers,
        )

        validation_data_loader = DataLoader(
            ChunkedWatermarkedSet(
                data_set_type=DataSetType.Validation, device=self._device
            ),
            batch_size=batch_size,
            shuffle=data_shuffle,
            num_workers=data_num_workers,
        )

        # make sure save path exists
        os.makedirs(save_path, exist_ok=True)

        epoch = 0
        train_losses = []
        val_losses = []
        while n_epochs < 0 or epoch < n_epochs:
            # train the model one epoch
            train_loss = self.train_epoch(train_data_loader, epoch)

            # test on the validation set
            val_loss = self.test(validation_data_loader)

            # print epoch summary
            print(f"Epoch {epoch} summary:")
            print(f"Train loss: {train_loss}")
            print(f"Validation loss: {val_loss}\n")

            # log losses
            val_losses.append(val_loss)
            train_losses.append(train_loss)

            # save model if necessary
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self.save(os.path.join(save_path, f"ckpt_{epoch}.pth"))

            # if validation loss hasn't improved in val_loss epochs, stop training
            if (
                0 < val_stop <= len(val_losses)
                and np.mean(val_losses[-val_stop + 1:]) > val_losses[-val_stop]
            ):
                print(
                    f"Validation loss hasn't improved in {val_stop} epochs. "
                    "Stopping training..."
                )
                break

            epoch += 1

        self.save(os.path.join(save_path, "final.pth"))
        return train_losses, val_losses

    def _decode_output(self, x):
        x = x.cpu().detach().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        return (x * 255).astype(np.uint8)

    def _encode_input(self, x):
        if len(x.shape) == 3:
            x = np.expand_dims(x, 0)
        x = np.transpose(x, (0, 3, 1, 2)) / 255  # convert to channels first
        return torch.from_numpy(x).float().to("cpu")

    def predict(self, img):
        if isinstance(img, str) or isinstance(img, Path):
            img = Image.open(img)
            img = np.asarray(img)

        # shape should be image or batch of images
        assert len(np.shape(img)) == 3 or len(np.shape(img)) == 4

        # predict multiple images
        if len(np.shape(img)) == 4:
            return [self.predict(x) for x in img]

        # pad to align to img size
        padded, padding = processing.pad(img, self.image_size, return_padding=True)

        # split image in multiple blocks, if too big
        split = processing.split_image(padded, self.image_size)

        # normalize and get input in tensor form
        encoded_input = self._encode_input(split)

        # run the network and decode the output (get numpy from tensor and un-normalize)
        output = self._decode_output(self.forward_pass(encoded_input))

        # piece image back together
        unsplit = processing.unsplit_image(output, padded.shape[0])

        # unpad and return
        return processing.unpad(unsplit, padding)

    def save(self, path):
        torch.save(self._model.state_dict(), path)

    def load(self, path):
        self._model.load_state_dict(torch.load(path))

    def test(self, testset: DataLoader):
        with torch.no_grad():
            return np.mean(
                [
                    self._lossfn(self(x), y_hat).item()
                    for x, y_hat in iter(testset)
                ]
            )

    def forward(self, x):
        return self._model(x)
