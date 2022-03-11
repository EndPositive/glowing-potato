import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import preprocessing
from datasets.chunked_watermarked_set import ChunkedWatermarkedSet, DataSetType
from models.swin_ir_multi import SwinIRMulti
from models.swin_wr_base import SwinWRBase
from swinir.models.network_swinir import SwinIR


EMBED_DIM = 180


class SwinWR(SwinWRBase):
    def __init__(
        self,
        image_size=(128, 128),
        inner_model: SwinIR = SwinIRMulti,
        train_last_layer_only=True,
        load_path=None,
        n_input_images=1,
    ):
        super().__init__(image_size)

        self._model = inner_model(
            upscale=1,
            in_chans=3,
            img_size=self.image_size[0],
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=EMBED_DIM,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
            n_input_images=n_input_images
        )

        if load_path is None:
            # download pretrained model
            state_dict = torch.utils.model_zoo.load_url(
                "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
                "005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
            )["params"]

            # remove the weights from the last layer (otherwise pytorch complains)
            del state_dict["conv_last.weight"]
            del state_dict["conv_last.bias"]

            self._model.load_state_dict(state_dict, strict=False)
        else:
            self.load(load_path)

        if train_last_layer_only:
            for param in self._model.parameters():
                param.requires_grad = False

            self._model.conv_last.weight.requires_grad = True
            self._model.conv_last.bias.requires_grad = True

        # define optimizer and loss func
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=0.001, weight_decay=0.0001
        )
        self._lossfn = nn.L1Loss()

        # define device to run on
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        print(f"Running model on {self._device}")

    def _decode_output(self, x):
        x = x.cpu().detach().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        return (x * 255).astype(np.uint8)

    def _encode_input(self, x):
        if len(x.shape) == 3:
            x = np.expand_dims(x, 0)
        x = np.transpose(x, (0, 3, 1, 2)) / 255  # convert to channels first
        return torch.from_numpy(x).float().to("cpu")

    def forward(self, x):
        return self._model(x)

    def save(self, path):
        torch.save(self._model.state_dict(), path)

    def load(self, path):
        self._model.load_state_dict(torch.load(path))

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

    def test(self, testset: DataLoader):
        with torch.no_grad():
            return np.mean(
                [
                    self._lossfn(self.forward_pass(x), y_hat).item()
                    for x, y_hat in iter(testset)
                ]
            )

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
                and np.mean(val_losses[-val_stop + 1 :]) > val_losses[-val_stop]
            ):
                print(
                    f"Validation loss hasn't improved in {val_stop} epochs. "
                    "Stopping training..."
                )
                break

            epoch += 1

        self.save(os.path.join(save_path, "final.pth"))
        return train_losses, val_losses

    def precompute_dataset(self, batch_size=64, output_to=preprocessing.OUTPUT_DIR_PRECOMPUTED):
        train_data_loader = DataLoader(
            ChunkedWatermarkedSet(
                data_set_type=DataSetType.Training, device=self._device, include_fn=True
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        for i, (x, _, fn) in enumerate(iter(train_data_loader)):
            features, residual = self._model.forward_feature_extraction(x)
            features = x.cpu().detach().numpy()
            residual = x.cpu().detach().numpy()

    def train_last_from(self, x, y_hat):
        for _ in range(1000):
            # zero the parameter gradients
            self._optimizer.zero_grad()

            # forward + backward + optimize
            y = self._model.forward_last(x[0], x[1])
            loss = self._lossfn(y, y_hat)
            loss.backward()
            self._optimizer.step()


if __name__ == "__main__":
    m = SwinWR()
    m.precompute_dataset()
