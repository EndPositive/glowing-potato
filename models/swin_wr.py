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
        self._model.to(self._device)
        print(f"Running model on {self._device}")

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
    m.train()
