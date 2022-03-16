import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import preprocessing
from datasets.chunked_watermarked_set import ChunkedWatermarkedSet, DataSetType
from models.swin_ir_multi import SwinIRMulti
from models.swin_wr_base import SwinWRBase
from swinir.models.network_swinir import SwinIR
from datasets.transform import TRANSFORM_SWIN


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
            n_input_images=n_input_images,
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
        self._model.to(self.device)
        print(f"Running model on {self.device}")

        self._transforms = TRANSFORM_SWIN

    def precompute_dataset(
        self,
        batch_size=64,
        output_x=preprocessing.INPUT_DIR_PRECOMPUTED,
        output_y=preprocessing.OUTPUT_DIR_PRECOMPUTED,
    ):
        train_data_loader = DataLoader(
            ChunkedWatermarkedSet(
                data_set_type=DataSetType.Training, device=self.device, include_fn=True
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # make sure output directory exists
        os.makedirs(output_x, exist_ok=True)
        os.makedirs(output_y, exist_ok=True)

        for x, y, fn in tqdm(train_data_loader):
            features, residual = self._model.forward_feature_extraction(x)
            features = features.cpu().detach().numpy()
            residual = residual.cpu().detach().numpy()

            for features_curr, residual_curr, fn_curr in zip(features, residual, fn):
                # write expected output data to file
                np.save(
                    os.path.join(
                        output_y, os.path.basename(fn_curr).split(".")[0] + ".npy"
                    ),
                    y,
                )

                # write features to file
                np.save(
                    os.path.join(
                        output_x,
                        os.path.basename(fn_curr).split(".")[0] + "_features.npy",
                    ),
                    features_curr,
                )

                # write residual to file
                np.save(
                    os.path.join(
                        output_x,
                        os.path.basename(fn_curr).split(".")[0] + "_residual.npy",
                    ),
                    residual_curr,
                )

    def forward_last(self, x):
        return self._model.forward_last(*x)


if __name__ == "__main__":
    m = SwinWR()
    m.train(batch_size=1)
    # m.train(from_precomputed_set=True, batch_size=1)
    # m.precompute_dataset(batch_size=10)
