import torch
from torch import nn

from swinir.models.network_swinir import SwinIR


class SwinIRMulti(SwinIR):
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1,
        upsampler="",
        resi_connection="1conv",
        n_input_images=1,
        **kwargs,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            ape,
            patch_norm,
            use_checkpoint,
            upscale,
            img_range,
            upsampler,
            resi_connection,
            **kwargs,
        )

        num_out_ch = in_chans
        self.n_input_images = n_input_images
        self.conv1 = nn.Conv2d(embed_dim * n_input_images, embed_dim, 1, 1)
        self.conv2 = nn.Conv2d(embed_dim, 128, 1, 1)
        self.conv3 = nn.Conv2d(128, 64, 3, 1, padding=1)
        self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, padding=1)

    def forward_feature_extraction(self, x):
        # x has shape (batch_size * n_input_images, 3, H, W)
        H, W = x.shape[2:]
        batch_size = x.shape[0] // self.n_input_images

        # same as SwinIR
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first

        # (batch_size * n_input_images, embed_dim, H, W) ->
        # (batch_size,  n_input_images, embed_dim, H, W)
        res = torch.reshape(
            res, (batch_size, self.n_input_images, self.embed_dim, H, W)
        )

        # transform to (batch_size, n_input_images * embed_dim, H, W)
        return torch.reshape(
            res, (batch_size, self.n_input_images * self.embed_dim, H, W)
        ), x

    def forward_last(self, x, residual):
        # 1x1 -> 1x1 -> 3x3 conv
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # our layer
        x = residual + self.conv_last(x)

        # batch normalization
        return x / self.img_range + self.mean

    def forward(self, x):
        return self.forward_last(*self.forward_feature_extraction(x))
