from pathlib import Path
import os, time

import numpy as np
from torch.optim import Optimizer
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn
from datasets.chunked_watermarked_set import DataSetType, ChunkedWatermarkedSet
from datasets.swin_precomputed_set import SwinPrecomputedSet
from typing import Callable
from tqdm import tqdm
from torchsummary import summary
from pytorch_msssim import MS_SSIM

from preprocessing import formatter as processing


class WRBase(nn.Module):
    _lossfn: nn.Module
    _optimizer: Optimizer
    _model: nn.Module
    _transforms: Callable

    def __init__(self, input_size=(128, 128)):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size if type(input_size) == tuple else (input_size, input_size)

    def summary(self):
        summary(self._model, (3, *self.input_size))

    def train_epoch(
        self, dataloader: DataLoader, epoch=0, log_every=200, from_precomputed_set=False
    ):
        self.train()
        epoch_loss = 0
        running_loss = 0
        # batch_time = AverageMeter('Time', ':6.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        # progress = ProgressMeter(
        #     len(dataloader),
        #     [batch_time, data_time],
        #     prefix="\nEpoch: [{}]".format(epoch)
        # )
        end = time.time()
        for i, (x, y_hat) in tqdm(enumerate(iter(dataloader)), total=len(dataloader)):
            # zero the parameter gradients
            # data_time.update(time.time() - end)
            self._optimizer.zero_grad()
            if x[0][0][0][0].item() == -1: 
                print(f'Out of memory, restarting')
                torch.cuda.empty_cache()
                time.sleep(5)
                continue
            # x, y_hat = x, y_hat

            # forward + backward + optimize
            y = self.forward_last(x) if from_precomputed_set else self(x)
            loss = self._lossfn(y, y_hat)
            loss.backward()
            self._optimizer.step()

            # add loss to return
            epoch_loss += loss.item()

            # batch_time.update(time.time() - end)
            # end = time.time()
            # print statistics
            # if log_every > 0:
            #     if (i + 1) % log_every == 0:
            #         progress.display(i)
                # running_loss += loss.item()
                #     print(
                #         f"[{epoch + 1}, {i + 1:5d}] "
                #         f"loss: {running_loss / log_every:.3f}"
                #     )
                #     running_loss = 0

        return epoch_loss / len(dataloader)

    def train(self, mode: bool = True):
        self._model.train(mode)

    def eval(self):
        self._model.eval()

    def train_model(
        self,
        n_epochs=-1,
        val_stop=5,
        save_path=".",
        save_every=-1,
        data_shuffle=True,
        data_num_workers=0,
        batch_size=16,
        from_precomputed_set=False,
        validate=False,
        data_set=None
    ):
        if data_set is None:
            data_set = (
                SwinPrecomputedSet(data_set_type=DataSetType.Training, device=self.device)
                if from_precomputed_set
                else ChunkedWatermarkedSet(
                    data_set_type=DataSetType.Training, device=self.device, transforms=self._transforms
                )
            )

        # load train and validation sets
        train_data_loader = DataLoader(
            data_set.train(),
            batch_size=batch_size,
            shuffle=data_shuffle,
            num_workers=data_num_workers,
        )

        # validation_data_loader = DataLoader(
        #     data_set,
        #     batch_size=batch_size,
        #     shuffle=data_shuffle,
        #     num_workers=data_num_workers,
        # )

        # make sure save path exists
        os.makedirs(save_path, exist_ok=True)

        epoch = 0
        train_losses = []
        val_losses = []
        while n_epochs < 0 or epoch < n_epochs:
            # train the model one epoch
            try:
                train_loss = self.train_epoch(
                    train_data_loader, epoch, from_precomputed_set=from_precomputed_set
                )
            except KeyboardInterrupt:
                break

            # test on the validation set
            val_loss = self.test(
                validation_data_loader, from_precomputed_set=from_precomputed_set
            ) if validate else None

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
            # if (
            #     0 < val_stop <= len(val_losses)
            #     and np.mean(val_losses[-val_stop + 1 :]) > val_losses[-val_stop]
            # ):
            #     print(
            #         f"Validation loss hasn't improved in {val_stop} epochs. "
            #         "Stopping training..."
            #     )
            #     break

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

    def predict(self, img: Image, max_batch_size=8) -> Image:
        self.eval()
        batch = self._transforms.get_prediction_batch(img)
        batches = torch.split(batch, max_batch_size)
        with torch.no_grad():
            pred = torch.cat([self(x.to(self.device)) for x in tqdm(batches)], 0)
        return self._transforms.image_from_prediction(pred, img)

    def save(self, path):
        torch.save(self._model.state_dict(), path)

    def load(self, path):
        self._model.load_state_dict(torch.load(path, map_location=self.device))

    def test(self, testset: DataLoader, msssim=False, from_precomputed_set=False):
        self.eval()
        msssim_fn = MS_SSIM(data_range=1)
        losses, msssim_losses = [], []
        with torch.no_grad():
            for x, y_hat in tqdm(iter(testset)):
                y = self.forward_last(x) if from_precomputed_set else self(x)
                losses.append(self._lossfn(y, y_hat).item())
                if msssim:
                    msssim_losses.append(msssim_fn(y, y_hat))

        return np.mean(losses), np.mean(msssim_losses) if msssim else np.mean(losses)

    def validate_from_checkpoints(self, path, data_set=None, max_batch_size=16):
        if data_set is None:
            data_set = ChunkedWatermarkedSet(
                device=self.device,
                transforms=self._transforms,
                split_size=(0.9, 0.001,)
            ).validate()

        data_loader = DataLoader(
            data_set,
            batch_size=max_batch_size,
            shuffle=False,
            num_workers=0,
        )

        ckpt_dir = Path(path)
        ckpt_files = list(ckpt_dir.glob("ckpt_*.pth"))

        print(f'Found {len(ckpt_files)} model files.')

        losses = {}
        for i, ckpt_path in enumerate(ckpt_files):
            # get epoch number
            epoch = int(ckpt_path.name[5:].split('.')[0])

            # load the model
            self.load(ckpt_path.as_posix())

            # get loss
            loss = self.test(data_loader)

            # save loss into dictionary
            losses[epoch] = loss

            print(
                f'Finished validating epoch {epoch}. ({i + 1} / {len(ckpt_files)})\n'
                f'Loss: {loss}\n'
                f'-----------------------------------------------------------------\n'
            )

        return list(losses.keys()), list(losses.values())

    def forward(self, x):
        return self._model(x)

    def forward_last(self, x):
        return self.forward(x)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)