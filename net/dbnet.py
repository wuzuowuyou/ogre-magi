import torch.nn as nn
from net.backbone import deformable_resnet50
from net.head import SegDetector
from net.loss.l1dice_loss import L1DiceLoss


class BasicModel(nn.Module):
    def __init__(self,):
        nn.Module.__init__(self)

        self.backbone = deformable_resnet50()
        self.decoder = SegDetector(in_channels=[256, 512, 1024, 2048], k=50, adaptive=True)

    def forward(self, data, *args, **kwargs):
        resout = self.backbone(data)
        segout = self.decoder(resout, *args, **kwargs)
        return segout


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)

class Db(nn.Module):
    def __init__(self, device, distributed: bool = False, local_rank: int = 0):
        super(Db, self).__init__()

        self.model = BasicModel()
        # for loading models
        # self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = L1DiceLoss()
        # self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data, training=self.training)

        if training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred

    def infer(self, input_tensor):
        pred = self.model(input_tensor, training=False)

        return pred