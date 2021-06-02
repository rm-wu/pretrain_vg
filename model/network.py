import torch
import logging
import torchvision
from torch import nn

import model.pooling as pooling
# from model.normalization import L2Norm
from model.metric_learning import ArcMarginProduct # TODO: use this in the main.py file?


class LandmarkNet(nn.Module):
    def __init__(self,
                 args,
                 num_classes,
                 # TODO: TOREMOVE
                 # pooling='GeM',
                 # args_pooling: dict={},
                 # use_fc=False,
                 #  fc_dim=512,
                 # dropout=0.0,
                 # loss_module='arcface',
                 # s=30.0,
                 # margin=0.50,
                 # ls_eps=0.0
                 ):
        super(LandmarkNet, self).__init__()
        self.encoder = get_encoder(args)
        self.pooling = get_pooling(args)

        self.use_fc = args.fc_output_dimension is not None  # TODO: add an ad-hoc argument ?
        if self.use_fc:
            self.fc_layer = nn.Sequential(
                nn.Dropout(p=args.fc_dropout),
                nn.Linear(args.features_dim, args.fc_output_dimension),
                nn.BatchNorm1d(args.fc_output_dimension))
            self._init_params()

        self.loss_module = args.loss_module
        if self.loss_module == 'arcface':
            self.final_layer = ArcMarginProduct(args.fc_output_dimension, num_classes,
                                                device=args.device,
                                                s=args.arcface_s,
                                                m=args.arcface_margin,
                                                easy_margin=False,
                                                ls_eps=args.arcface_ls_eps)  # label smoothing
        # TODO: add cosface and other loss_modules?
        # elif loss_module == 'cosface':
        #     self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        # elif loss_module == 'adacos':
        #     self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final_layer = nn.Linear(args.fc_output_dimension, num_classes)

    def _init_params(self):
        """
        Initialization of the FC layer following smlyaka implementation
        """
        nn.init.xavier_normal_(self.fc_layer[1].weight)
        nn.init.constant_(self.fc_layer[1].bias, 0)
        nn.init.constant_(self.fc_layer[2].weight, 1)
        nn.init.constant_(self.fc_layer[2].bias, 0)

    def forward(self, x, labels):
        x = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final_layer(x, labels)
        else:
            logits = self.final(x)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = self.pooling(x).view(batch_size, -1)    # This could be avoided using nn.Sequential w/ Flatten
        if self.use_fc:
            x = self.fc_layer(x)
        return x


# TODO: TOREMOVE se non viene utilizzato nel pooling layer
class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


def get_pooling(args):
    """
    Returns the chosen pooling layer
    TODO: lasciare le normalizzazioni utilizzate in benchmarking_vg???
    """
    if args.pooling == "gem":
        return pooling.GeM()
    elif args.pooling == "spoc":
        return pooling.SPoC()
    elif args.pooling == "mac":
        return pooling.MAC()
    elif args.pooling == "rmac":
        return pooling.RMAC()

    # TODO: TOREMOVE ???
    # if args.pooling == "gem":
    #     return nn.Sequential(L2Norm(), # TOREMOVE ??
    #                          pooling.GeM(),
    #                          Flatten())
    # elif args.pooling == "spoc":
    #     return nn.Sequential(L2Norm(), pooling.SPoC(), Flatten())
    # elif args.pooling == "mac":
    #     return nn.Sequential(L2Norm(), pooling.MAC(), Flatten())
    # elif args.pooling == "rmac":
    #     return nn.Sequential(L2Norm(), pooling.RMAC(), Flatten())


def get_encoder(args):
    """
    Function that returns the selected model as a feature extractor (i.e. w/o final pooling and linear layers)
    """
    if args.arch.startswith("r"):  # It's a ResNet
        if args.arch.startswith("r18"):
            encoder = torchvision.models.resnet18(pretrained=True)
        elif args.arch.startswith("r50"):
            encoder = torchvision.models.resnet50(pretrained=True)
        elif args.arch.startswith("r101"):
            encoder = torchvision.models.resnet101(pretrained=True)
    elif args.arch == "vgg16":
        encoder = torchvision.models.vgg16(pretrained=True)
        # layers = list(encoder.features.children())[:-2]
    # Drop original final pooling/linear layers
    encoder = nn.Sequential(*(list(encoder.children())[:-2]))
    # Dynamically obtain number of channels in output
    args.features_dim = get_output_channels_dim(encoder)
    return encoder


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]
