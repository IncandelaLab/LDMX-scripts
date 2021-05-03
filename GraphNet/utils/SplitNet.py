from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from utils.ParticleNet import *

torch.set_default_dtype(torch.float32)

class SplitNet(nn.Module):
    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128,0.1)],
                 use_fusion=False,
                 return_softmax=False,
                 **kwargs):
        super(SplitNet, self).__init__(**kwargs)
        print("INITIALIZING SPLITNET")

        self.nRegions = 1

        # Particle nets:
        if self.nRegions == 1:
            self.eNet = ParticleNet(input_dims=input_dims, num_classes=2, conv_params=conv_params, \
                                    fc_params=fc_params, use_fusion=use_fusion, return_softmax = return_softmax)
        else:
            self.particleNets = []
            for i in range(self.nRegions):
                self.particleNets.append(ParticleNet(input_dims=input_dims, num_classes=2, conv_params=conv_params, \
                                                     fc_params=fc_params, use_fusion=use_fusion, return_softmax = return_softmax))
        print("INITIALIZED PARTICLENET")
        
        #self.pNet = ParticleNet(input_dims=input_dims, num_classes=2, conv_params=conv_params, \
        #                        fc_params=fc_params, use_fusion=use_fusion, return_softmax = return_softmax)
        # NEW
        #self.oNet = ParticleNet(input_dims=input_dims, num_classes=2, conv_params=conv_params, \
        #                        fc_params=fc_params, use_fusion=use_fusion, return_softmax = return_softmax)
        
        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)

        # Fully connected layer:
        # NEW:  Modified PN to return x instead of output, moved FC layer here instead (and resized).
        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            fcs.append(nn.Sequential(nn.Linear(self.nRegions*in_chn, self.nRegions*channels), Mish(), nn.Dropout(drop_rate)))
        fcs.append(nn.Linear(self.nRegions*fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.return_softmax = return_softmax

        print("FINISHED INIT")

    def forward(self, points, features):
        # Divide up provided points+features, then hand them to the PNs
        # Points are [nregions] x 128  x 3 x 50 (note: nregions axis is gone for 1 region)
        # Note:  points[:,0].shape = (128, 3, 50)
        if self.nRegions == 1:
            x_e = self.eNet(points, features)
            output = self.fc(x_e)
        else:
            xi = []
            for i in range(self.nRegions):
                xi.append(self.particleNets[i](points[i], features[i]))
            output = self.fc(torch.cat(xi, dim=1))
        #x_e = self.eNet(points[:,0], features[:,0])
        #x_p = self.pNet(points[:,1], features[:,1])
        #x_o = self.oNet(points[:,2], features[:,2])
        #output = self.fc(x_e)
        #output = self.fc(torch.cat((x_e, x_p), dim=1))  #, x_o), dim=1))
        if self.return_softmax:
            output = torch.softmax(output, dim=1)
        return output
