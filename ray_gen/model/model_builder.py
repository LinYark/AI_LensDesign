import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        self.model = 1

    def forward(self, data):
        vtemplate,vsearch,itemplate ,isearch= data['vz'].cuda(),data['vx'].cuda(),data['iz'].cuda(),data['ix'].cuda()
        label_cls ,label_loc= data['vcls'].cuda(),data['vloc'].cuda()

        # get feature
        vzf ,vxf = self.vbackbone(vtemplate),self.vbackbone(vsearch)
        izf ,ixf = self.ibackbone(itemplate),self.ibackbone(isearch)

        vzf ,vxf,izf ,ixf   = self.vneck(vzf ),self.vneck(vxf ),self.ineck(izf ),self.ineck(ixf )
        vzf,izf,vxf,ixf = self.enhance(vzf,izf,vxf,ixf)
        # (vzf,izf),(vxf,ixf) = self.commonwash(vzf,izf),self.commonwash(vxf,ixf)

        vcls = self.rgbhead(vzf, vxf)
        icls = self.thead(izf, ixf)
        vcls_l = self.log_softmax(vcls)
        icls_l = self.log_softmax(icls)
        vcls_loss = select_cross_entropy_loss(vcls_l, label_cls)
        icls_loss = select_cross_entropy_loss(icls_l, label_cls)
        vb,vc,vw,vh = vcls.size()
        ib,ic,iw,ih = icls.size()
        vclsmask  = vcls.view(vb,2,-1).permute(0,2,1).softmax(-1)[:,:, 1].view(vb,1,vw,vh)#是的，31*31
        iclsmask  = icls.view(ib,2,-1).permute(0,2,1).softmax(-1)[:,:, 1].view(ib,1,iw,ih)


        zf, xf = self.Zmix(vzf,izf),self.Xmixmask(vxf,ixf,vclsmask.detach(),iclsmask.detach())#.detach()
        cls, loc = self.head(zf, xf)
        cls = self.log_softmax(cls)

        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        outputs['total_loss'] = cls_loss + loc_loss + 0.2*vcls_loss +  0.2*icls_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['vcls_loss'] = vcls_loss
        outputs['icls_loss'] = icls_loss
        return outputs



if __name__ == "__main__":
    model = ModelBuilder()
    load_data = torch.load( './pretrain_models/model.pth'  )

    model_dict =  model.state_dict()
    state_dict = {}
    for k,v in load_data['state_dict'].items():
        if 'v'+k in model_dict.keys():
            state_dict['v'+k]=v
        if 'i'+k in model_dict.keys():
            state_dict['i'+k]=v
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.vbackbone.bn1.bias.data[0] +=1






