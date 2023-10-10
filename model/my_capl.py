import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import model.resnet as models
import random

manual_seed=321
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)


# Masked Average Pooling
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)    


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, \
        zoom_factor=8, num_sp = 1,criterion=nn.CrossEntropyLoss(ignore_index=255), \
        BatchNorm=nn.BatchNorm2d, pretrained=True, args=None):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.classes = classes
        self.num_sp = num_sp
        self.train_iter = args.train_iter
        self.eval_iter = args.eval_iter
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
        fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 512, kernel_size=1)
        )

        main_dim = 512
        self.main_proto = nn.Parameter(torch.randn(self.classes, num_sp, main_dim).cuda())
        gamma_dim = 1
        self.gamma_conv = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, gamma_dim)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, self.classes, kernel_size=1)
        )

        self.args = args

    def forward(self, x, y=None, s_init_seed = None, gened_proto=None, base_num=16, novel_num=5, iter=None, \
                gen_proto=False, eval_model=False, visualize=False):

        self.iter = iter
        self.base_num = base_num

        def WG(x, y, proto, target_cls, s_seed):
            """
            x: (1,c,h,w)
            y: (1,h,w)
            s_init_seed: (num_sp, 2)
            """
            b, c, h, w = x.size()[:]
            tmp_y = F.interpolate(y.float().unsqueeze(1), size=(h, w), mode='nearest') 
            out = x.clone()
            tmp_y = (tmp_y==target_cls).float()
            unique_y = list(tmp_y.unique())  
            assert len(unique_y)==2       
            new_gen_proto = proto.data.clone()
            # add codes to generate the prototypes
            ## code starts here
            ########################### Adaptive Superpixel Clustering ###########################
            num_sp, _ = s_seed.size()  # bs x shot x max_num_sp x 2
            sp_center_list = []
            with torch.no_grad():
                supp_feat_ = x[0]                                 # c x h x w
                supp_mask_ = tmp_y[0]                             # 1 x h x w
                s_seed_ = s_seed                                  # max_num_sp x 2
                num_sp = max(len(torch.nonzero(s_seed_[:, 0])), len(torch.nonzero(s_seed_[:, 1])))
                # if num_sp == 0 or 1, use the Masked Average Pooling instead
                if (num_sp == 0) or (num_sp == 1):
                    supp_proto = Weighted_GAP(supp_feat_.unsqueeze(0), supp_mask_.unsqueeze(0))  # 1 x c x 1 x 1
                    sp_center_list.append(supp_proto.squeeze().unsqueeze(-1))                    # c x 1
                else:
                    s_seed_ = s_seed_[:num_sp, :]  # num_sp x 2
                    sp_init_center = supp_feat_[:, s_seed_[:, 0], s_seed_[:, 1]]  # c x num_sp (sp_seed)
                    sp_init_center = torch.cat([sp_init_center, s_seed_.transpose(1, 0).float()], dim=0)  # (c + xy) x num_sp

                    if self.training:
                        sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center, n_iter=self.train_iter)
                        sp_center_list.append(sp_center)
                    else:
                        sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center, n_iter=self.eval_iter)
                        sp_center_list.append(sp_center)

            sp_center = torch.cat(sp_center_list, dim=1)   # c x num_sp_all (collected from all shots)
            assert sp_center.size()== torch.Size((c, num_sp))
            ## code ends here 
            return sp_center.t() # (num_sp, c)        

        def generate_fake_proto(proto, x, y, s_init_seed):
            """
            proto: (cls_num, num_sp, c)
            x: (b,c,h,w)
            y: (b,h,w) or (b,1,h,w) (need to be checked)
            s_init_seed: (b, cls_num, num_sp, 2) 
            """
            b, c, h, w = x.size()[:]
            assert y.size() == torch.Size([b,h,w])
            tmp_y = F.interpolate(y.float().unsqueeze(1), size=(h,w), mode='nearest')
            unique_y = list(tmp_y.unique())
            raw_unique_y = list(tmp_y.unique())
            if 0 in unique_y:
                unique_y.remove(0)
            if 255 in unique_y:
                unique_y.remove(255)

            novel_num = len(unique_y) // 2
            fake_novel = random.sample(unique_y, novel_num)

            for fn in fake_novel:
                unique_y.remove(fn) 

            # fake base classes === fake_context
            fake_context = unique_y
            
            new_proto = self.main_proto.clone()  # cls_num, num_sp, c
            new_proto = new_proto / (torch.norm(new_proto, 2, -1, True) + 1e-12)
            x = x / (torch.norm(x, 2, 1, True) + 1e-12)

            for fn in fake_novel:
                # generate prototypes for fake novel class fn


                # rule to update or replace the prototypes for fake novel class
                pass

            for fc in fake_context:
                # generate prototypes for fake context or base classes

                # rule to update the memory bank
                pass

            if random.random() > 0.5 and 0 in raw_unique_y:
                # generate prototypes for background class=0

                #rule to update the memory bank
                pass

            return new_proto, replace_proto

        if gen_proto:
            # proto generation
            # supp_x: [cls, s, c, h, w]
            # supp_y: [cls, s, h, w]
            self.training = False
            x = x[0]
            y = y[0]        
            s_init_seed = s_init_seed[0]    
            cls_num = x.size(0)
            shot_num = x.size(1)
            """
            x: (5,1,3,257,257)
            y: (5,1,257,257)
            s_init_seed: (5,21,num_sp,2)

            return: gened_prototype -> (cls_num, num_sp, c)
            """
            with torch.no_grad():
                gened_proto = self.main_proto.clone()
                tmp_x_feat_list = []
                tmp_gened_proto_list = []
                for idx in range(cls_num):
                    tmp_x = x[idx] 
                    tmp_y = y[idx]
                    raw_tmp_y = tmp_y
                    # look here again
                    init_seed = s_init_seed[idx]  # num_classes, num_sp, 2
                    # look above again
                    tmp_x = self.layer0(tmp_x)
                    tmp_x = self.layer1(tmp_x)
                    tmp_x = self.layer2(tmp_x)
                    tmp_x = self.layer3(tmp_x)
                    tmp_x = self.layer4(tmp_x)
                    layer4_x = tmp_x
                    tmp_x = self.ppm(tmp_x)
                    ppm_feat = tmp_x.clone()
                    tmp_x = self.cls(tmp_x) 
                    tmp_x_feat_list.append(tmp_x)

                    tmp_cls = idx + base_num
                    init_seed = init_seed[tmp_cls] # num_sp, 2

                    tmp_gened_proto = WG(x=tmp_x, y=tmp_y, proto=self.main_proto, target_cls=tmp_cls, s_seed=init_seed) # (num_sp, c)
                    tmp_gened_proto_list.append(tmp_gened_proto)

                    gened_proto[tmp_cls, :] = tmp_gened_proto

                # crosscheck/modify the  below lines
                ## update the memory for base classes
                ## code starts here        
                gened_proto = gened_proto / (torch.norm(gened_proto, 2, 1, True) + 1e-12) # (cls_num, num_sp, c)
                ## code ends here
            return gened_proto  


        else:
            x_size = x.size()
            assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
            h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
            w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)            
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x_tmp = self.layer3(x)
            x = self.layer4(x_tmp)
            x = self.ppm(x)
            x = self.cls(x)
            raw_x = x.clone()              

        
            if eval_model: 
                #### evaluation
                if len(gened_proto.size()[:]) == 4:
                    gened_proto = gened_proto[0]   
                if visualize:
                    vis_feat = x.clone()    
                # change the prediction function
                ## code starts here
                x = self.get_pred(raw_x, self.main_proto)
                ## code ends here
    
            else:
                ##### training
                self.training = True
                fake_num = x.size(0) // 2              
                ori_new_proto, replace_proto = generate_fake_proto(proto=self.main_proto, x=x[fake_num:], y=y[fake_num:], s_init_seed = s_init_seed[fake_num:])                    
                # modify the prediction rule
                ## code starts here
                x = self.get_pred(x, ori_new_proto)  

                ## code ends here                                          

            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            if self.training:
                ## add all loss functions and auxillary losses
                ## code starts here
                main_loss = self.criterion(x, y)
                aux_loss = 0
                ## code ends here
                return x.max(1)[1], main_loss, aux_loss
            else:
                if visualize:
                    return x, vis_feat
                else:
                    return x
    
    def cosine_similarity(self, x, protos):
        """
        x : has shape (b, c, h, w)
        protos: has shape (num_classes, num_sp, c)
        """
        assert len(protos.shape[:]) == 3
        b, c, h, w = x.size()[:]
        cls_num = protos.size(0)
        num_sp = protos.size(1)
        x = x / torch.norm(x, 2, 1, True) # l2 norm on dimension 1, with keepdims = True
        proto = proto / torch.norm(proto, 2, -1, True)
        x = x.contiguous().view(b, c, h*w)  # b, c, hw
        protos = protos.contiguous().view(1, cls_num*num_sp, c)  # 1, num_classes*num_sp, c
        sim = proto @ x  # b, cls*num_sp, hw
        sim = sim.contiguous().view(b, cls_num, num_sp, h, w)
        sim = sim.sum(dim=2)
        assert sim.size() == torch.Size([b, cls_num, h, w])
        return sim
    
    def feature_voting(self, weights, protos):
        """
        weights: has shape (b, cls_num, h, w)
        protos: has shape (num_classes, num_sp, c)
        """
        voted_features = []
        b, cls_num, h, w = weights.size()[:]
        cls_num, num_sp, c = protos.size()[:]
        weights = weights.permute((0, 2, 3, 1))
        weights = weights.contiguous().view(b,h*w, cls_num)   # b, h*w, cls_num
        for n in range(self.num_sp):
            d_n = protos[:,n,:]  # cls_num, c
            rv_n = weights @ d_n  # b, h*w, c
            assert rv_n.size() == torch.Size([b, h*w, c])
            rv_n = rv_n.contiguous().view(b, h, w, c).permute((0,-1, 1, 2))
            voted_features.append(rv_n)
        rv = torch.cat(voted_features, dim=1)
        assert rv.size() == torch.Size([b, self.num_sp*c, h, w])
        return rv
    
    def memory_update():
        pass

    def get_pred():
        pass


