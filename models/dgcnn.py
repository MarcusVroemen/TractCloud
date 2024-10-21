"""
Reference from https://github.com/antao97/dgcnn.pytorch

Modified by 
@Author: Tengfei Xue
@Contact: txue4133@uni.sydney.edu.au
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def tract_knn(x, k):
    """
        input 
            x: [N_fiber, num_dims, N_point]
            k: number of points forming graph for each streamline (point-level neighbors)
        output 
            idx: [N_fiber, N_point, k]
    """
    
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)   # minus distance. 

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (N_fiber, N_point, k). Since pairwise_distance is minus here, the larger value, the smaller distance.
    
    return idx

def get_tract_graph_feature(x, k_point_level=15, device=torch.device('cuda')):
    """"Get graph feature for points on a single streamline.
        input x:[N_fiber, num_dims, N_point]
        k: number of points to be selected for every point location
        idx: graph indices
        device: the device to run the code    
    """     
    num_fibers = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)

    if k_point_level >= num_points:
        idx = torch.arange(0, num_points)[None,None,:].repeat(num_fibers, num_points, 1).to(device)  # all points are connected to each other. max=Npoint-1
    else:    
        idx = tract_knn(x, k=k_point_level)   # (N_fiber, N_point, k). max=N_point-1

    idx_base = torch.arange(0, num_fibers, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)    # (N_fiber*N_point*k), max=N_fiber*N_point-1

    x = x.transpose(2, 1).contiguous()   # (num_fibers, num_dims, num_points)  -> (num_fibers, num_points, num_dims)
    # assign the idx when the dim of 'idx' is larger than num_fibers*num_points, but the max 'idx' is num_fibers*num_points-1
    feature = x.view(num_fibers*num_points, -1)[idx, :]   # (num_fibers, num_points, num_dims)->(num_fibers*num_points, num_dims)->(num_fibers*num_points*k, num_dims)
    feature = feature.view(num_fibers, num_points, k_point_level, num_dims) 
    x = x.view(num_fibers, num_points, 1, num_dims).repeat(1, 1, k_point_level, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature     # (num_fiber, 2*num_dims, num_points, k)

class tract_DGCNN_cls(nn.Module):
    def __init__(self, num_classes_0, args, device, num_classes_1=None):
        super(tract_DGCNN_cls, self).__init__()
        self.args = args
        self.fiber_level_k = args.k
        self.fiber_level_k_global = args.k_global 
        self.k_point_level = args.k_point_level
        self.device = device
        self.num_out_classes_0 = num_classes_0
        self.num_out_classes_1 = num_classes_1
        self.depth = args.depth  # Added depth parameter to control number of layers
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Conv2d(3*2, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))

        # Additional convolutional layers based on depth
        self.extra_convs = nn.ModuleList()
        for _ in range(self.depth - 4):  # Add additional conv layers beyond the 4 already defined
            conv = nn.Sequential(
                nn.Conv2d(256*2, 256, kernel_size=1, bias=False),  # Same architecture as previous layers
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.extra_convs.append(conv)

        # Fully connected layers
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        
        self.linear2_0 = nn.Linear(512, 256)
        self.bn7_0 = nn.BatchNorm1d(256)
        self.dp2_0 = nn.Dropout(p=args.dropout)
        self.linear3_0 = nn.Linear(256, self.num_out_classes_0)
        if  self.num_out_classes_1!=None:
            self.linear2_1 = nn.Linear(512, 256)
            self.bn7_1 = nn.BatchNorm1d(256)
            self.dp2_1 = nn.Dropout(p=args.dropout)
            self.linear3_1 = nn.Linear(256, self.num_out_classes_1)

    def forward(self, x, info_point_set, task_id=0):
        num_fiber = x.size(0)
        num_points = x.size(2)

        if self.fiber_level_k + self.fiber_level_k_global == 0:
            x = get_tract_graph_feature(x, k_point_level=self.k_point_level, device=self.device)
        else:
            x = x[:, :, :, None].repeat(1, 1, 1, self.fiber_level_k + self.fiber_level_k_global)
            x = torch.cat((info_point_set - x, x), dim=1)

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_tract_graph_feature(x1, k_point_level=self.k_point_level, device=self.device)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_tract_graph_feature(x2, k_point_level=self.k_point_level, device=self.device)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_tract_graph_feature(x3, k_point_level=self.k_point_level, device=self.device)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # Apply additional convolutional layers if depth > 4
        for conv in self.extra_convs:
            x = get_tract_graph_feature(x4, k_point_level=self.k_point_level, device=self.device)
            x = conv(x)
            x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(num_fiber, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(num_fiber, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7_0(self.linear2_0(x)), negative_slope=0.2)
        x = self.dp2_0(x)
        if task_id==0:
            x = self.linear3_0(x)
        elif task_id==1:
            # x = F.leaky_relu(self.bn7_1(self.linear2_1(x)), negative_slope=0.2)
            # x = self.dp2_1(x)
            x = self.linear3_1(x)
        x = F.log_softmax(x, dim=1)

        return x