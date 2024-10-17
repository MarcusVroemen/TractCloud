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
    def __init__(self, num_classes, args, device):
        super(tract_DGCNN_cls, self).__init__()
        self.args = args
        self.fiber_level_k = args.k
        self.fiber_level_k_global = args.k_global 
        self.k_point_level = args.k_point_level
        self.device = device
        self.num_out_classes = num_classes
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
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, self.num_out_classes)

    def forward(self, x, info_point_set):
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
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        x = F.log_softmax(x, dim=1)

        return x

# class tract_DGCNN_cls(nn.Module):
#     def __init__(self, num_classes, args, device):
#         super(tract_DGCNN_cls, self).__init__()
#         self.args = args
#         self.fiber_level_k = args.k
#         self.fiber_level_k_global = args.k_global 
#         self.k_point_level = args.k_point_level
#         self.device = device
#         self.num_out_classes = num_classes
#         # self.marking = args.mark_endpoints
        
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm1d(args.emb_dims)

#         self.conv1 = nn.Sequential(nn.Conv2d((3)*2, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout(p=args.dropout)
#         self.linear2 = nn.Linear(512, 256)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.dp2 = nn.Dropout(p=args.dropout)
#         self.linear3 = nn.Linear(256, self.num_out_classes)

#     def create_fixed_attention_map(self, num_fibers, num_points):
#         # Fixed attention: Higher weights for the first and last points
#         attention_weights = torch.ones(num_fibers, 1, num_points).to(self.device)
#         attention_weights[:, :, 0] = 2  # Start point
#         attention_weights[:, :, -1] = 2  # End point
#         return attention_weights

#     def forward(self, x, info_point_set):
#         """x: (num_fiber, 3, num_points)
#            info_point_set: local+global feature (num_fiber, 4, num_points, fiber_level_k)
#         """
#         num_fiber = x.size(0)
#         num_points = x.size(2)

#         # Apply fixed attention map
#         # attention_map = self.create_fixed_attention_map(num_fiber, num_points)
        
#         # Use attention map to reweight the input features
#         # if self.marking:
#         #     x = x * attention_map

#         if self.fiber_level_k + self.fiber_level_k_global == 0:
#             # Only use neighbor points for each streamline (process each streamline individually), no local+global info
#             x = get_tract_graph_feature(x, k_point_level=self.k_point_level, device=self.device)      # (num_fiber, 4, num_points) -> (num_fiber, 8, num_points, k)

#         else:
#             # Input local+global info for each streamline
#             x = x[:, :, :, None].repeat(1, 1, 1, self.fiber_level_k + self.fiber_level_k_global)  # (num_fiber, 4, num_points) -> (num_fiber, 4, num_points, fiber_k)
#             x = torch.cat((info_point_set - x, x), dim=1)  # (num_fiber, 8, num_points, fiber_k)

#         x = self.conv1(x)  # (num_fiber, 8, num_points, fiber_k) -> (num_fiber, 64, num_points, fiber_k)
#         x1 = x.max(dim=-1, keepdim=False)[0]  # (num_fiber, 64, num_points, fiber_k) -> (num_fiber, 64, num_points)

#         x = get_tract_graph_feature(x1, k_point_level=self.k_point_level, device=self.device)  # (num_fiber, 64, num_points) -> (num_fiber, 128, num_points, k)
#         x = self.conv2(x)  # (num_fiber, 128, num_points, k) -> (num_fiber, 64, num_points, k)
#         x2 = x.max(dim=-1, keepdim=False)[0]  # (num_fiber, 64, num_points, k) -> (num_fiber, 64, num_points)

#         x = get_tract_graph_feature(x2, k_point_level=self.k_point_level, device=self.device)  # (num_fiber, 64, num_points) -> (num_fiber, 128, num_points, k)
#         x = self.conv3(x)  # (num_fiber, 128, num_points, k) -> (num_fiber, 128, num_points, k)
#         x3 = x.max(dim=-1, keepdim=False)[0]  # (num_fiber, 128, num_points, k) -> (num_fiber, 128, num_points)

#         x = get_tract_graph_feature(x3, k_point_level=self.k_point_level, device=self.device)  # (num_fiber, 128, num_points) -> (num_fiber, 256, num_points, k)
#         x = self.conv4(x)  # (num_fiber, 256, num_points, k) -> (num_fiber, 256, num_points, k)
#         x4 = x.max(dim=-1, keepdim=False)[0]  # (num_fiber, 256, num_points, k) -> (num_fiber, 256, num_points)

#         x = torch.cat((x1, x2, x3, x4), dim=1)  # (num_fiber, 64+64+128+256, num_points)

#         x = self.conv5(x)  # (num_fiber, 512, num_points) -> (num_fiber, emb_dims, num_points)
#         x1 = F.adaptive_max_pool1d(x, 1).view(num_fiber, -1)  # (num_fiber, emb_dims, num_points) -> (num_fiber, emb_dims)
#         x2 = F.adaptive_avg_pool1d(x, 1).view(num_fiber, -1)  # (num_fiber, emb_dims, num_points) -> (num_fiber, emb_dims)
#         x = torch.cat((x1, x2), 1)  # (num_fiber, emb_dims*2)

#         x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (num_fiber, emb_dims*2) -> (num_fiber, 512)
#         x = self.dp1(x)
#         x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (num_fiber, 512) -> (num_fiber, 256)
#         x = self.dp2(x)
#         x = self.linear3(x)  # (num_fiber, 256) -> (num_fiber, output_channels)
#         x = F.log_softmax(x, dim=1)

#         return x

import math

def add_endpoint_marker(x):
    # """Add sophisticated positional encoding as an additional feature."""
    # num_fibers, num_dims, num_points = x.size()
    
    # # Add endpoint marker: 1 for start and end, 0 for others
    # endpoint_marker = torch.zeros(num_fibers, 1, num_points, device=x.device)
    # endpoint_marker[:, :, 0] = 1  # Start point
    # endpoint_marker[:, :, -1] = 1  # End point
    
    # # Concatenate positional encoding and endpoint marker to the original features
    # x = torch.cat((x, endpoint_marker), dim=1)  # New shape: [N_fiber, num_dims + num_dims + 1, N_point]
    
    """Add normalized positional encoding (0 to 1) to each point."""
    
    num_fibers, num_dims, num_points = x.size()

    # Normalized positional encoding: 0 for the first point, ..., 1 for the last
    position_marker = torch.linspace(0, 1, steps=num_points, device=x.device)  # Create [0, ..., 1]
    position_marker = position_marker.unsqueeze(0).unsqueeze(0).repeat(num_fibers, 1, 1)  # Expand to match x: [N_fiber, 1, N_point]
    
    # Concatenate position encoding with the original features
    x = torch.cat((x, position_marker), dim=1)  # New shape: [N_fiber, num_dims + 1, N_point]
    # import pdb
    # pdb.set_trace()
    return x
