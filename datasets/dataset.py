from __future__ import print_function
import time
import torch.utils.data as data
import torch
import numpy as np
import pickle
import sys
import os
import whitematteranalysis as wma
from pytorch3d.transforms import RotateAxisAngle, Scale, Translate
import pdb

sys.path.append('../')
import utils.tract_feat as tract_feat
from utils.funcs import obtain_TractClusterMapping, cluster2tract_label,\
    get_rot_axi, array2vtkPolyData, makepath
from utils.fiber_distance import MDF_distance_calculation, MDF_distance_calculation_endpoints   

class RealData_PatchData(data.Dataset):
    def __init__(self, feat, k, k_global, cal_equiv_dist=False, use_endpoints_dist=False, 
                 rough_num_fiber_each_iter=10000, k_ds_rate=0.1):
        self.feat = feat.astype(np.float32)   # [n_fiber, n_point, n_feat]
        self.k = k
        self.k_global = k_global
        self.cal_equiv_dist = cal_equiv_dist
        self.use_endpoints_dist = use_endpoints_dist
        self.rough_num_fiber_each_iter = rough_num_fiber_each_iter  # this will be a rough targeted number of fibers in each iteration.
        self.k_ds_rate=k_ds_rate
        
        num_fiber = self.feat.shape[0]
        num_point = self.feat.shape[1]
        num_feat_per_point = self.feat.shape[2]
        
        if self.k_global==0:
            self.global_feat = np.zeros((1, num_point, num_feat_per_point, 1), dtype=np.float32) # [1, n_point, n_feat, 1]
        else:
            random_idx = np.random.randint(0, num_fiber, self.k_global)
            self.global_feat = self.feat[random_idx,...]  # [k_global, n_point, n_feat]. This is the global (random) feature for all fibers in a test subject
            self.global_feat = self.global_feat.transpose(1,2,0)[None,:,:,:].astype(np.float32)  # [1, n_point, n_feat, k_global] 
        
        if self.k==0:
            self.local_feat = np.zeros((num_fiber, num_point, num_feat_per_point, 1), dtype=np.float32) # [n_fiber, n_point, n_feat, 1]
        else:
            self.local_feat = np.zeros((num_fiber, num_point, num_feat_per_point, self.k), dtype=np.float32)  # [n_fiber, k, n_point, n_feat]
            num_iter = num_fiber // self.rough_num_fiber_each_iter
            self.num_fiber_each_iter = (num_fiber // num_iter) + 1
            for i_iter in range(num_iter):
                cur_feat = self.feat[i_iter*self.num_fiber_each_iter:(i_iter+1)*self.num_fiber_each_iter,...]  # [n_fiber, n_point, n_feat]
                cur_feat = np.transpose(cur_feat,(0,2,1))  #  [n_fiber, n_point, n_feat]->[n_fiber,n_feat,n_point]
                cur_local_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k, self.use_endpoints_dist, self.cal_equiv_dist)      # [n_fiber*k, n_feat, n_point]
                cur_local_feat = cur_local_feat.reshape(cur_feat.shape[0], self.k, num_feat_per_point, num_point)  # [n_fiber, k, n_feat, n_point]
                cur_local_feat = np.transpose(cur_local_feat,(0,3,2,1))  # [n_fiber, k, n_feat, n_point]->[n_fiber, n_point, n_feat, k]
                self.local_feat[i_iter*self.num_fiber_each_iter:(i_iter+1)*self.num_fiber_each_iter,...] = cur_local_feat
            
            
    def __getitem__(self, index):
        point_set = self.feat[index]    # [n_point, n_feat]
        klocal_point_set = self.local_feat[index]   # [n_point, n_feat, k]

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
            klocal_point_set = torch.from_numpy(klocal_point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            klocal_point_set = torch.from_numpy(klocal_point_set.astype(np.float32))
            print('Feature is not in float32 format')

        return point_set, klocal_point_set

    def __len__(self):
        return self.feat.shape[0]


class unrelatedHCP_PatchData(data.Dataset):
    def __init__(self, root, out_path, logger, split='train', num_fiber_per_brain=10000,num_point_per_fiber=15, 
                 use_tracts_training=False, k=0, k_global=0, rot_ang_lst=[0,0,0], scale_ratio_range=[0,0], trans_dis=0.0,
                 aug_axis_lst=['LR','AP', 'SI'], aug_times=10, cal_equiv_dist=False, k_ds_rate=0.1, recenter=False, include_org_data=False):        
        self.root = root
        self.out_path = out_path
        self.split = split
        self.logger = logger
        self.num_fiber = num_fiber_per_brain
        self.num_point = num_point_per_fiber
        self.use_tracts_training = use_tracts_training
        self.k = k
        self.k_global = k_global
        self.rot_ang_lst = rot_ang_lst
        self.scale_ratio_range = scale_ratio_range
        self.trans_dis = trans_dis
        self.aug_axis_lst = aug_axis_lst
        self.aug_times = aug_times
        self.k_ds_rate=k_ds_rate  
        self.recenter = recenter
        self.include_org_data = include_org_data
        
        # data save for debugging
        self.save_aug_data = True
        
        # algorithm tests
        self.cal_equiv_dist = cal_equiv_dist
        self.use_endpoints_dist = False
        self.logger.info('cal_equiv_dist: {}, use_endpoints_dist: {}'
                    .format(self.cal_equiv_dist, self.use_endpoints_dist))
                
        # load data
        with open(os.path.join(root, '{}.pickle'.format(split)), 'rb') as file:
            # Load the data from the file
            data_dict = pickle.load(file)
        self.features = data_dict['feat']
        self.labels = data_dict['label']
        # pdb.set_trace()
        #! self.label_names = data_dict['label_name']
        self.subject_ids = data_dict['subject_id']
        self.logger.info('Load {} data'.format(self.split))
        
        # calculate brain-level features  [n_subject, n_fiber, n_point, n_feat], labels [n_subject, n_fiber, n_point or 1]
        self.brain_features, self.brain_labels = self._cal_brain_feat()
        
        # calculate local global features/representations [n_subject*n_fiber, n_point, n_feat], [n_subject*n_fiber, n_point or 1], [n_subject*n_fiber, n_point, n_feat, k]
        # [n_subject*n_fiber, n_point, n_feat], [n_subject*n_fiber, n_point or 1], [n_subject*n_fiber, n_point, n_feat, k], [n_subject, n_point, n_feat, k_global], [n_subject*n_fiber, 1]
        self.org_feat, self.org_label, self.local_feat, self.global_feat, self.new_subidx = self._cal_info_feat()
        

    def __getitem__(self, index):
        point_set = self.org_feat[index]
        label = self.org_label[index]
        klocal_point_set = self.local_feat[index]
        new_subidx = self.new_subidx[index]

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
            klocal_point_set = torch.from_numpy(klocal_point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            klocal_point_set = torch.from_numpy(klocal_point_set.astype(np.float32))
            print('Feature is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(label)
            new_subidx = torch.from_numpy(new_subidx)
        else:
            label = torch.from_numpy(label.astype(np.int64))
            new_subidx = torch.from_numpy(new_subidx.astype(np.int64))
            print('Label is not in int64 format')

        return point_set, label, klocal_point_set, new_subidx

    def __len__(self):
        return self.org_feat.shape[0]


    def _cal_brain_feat(self, chunk_size=5):
        """Process data for classification and segmentation in chunks to reduce memory usage."""
        """Process data for classification and segmentation. Get data in both brain and streamline level.
           Brain features are used for calculating the local-global representation.

        Args (self):
            features (array):[n_fiber, n_point, n_feat] original feature in fiber-level 
            subject_ids (array): [n_fiber] subject id for streamlines
            num_fiber_per_brain (int): number of fiber per brain
            num_point_per_fiber (int): number of point per fiber
            use_tracts_training (bool): whether to use tract label for training
        Returns:
            brain_feature (array):[n_subject, n_fiber, n_point, n_feat] feature in brain-level 
            brain_label (array): [n_subject, n_fiber, n_point or 1] label in brain-level
        """ 
        num_feat_per_point = self.features.shape[2]
        unique_subject_ids = np.unique(self.subject_ids)
        num_subject = len(unique_subject_ids)

        # Initialize lists to store chunk results
        all_brain_features = []
        all_brain_labels = []
        
        for chunk_start in range(0, num_subject, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_subject)
            current_chunk_size = chunk_end - chunk_start

            if self.aug_times > 0:  # augmented data
                brain_features = np.zeros((current_chunk_size*self.aug_times, self.num_fiber, self.num_point, num_feat_per_point),dtype=np.float32)
                brain_labels = np.zeros((current_chunk_size*self.aug_times, self.num_fiber,1), dtype=np.int64)
                aug_matrices = np.zeros((current_chunk_size, self.aug_times, 4, 4), dtype=np.float32)
            else:  # non-augmented data
                brain_features = np.zeros((current_chunk_size, self.num_fiber, self.num_point, num_feat_per_point),dtype=np.float32)
                brain_labels = np.zeros((current_chunk_size, self.num_fiber,1), dtype=np.int64)
            
            for i, unique_id in enumerate(unique_subject_ids[chunk_start:chunk_end]):
                # pdb.set_trace()
                cur_idxs = np.where(self.subject_ids == unique_id)[0]
                np.random.shuffle(cur_idxs)
                cur_select_idxs = cur_idxs[:self.num_fiber]
                cur_features = self.features[cur_select_idxs,:,:]
                cur_labels = self.labels[cur_select_idxs, None]

                if self.aug_times > 0:
                    cur_features = torch.from_numpy(cur_features)
                    aug_features = np.zeros((self.aug_times, *cur_features.shape))

                    for i_aug in range(self.aug_times):
                        trot = None
                        cur_angles = []

                        # rotations, scales, translations (same as in the original function)
                        for i, rot_ang in enumerate(self.rot_ang_lst):
                            angle = ((torch.rand(1) - 0.5) * 2 * rot_ang).item()
                            rot_axis_name = get_rot_axi(self.aug_axis_lst[i])
                            cur_trot = RotateAxisAngle(angle=angle, axis=rot_axis_name, degrees=True)
                            cur_angles.append(round(angle, 1))
                            trot = trot.compose(cur_trot) if trot else cur_trot

                        scale_r = torch.distributions.Uniform(1 - self.scale_ratio_range[0], 1 + self.scale_ratio_range[1]).sample().item() if self.scale_ratio_range[0] or self.scale_ratio_range[1] else 1.0
                        trot = trot.compose(Scale(scale_r))

                        LR_trans = ((torch.rand(1) - 0.5) * 2 * self.trans_dis).item()
                        AP_trans = ((torch.rand(1) - 0.5) * 2 * self.trans_dis).item()
                        SI_trans = ((torch.rand(1) - 0.5) * 2 * self.trans_dis).item()
                        trot = trot.compose(Translate(LR_trans, AP_trans, SI_trans))

                        aug_matrices[i, i_aug, :, :] = np.array(trot.get_matrix())
                        aug_feat = trot.transform_points(cur_features.float()).numpy()

                        if self.recenter:
                            aug_feat = center_tractography(self.root, aug_feat) #!!!! compute this for 
                            self.logger.info('Subject idx {} (unique ID {}, aug {}): rotation {}, scale {}, translation {} (centered). Aug axis order: {}'
                                            .format(i_subject, unique_id, i_aug, cur_angles, scale_r, [LR_trans, AP_trans, SI_trans], self.aug_axis_lst))
                        else:
                            self.logger.info('Subject idx {} (unique ID {}, aug {}): rotation {}, scale {}, translation {}. Aug axis order: {}'
                                            .format(i_subject, unique_id, i_aug, cur_angles, scale_r, [LR_trans, AP_trans, SI_trans], self.aug_axis_lst))

                        aug_features[i_aug, ...] = aug_feat

                    brain_features[i * self.aug_times:(i + 1) * self.aug_times, :, :, :] = aug_features

                else:
                    brain_features[i, :, :, :] = cur_features

                if self.use_tracts_training:
                    ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping()
                    cur_labels = cluster2tract_label(cur_labels, ordered_tract_cluster_mapping_dict, output_lst=False)
                
                if self.aug_times > 0:
                    brain_labels[i * self.aug_times:(i + 1) * self.aug_times, ...] = cur_labels[None, ...].repeat(self.aug_times, axis=0)
                else:
                    brain_labels[i, ...] = cur_labels

            # Append results from this chunk
            all_brain_features.append(brain_features)
            all_brain_labels.append(brain_labels)

        # Concatenate all chunks into final arrays
        brain_features = np.concatenate(all_brain_features, axis=0)
        brain_labels = np.concatenate(all_brain_labels, axis=0)
        
        return brain_features, brain_labels
        
        
    def _cal_info_feat(self, chunk_size=5):
        """
        Calculate local-global representations using chunk processing to reduce memory usage.
        
        Args (self):
            n_subject=num_unique_subjects*self.aug_times
            brain_feature (array): [n_subject, n_fiber, n_point, n_feat] feature in brain-level 
            brain_label (array): [n_subject, n_fiber, n_point or 1] label in brain-level
            k (int, optional): How many k nearest neighbors are needed.
        
        Returns:
            fiber_feat (array): [n_subject*n_fiber, n_point, n_feat] feature in fiber-level
            fiber_label (array): [n_subject*n_fiber, n_point or 1] label in fiber-level
            local_feat (array): [n_subject*n_fiber, n_point, n_feat, k] k nearest neighbor streamline feature (local)
            global_feat (array): [n_subject, n_point, n_feat, k_global] randomly selected streamline feature (global)
        """
        
        num_subjects = self.brain_features.shape[0]
        num_feat_per_point = self.brain_features.shape[-1]

        all_local_feat = []
        all_global_feat = []
        all_fiber_feat = []
        all_fiber_label = []
        all_new_subidx = []

        for chunk_start in range(0, num_subjects, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_subjects)
            current_chunk_size = chunk_end - chunk_start

            if self.k > 0:
                local_feat = np.zeros((current_chunk_size, self.num_fiber, self.num_point, num_feat_per_point, self.k), dtype=np.float32)
            else:
                local_feat = np.zeros((current_chunk_size, self.num_fiber, self.num_point, num_feat_per_point, 1), dtype=np.float32)
                local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, 1)

            if self.k_global > 0:
                global_feat = np.zeros((current_chunk_size, self.num_point, num_feat_per_point, self.k_global), dtype=np.float32)
            else:
                global_feat = np.zeros((current_chunk_size, self.num_point, num_feat_per_point, 1), dtype=np.float32)

            new_subidx = np.zeros((current_chunk_size, self.num_fiber), dtype=np.int64)

            for i, cur_idx in enumerate(range(chunk_start, chunk_end)):
                time_start = time.time()
                cur_feat = self.brain_features[cur_idx, ...]
                cur_feat = np.transpose(cur_feat, (0, 2, 1))

                if self.k > 0:
                    cur_local_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k, self.use_endpoints_dist, self.cal_equiv_dist) #!
                    cur_local_feat = cur_local_feat.reshape(self.num_fiber, self.k, num_feat_per_point, self.num_point)
                    cur_local_feat = np.transpose(cur_local_feat, (0, 3, 2, 1))

                if self.k_global > 0:
                    random_idx = np.random.randint(0, cur_feat.shape[0], self.k_global)
                    cur_global_feat = cur_feat[random_idx, ...]
                    cur_global_feat = cur_global_feat.transpose(2, 1, 0)

                cur_subidx = np.ones((cur_feat.shape[0]), dtype=np.int64) * cur_idx
                time_end = time.time()

                if self.aug_times > 0:
                    self.logger.info('Subject {} Aug {} with {} fibers feature calculation time: {:.2f} s'
                                    .format(cur_idx // self.aug_times, cur_idx % self.aug_times, self.num_fiber, time_end - time_start))
                else:
                    self.logger.info('Subject {} (No Aug) with {} fibers feature calculation time: {:.2f} s'
                                    .format(cur_idx, self.num_fiber, time_end - time_start))

                if self.k > 0:
                    local_feat[i, ...] = cur_local_feat
                if self.k_global > 0:
                    global_feat[i, ...] = cur_global_feat
                new_subidx[i, ...] = cur_subidx

            if self.k > 0:
                local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, self.k)
            new_subidx = new_subidx.reshape(-1, 1)

            fiber_feat = self.brain_features[chunk_start:chunk_end].reshape(-1, self.num_point, num_feat_per_point)
            fiber_label = self.brain_labels[chunk_start:chunk_end].reshape(-1, 1)

            all_local_feat.append(local_feat)
            all_global_feat.append(global_feat)
            all_fiber_feat.append(fiber_feat)
            all_fiber_label.append(fiber_label)
            all_new_subidx.append(new_subidx)

        local_feat = np.concatenate(all_local_feat, axis=0)
        global_feat = np.concatenate(all_global_feat, axis=0)
        fiber_feat = np.concatenate(all_fiber_feat, axis=0)
        fiber_label = np.concatenate(all_fiber_label, axis=0)
        new_subidx = np.concatenate(all_new_subidx, axis=0)

        return fiber_feat, fiber_label, local_feat, global_feat, new_subidx
    


def cal_local_feat(cur_feat, k_ds_rate, k, use_endpoints_dist, cal_equiv_dist):
    """ Calculate the local feature for all streamlines in cur_feat
    Args:
        cur_feat: [n_fiber, n_feat, n_point]
        k_ds_rate: the rate of downsample the fibers to calculate the distance matrix. 1 means no downsample
        k: the number of nearest neighbor streamlines (local)
        use_endpoints_dist (bool): whether to use the distance between endpoints to calculate the distance matrix
        cal_equiv_dist (bool): whether to calculate the equivalent distance matrix

    Returns:
        cur_local_feat: [n_fiber*k, n_feat, n_point], features of the k nearest neighbor streamlines
    """
    # local feat
    # [n_fiber, k], [n_fiber, k], [n_fiber,n_feat,n_point], [n_fiber,n_feat,n_point]
    near_idx, near_flip_mask, ds_cur_feat, ds_cur_feat_equiv = dist_mat_knn(
                        torch.from_numpy(cur_feat), k_ds_rate, k, use_endpoints_dist, cal_equiv_dist)
    cur_local_feat_org = ds_cur_feat[near_idx.reshape(-1),...]  # [n_fiber*k, n_feat, n_point]
    cur_local_feat_equiv = ds_cur_feat_equiv[near_idx.reshape(-1),...]  # [n_fiber*k, n_feat, n_point]
    near_flip_mask = near_flip_mask.reshape(-1)[:,None,None]  # [n_fiber*k, 1, 1]
    near_nonflip_mask = 1-near_flip_mask
    cur_local_feat = cur_local_feat_org*near_nonflip_mask + cur_local_feat_equiv*near_flip_mask  # [n_fiber*k, n_feat, n_point]

    return cur_local_feat 


def dist_mat_knn(brain_feat, k_ds_rate, k, use_endpoints_dist, cal_equiv_dist):
    """ 
        calculate the distance between streamlines (fibers) and then find the neighbor streamlines (fibers)
        input (self) 
            brain_feat: [n_fiber, n_feat, n_point]
            k_ds_rate (float): the rate of downsample the fibers to calculate the distance matrix
            k (int): the number of nearest neighbors
            use_endpoints_dist (bool): whether to use endpoints distance to calculate the nearest neighbor streamlines (fibers)
            cal_equiv_dist (bool): whether to calculate the equivalent distance
        output 
            idx (the nearest idx): [n_fiber, k]
            flip_mask (the flip mask): [n_fiber, k]. whether the nearest fiber feature should use flipped fiber (reverse order)
            ds_brain_feat (the downsampled brain_feat): [n_fiber, n_feat, n_point]
            ds_brain_feat_equiv (the downsampled equivalent (reverse order) brain_feat): [n_fiber, n_feat, n_point]
    """
        
    # calculate the distance matrix. Take the minus since we wanna find the smallest distance using topk.
    if 0 < k_ds_rate < 1:
        num_ds_feat =  int(brain_feat.shape[0]*k_ds_rate)
        ds_indices = np.random.choice(brain_feat.shape[0], size=num_ds_feat, replace=False)
        downsample_feat = brain_feat[ds_indices,:,:]  # [n_ds_fiber, n_point, n_feat]
    else:
        downsample_feat = brain_feat
    if use_endpoints_dist:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation_endpoints(brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)  # (n_fiber, n_ds_fiber) for dist_mat
    else:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation(brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)  # (n_fiber, n_ds_fiber)
    
    topk_idx = dist_mat.topk(k=k, largest=False, dim=-1)[1]   # (N_fiber, k). largest is False then the k smallest elements are returned.
    near_idx = topk_idx[:,:]   # (N_fiber, k)
    # print(dist_mat[0,near_idx[0,:]])
    near_flip_mask = torch.gather(flip_mask, dim=1, index=near_idx) # (N_fiber, k). The flip mask of the info fibers (neighbor).

    return near_idx.numpy(), near_flip_mask.numpy(), ds_brain_feat.numpy(), ds_brain_feat_equiv.numpy()

        
def center_tractography(input_path, feat_RAS, out_path=None, logger=None, tractography_name=None,save_data=False):
    """Recenter the tractography to atlas center
        feat_RAS: [n_fiber, n_point, n_feat]"""
    HCP_center = np.load(os.path.join(input_path, 'HCP_mass_center.npy'))  # (15(n_point),3(n_feat)) from 100 unrelated HCP subjects (atlas). The calculation function is in func_intra.py
    test_subject_center = np.mean(feat_RAS, axis=0)
    displacement = HCP_center - test_subject_center
    c_feat_RAS = feat_RAS + displacement  # recenter the tractography to HCP atlas center
    if save_data:
        recenter_path = os.path.join(out_path, 'recentered_tractography')
        makepath(recenter_path)
        feat_RAS_pd = array2vtkPolyData(c_feat_RAS)
        wma.io.write_polydata(feat_RAS_pd, os.path.join(recenter_path, 'recentered_{}'.format(tractography_name)))
        logger.info('Saved recentered tractography to {}'.format(recenter_path))
    return c_feat_RAS
