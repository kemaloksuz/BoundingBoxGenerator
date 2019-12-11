import numpy as np
import torch

from .random_sampler import RandomSampler
import pdb
class ForegroundBalancedPosSampler(RandomSampler):

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            # Get unique classes and find length
            #pdb.set_trace()
            unique_classes = assign_result.labels[pos_inds].unique()
            num_classes = len(unique_classes)
            # Create fg_num_rois sized array with all probs 1/unique_classes            
            probs=torch.ones(pos_inds.numel()).type(torch.cuda.FloatTensor)/num_classes
            # Find each positive RoI Number From Each Class with indices and normalize probs            
            for i in unique_classes:
                classes_inds = torch.nonzero(assign_result.labels == i.item())
                index=(pos_inds==classes_inds).nonzero()[:,1]
                probs[index]/=index.numel()
            # Sample according to probs  
            # pdb.set_trace()
            probs_numpy=probs.cpu().numpy()
            sampled_inds=torch.from_numpy(np.random.choice(pos_inds.cpu().numpy(), num_expected, False, probs_numpy)).cuda()     
            return sampled_inds
