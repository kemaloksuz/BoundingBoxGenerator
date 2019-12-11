import numpy as np
import torch
from .random_sampler import RandomSampler

from .sampling_result import SamplingResult
from .bounding_box_generator import BoxSampler

import pdb

class pRoIGenerator(RandomSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 IoUWeights=[0.73,0.12,0.15,0.05,0],
                 **kwargs):
        super(pRoIGenerator, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)

        self.IoUWeights=torch.tensor(IoUWeights, dtype=torch.float)
        self.pos_number=int(self.num * self.pos_fraction)
        self.sampler=BoxSampler(self.pos_number)
        
    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        bboxes = bboxes[:, :4]
        for key, value in kwargs.items(): 
            img_meta=value
            break
        import pdb
        img_size=[img_meta["img_shape"][1],img_meta["img_shape"][0]]

        num_expected_pos = int(self.num * self.pos_fraction)

        generated_boxes, generated_box_labels, overlaps,gt_inds=self.sampler.sample(\
                        torch.cat((gt_bboxes,gt_labels.unsqueeze(1).type(torch.cuda.FloatTensor)),dim=1),\
                        img_size,self.IoUWeights)

        if len(generated_boxes)>self.pos_number:
            idx=self.random_choice(torch.tensor(range(len(generated_boxes))).type(torch.cuda.LongTensor) ,self.pos_number)
            generated_boxes=generated_boxes[idx]
            generated_box_labels=generated_box_labels[idx]
            overlaps=overlaps[idx]
            gt_inds=gt_inds[idx]

        pos_inds=torch.tensor(range(len(bboxes),len(bboxes)+self.pos_number)).type(torch.cuda.LongTensor)
        bboxes=torch.cat((bboxes,generated_boxes), dim=0)
        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        assign_result.add_bboxes_(gt_inds, overlaps, generated_box_labels)

        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags)
