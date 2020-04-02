import torch

from mmdet.core.bbox.samplers.bounding_box_generator import BoxSampler


def main():
    #given a BB generate a set of BBs with desired IoUs
    Generator = BoxSampler()
    image_size = [1333, 800]
    B = torch.tensor([400., 200., 700., 500., 0]).cuda()
    IoUs= torch.tensor([0.5, 0.6, 0.7]).cuda()
    sampled_boxes, IoUs = Generator.sample_single(B, IoUs, image_size)
    print("sampled_boxes= ", sampled_boxes)
    print("IoUs of the sampled_boxes= ", IoUs)

if __name__ == '__main__':
    main()
