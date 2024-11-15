from mmdet.datasets.builder import DATASETS
from torch.utils.data import Dataset

@DATASETS.register_module()
class Rastervision_MMDet_Wrapper(Dataset):
    def __init__(self, rastervision_OD_dataset, pipeline, **kwargs):
        super().__init__(**kwargs)
        self.rastervision_dataset = rastervision_OD_dataset
        self.pipeline = pipeline

    def __len__(self):
        return len(self.rastervision_dataset)

    def __coco_box__(self, label):
        boxes = label.convert_boxes('xywh')
        labels =label.get_field('class_ids')
        return boxes,labels

    def __getitem__(self, idx):
        img, target = self.rastervision_dataset[idx]

        #Convert target to MMDetection format
        gt_bboxes, gt_labels = self.__coco_box__(target)
        return{
            'img' : img,
            'gt_bboxes' : gt_bboxes,
            'gt_labels' : gt_labels
        }

    