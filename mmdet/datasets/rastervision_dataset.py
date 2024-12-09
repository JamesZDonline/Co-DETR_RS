from mmdet.datasets.builder import DATASETS
from mmdet.datasets import CustomDataset
from collections import OrderedDict
from mmdet.core import eval_map, eval_recalls
from mmcv.utils import print_log
import torch
from .pipelines import Compose
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd
import os
import tqdm
import rasterio
from rastervision.core.data import(
    RasterioSource,
    ObjectDetectionLabelSource,
    RasterioCRSTransformer,
    GeoJSONVectorSource,
    Scene,
    ClassInferenceTransformer,
    BufferTransformer,
    ClassConfig,
    ObjectDetectionLabels
)
from rastervision.pytorch_learner import(
    ObjectDetectionSlidingWindowGeoDataset,
    # ObjectDetectionRandomWindowGeoDataset
)
from .custom_od_random_window_geodataset import CustomODRandomWindowGeoDataset

@DATASETS.register_module()
class RasterVisionDataset(CustomDataset):
    CLASSES = ['Arch. Corral', 'Arch Estructura','Arch Patio','Mod. Corral','Mod. Estructura','Mod Patio']
    COLORS = ['lightblue','green','purple','darkblue','darkgreen','orange']

    def __init__(
            self,
            image_dir:str,
            vector_dir:str,
            pipeline,
            scene_csv_path:str,
            class_config:ClassConfig=None,
            data_type:str="training",
            test_code:bool = False,
            max_windows:int = 100,
            neg_ratio:float = 10,
            resize_dim:int = 224,
            within_aoi:bool =True,
            rgb: bool = False,
            **kwargs):
        self.image_dir = image_dir
        self.vector_dir = vector_dir
        self.scene_path = scene_csv_path
        self.data_type:str = data_type
        self.testing = test_code
        self.neg_ratio = neg_ratio
        self.max_windows = max_windows
        self.rgb = rgb
        self.resize_dim = resize_dim
        self.within_aoi = within_aoi

        #Setup class_config if not passed in init
        if class_config==None:
            self.class_config = ClassConfig(
            names=self.CLASSES,
            colors=self.COLORS,
            null_class=None)
        else:
            self.class_config=class_config
        
        #Read in the Scenes.csv file
        print(f"Reading {self.scene_path}")
        try:
            data_df = pd.read_csv(self.scene_path)
            if self.testing:
                data_df = data_df.sample(n=20)
            if not self.data_type=='training':
                data_df = data_df[data_df['dataset']==0]
        except Exception as e:
            print("Can't read Scene.csv")
            print(str(e))
        
        #Construct a list of scenes. Each row in scenes.csv becomes a separate scene
        print('Building Scenes')
        labeled_sceneList = []
        for index,row in data_df.iterrows():
            aoi_fullpath = os.path.join(self.vector_dir,*row['aoi_path'].split("\\"))
            image_fullpath = os.path.join(image_dir,*row['image_path'].split("\\"))
            label_fullpath = os.path.join(vector_dir,*row['label_path'].split("\\"))
            labeled_sceneList.append(self._create_OD_scene(aoi_fullpath,image_fullpath,label_fullpath,class_config=self.class_config))
        
        #Construct a list of datasets from the list of scenes
        print("Building Datasets")
        labeled_dataset_list = []
        for scene_ in labeled_sceneList:
            labeled_dataset_list.append(self._create_OD_dataset(scene_,neg_ratio=self.neg_ratio,within_aoi=self.within_aoi,max_windows=self.max_windows))
        self.rastervision_dataset = ConcatDataset(labeled_dataset_list)
        print(f'There are {len(self.rastervision_dataset)} images in the dataset!')
        super(RasterVisionDataset,self).__init__(img_prefix=image_dir,pipeline=pipeline,ann_file = scene_csv_path,**kwargs)
        self.pipeline = Compose(pipeline)
        


    def _create_OD_scene(self, aoi_path, image_path,label_path,class_config)-> Scene:
        crs_transformer = RasterioCRSTransformer.from_uri(image_path)

        rasterSource = RasterioSource(
                image_path, #path to the image
                allow_streaming=True, # allow_streaming so we don't have to load the whole image
            ) 
        
        pixel_size = self._find_patch_size(rasterSource.imagery_path)
        patch_ground_size = 256*.3
        #Default patch size
        msize=round(patch_ground_size/pixel_size)

        # Create an extent to clip everything to that is slightly larger than the AOI
        aoiSource = GeoJSONVectorSource(
            aoi_path,
            crs_transformer,
            # vector_transformers=[BufferTransformer(geom_type='Polygon', default_buf=256)]
        )
    
        # Extract AOI extent
        myextent=aoiSource.extent
        use_sliding_windows = self.data_type!="training"  or self.approx_num_aoi_chips(myextent, msize)<=self.max_windows 
        if use_sliding_windows:
            rasterSource = RasterioSource(
                image_path, #path to the image
                allow_streaming=True, # allow_streaming so we don't have to load the whole image
                bbox=myextent
                ) # Clip the image to the extent of the aoi. This means chip windows will only be created within the bounds of the aoi extent
            #Create the AOI
            aoiSource = GeoJSONVectorSource(
                aoi_path,rasterSource.crs_transformer)         
            

        #If there are labels, import them as GeoJSONVectorSource
        if not os.path.exists(label_path):
            print("No Label geojson exists")
            print(label_path)
            labelSource=None
        if label_path is not None and os.path.exists(label_path):
            #import labels as a GeoJSONVectorSource
            labelVectorSource = GeoJSONVectorSource(
                label_path, # path to the label geojson
                crs_transformer, # convert labels from geographic to pixel coordinates
                vector_transformers=[
                    ClassInferenceTransformer(
                        default_class_id=0
                    )
                ]
            )

        #Convert the label vector to a lable source (format suitable for machine learning)
            try:
                labelSource=ObjectDetectionLabelSource(vector_source=labelVectorSource, #use the above label vectors
                    bbox=rasterSource.bbox, #clip to aoi extent
                    )
            except ValueError as e:
                pass
                # labelSource=None
                # print("There was an error. Maybe there are no labels in the AOI?")
                print(aoi_path)
                print(e)

        #Finally create the scene. set the id to "platform" so we can use that information later when we create the dataset
        scene=Scene(
            id=os.path.splitext(os.path.basename(aoi_path))[0],
            raster_source=rasterSource,
            label_source=labelSource,
            aoi_polygons=aoiSource.get_geoms())
        return scene

    def _find_patch_size(self, imagery_path:str)->int:
        with rasterio.open(imagery_path) as image:
            target_crs = rasterio.crs.CRS.from_string('EPSG:3857')
            # print(target_crs)
            img_crs = image.crs
            # print(img_crs)
            transform = image.transform
            width=image.width
            height=image.height
            left=transform[2]
            right = left+transform[0]*width
            bottom=transform[5]+transform[4]*height
            top=transform[5]
            pixel_size = rasterio.warp.calculate_default_transform(src_crs=img_crs,dst_crs=target_crs,width=width,height=height,left=left,right=right,bottom=bottom,top=top)[0][0]
            return pixel_size

    def approx_num_aoi_chips(self,aoi_box,chip_size:int) -> int:
        height = aoi_box.ymax-aoi_box.ymin
        width = aoi_box.xmax-aoi_box.xmin
        approx_num_chips = int((height*width)/chip_size**2)
        # print(f"AOI contains approximately {approx_num_chips} chips")
        return approx_num_chips


    def _create_OD_dataset(self,scene:Scene,neg_ratio:float=10,max_windows:int=100,num_pixels:int=256,within_aoi:bool = True)-> CustomODRandomWindowGeoDataset|ObjectDetectionSlidingWindowGeoDataset:
        pixel_size = self._find_patch_size(scene.raster_source.imagery_path)
        patch_ground_size = num_pixels*.3
        #Default patch size
        msize=round(patch_ground_size/pixel_size)
        mstride=msize
        
        aoi_box = scene.bbox
        use_sliding_windows = self.data_type!="training"  or self.approx_num_aoi_chips(aoi_box, msize)<=max_windows 


        if not use_sliding_windows:
            # print("Finding Random Windows")
            try:
                return self._random_window_dataset(scene=scene,neg_ratio=neg_ratio,within_aoi=True,num_pixels=num_pixels,max_windows=max_windows)
            except:
                return self._random_window_dataset(scene=scene,neg_ratio=None,within_aoi=True,num_pixels=num_pixels,max_windows=5)


        
        #Create the Dataset
        ds = ObjectDetectionSlidingWindowGeoDataset(
            scene=scene, # a scene object as created in step 1
            size=msize, # the dimension of the patch
            stride=mstride, # equal to the patch so there is no overlap and no gaps
            # out_size=256, # reshape the patch to be 256x256
            # pad_direction="both",
            within_aoi=within_aoi 
        )
        return(ds)
    
    def extract_data_info(self,dataset, window):
        #Pull the labels for a given window
        labels = dataset.scene.label_source.get_labels(window)
        # Find the dimensions of an image chip from the given dataset (note: this is different depending on the spatial resolution of the image)
        width,height = dataset.size
        # Calculate a resize ratio. This will be used to transform the bbox in the same way the resize will to the image
        resize_ratio =self.resize_dim/width
        #convert the label bboxes into local (chip) pixel coordinates)
        window_global = window.to_global_coords(dataset.scene.bbox)
        bboxes = labels.get_npboxes()
        bboxes = ObjectDetectionLabels.global_to_local(bboxes,window_global)
        bboxes = torch.from_numpy(bboxes)
        bboxes = torch.clamp(bboxes*resize_ratio,min=0,max=224)
        bboxes = bboxes[:, [1, 0, 3, 2]]

        # boxes = torch.from_numpy(labels.get_npboxes())
        ann_info = {'bboxes':bboxes,'labels':torch.from_numpy(labels.get_class_ids())}
        return dict(filename=dataset.scene.id, ori_width = width, ori_height=height,width=width, height=height, ann=ann_info)
    
        #MMDET expects this method.
    def load_annotations(self, ann_file):
        print('LOADING ANNOTATIONS')
        
        if self.data_type=="training":
            data_infos = []
            
            for dataset in (self.rastervision_dataset.datasets):
                for chip in range(len(dataset)):
                    img_info = f"{dataset.scene.id}"
                    ann_info = "nonesense"
                    data_infos.append(dict(filename=img_info, width=224, height=224, ann=ann_info))
        else:
            data_infos = [self.extract_data_info(dataset,window) for dataset in self.rastervision_dataset.datasets for window in dataset.windows]
        return data_infos
    
    # def get_ann_info(self,idx):

    
    def _random_window_dataset(self,scene:Scene,neg_ratio:float,within_aoi:bool,num_pixels:int,max_windows:int)-> CustomODRandomWindowGeoDataset:
        ds = CustomODRandomWindowGeoDataset(
            scene=scene,
            neg_ratio=neg_ratio,
            within_aoi=within_aoi,
            size_lims=(num_pixels,num_pixels+1),
            out_size=num_pixels,
            max_windows=max_windows
        )
        return ds

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        # results['bbox_fields'] = ['gt_bboxes']
        results['mask_fields'] = []
        results['seg_fields'] = []
    
        # idx -> data that has been through the defined augmentation pipeline
    def prepare_train_img(self, idx):
        # Grab an image and target from the rastervision dataset
        img, target = self.rastervision_dataset[idx]
        if self.rgb:
            img = img[[4,2,1],:,:]
        #MMDEt pipeline expects a numpy in 
        img=img.permute(1,2,0)
        img = img.numpy()
        # img = np.transpose(img)
        
        #Convert target to MMDetection format
        gt_bboxes=target.boxes.numpy()
        gt_labels = target.get_field('class_ids')
        results = {'img' : img,
                   'filename':'transformed_'+self.data_infos[0]['filename'],
                   'ori_filename':self.data_infos[0]['filename'],
                   'ori_shape' : img.shape,
            "img_shape":img.shape,
            'gt_bboxes' : gt_bboxes,
            'gt_labels' : gt_labels,
            'bbox_fields' : ['gt_bboxes']
            
        }
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def prepare_test_img(self, idx):
        # Grab an image and target from the rastervision dataset
        img, target = self.rastervision_dataset[idx]
        if self.rgb:
            img = img[[4,2,1],:,:]
        #MMDEt pipeline expects a numpy in 
        img=img.permute(1,2,0)
        img = img.numpy()
        results = {'img' : img,
                   'filename':"none",
                   'ori_filename':"none",
                   'ori_shape' : img.shape,
            "img_shape":img.shape,
            'bbox_fields' : [],
            'img_norm_cfg':dict(mean=[.5,.5,.5], std=[.5,.5,.5], to_rgb=True)
        }
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def evaluate(self,
                    results,
                    metric='mAP',
                    logger=None,
                    proposal_nums=(100, 300, 1000),
                    iou_thr=0.5,
                    scale_ranges=None):
            """Evaluate the dataset.

            Args:
                results (list): Testing results of the dataset.
                metric (str | list[str]): Metrics to be evaluated.
                logger (logging.Logger | None | str): Logger used for printing
                    related information during evaluation. Default: None.
                proposal_nums (Sequence[int]): Proposal number used for evaluating
                    recalls, such as recall@100, recall@1000.
                    Default: (100, 300, 1000).
                iou_thr (float | list[float]): IoU threshold. Default: 0.5.
                scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                    Default: None.
            """

            if not isinstance(metric, str):
                assert len(metric) == 1
                metric = metric[0]
            allowed_metrics = ['mAP', 'recall']
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
            # annotations = [{'bboxes':self[i]['gt_bboxes'],'labels':self[i]['gt_labels']} for i in range(len(self))]
            
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            eval_results = OrderedDict()
            iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
            if metric == 'mAP':
                assert isinstance(iou_thrs, list)
                mean_aps = []
                for iou_thr in iou_thrs:
                    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    mean_ap, _ = eval_map(
                        results,
                        annotations,
                        scale_ranges=scale_ranges,
                        iou_thr=iou_thr,
                        dataset=self.CLASSES,
                        logger=logger)
                    mean_aps.append(mean_ap)
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            elif metric == 'recall':
                gt_bboxes = [ann['bboxes'] for ann in annotations]
                recalls = eval_recalls(
                    gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
                for i, num in enumerate(proposal_nums):
                    for j, iou in enumerate(iou_thrs):
                        eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
                if recalls.shape[1] > 1:
                    ar = recalls.mean(axis=1)
                    for i, num in enumerate(proposal_nums):
                        eval_results[f'AR@{num}'] = ar[i]
            return eval_results

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann']
    


    def __len__(self):
        return len(self.rastervision_dataset)
       

    def _coco_box(self, label):
        boxes = label.convert_boxes('xywh')
        labels =label.get_field('class_ids')
        return boxes,labels

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    