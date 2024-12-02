from mmdet.datasets.builder import DATASETS
from mmdet.datasets import CustomDataset
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
    ClassConfig
)
from rastervision.pytorch_learner import(
    ObjectDetectionSlidingWindowGeoDataset,
    ObjectDetectionRandomWindowGeoDataset
)

@DATASETS.register_module()
class RasterVisionDataset(CustomDataset):
    CLASSES = ['nothing', 'Arch. Corral', 'Arch Estructura','Arch Patio','Huaca','Mod. Corral','Mod. Estructura','Mod Patio','Arch. Cista','error']
    COLORS = ['lightgray', 'lightblue','orange','green','purple','darkblue','yellow','darkgreen','black','red']

    def __init__(
            self,
            image_dir:str,
            vector_dir:str,
            pipeline,
            scene_csv_path:str,
            class_config:ClassConfig=None,
            dataset_type:str="sliding_window",
            testing:bool = False,
            neg_ratio:float = 10,
            rgb: bool = False,
            **kwargs):
        self.image_dir = image_dir
        self.vector_dir = vector_dir
        self.scene_path = scene_csv_path
        self.dataset_type = dataset_type
        self.testing = testing
        self.neg_ratio = neg_ratio
        self.rgb = rgb
        # self.class_config = class_config
        if class_config==None:
            self.class_config = ClassConfig(
            names=self.CLASSES,
            colors=self.COLORS,
            null_class='nothing')
        else:
            self.class_config=class_config
        
        #Read in the Scenes.csv file
        print(f"Reading {self.scene_path}")
        try:
            data_df = pd.read_csv(self.scene_path)
            if self.testing:
                data_df = data_df.sample(n=15)
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
            labeled_dataset_list.append(self._create_OD_dataset(scene_,self.dataset_type,neg_ratio=self.neg_ratio))
        self.rastervision_dataset = ConcatDataset(labeled_dataset_list)
        super(RasterVisionDataset,self).__init__(img_prefix=image_dir,pipeline=pipeline,ann_file = scene_csv_path,**kwargs)
        self.pipeline = Compose(pipeline)


    def _create_OD_scene(self, aoi_path, image_path,label_path,class_config):
        crs_transformer = RasterioCRSTransformer.from_uri(image_path)

        # Create an extent to clip everything to that is slightly larger than the AOI
        # aoiSource = GeoJSONVectorSource(
        #     aoi_path,crs_transformer,vector_transformers=[BufferTransformer(geom_type='Polygon', default_buf=256)])

        #Extract AOI extent
        # myextent=aoiSource.extent

        rasterSource = RasterioSource(
            image_path, #path to the image
            allow_streaming=True, # allow_streaming so we don't have to load the whole image
            # bbox=myextent
            ) # Clip the image to the extent of the aoi. This means chip windows will only be created within the bounds of the aoi extent
        
        #Create the AOI
        aoiSource = GeoJSONVectorSource(
            aoi_path,rasterSource.crs_transformer,bbox=rasterSource.bbox)

        #If there are labels, import them as GeoJSONVectorSource, clipping them to the AOI extent using bbox
        labelSource=None
        # print(label_path)
        if not os.path.exists(label_path):
            print("No Label geojson exists")
        if label_path is not None and os.path.exists(label_path):
            #import labels as a GeoJSONVectorSource
            labelVectorSource = GeoJSONVectorSource(
                label_path, # path to the label geojson
                crs_transformer, # convert labels from geographic to pixel coordinates
                bbox=rasterSource.bbox, # clip them to the AOI extent
                vector_transformers=[
                    ClassInferenceTransformer(
                        default_class_id=class_config.get_class_id('error') #use class config
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

    def _find_patch_size(self, imagery_path:str):
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


    def _create_OD_dataset(self,scene:Scene,dataset_type:str = "sliding_window",neg_ratio:float=10,max_windows:int=50,num_pixels:int=256):
        pixel_size = self._find_patch_size(scene.raster_source.imagery_path)
        patch_ground_size = num_pixels*.3
        #Default patch size
        msize=round(patch_ground_size/pixel_size)
        mstride=msize
        
        if dataset_type == "random_window":
            try:
                return self._random_window_dataset(scene=scene,neg_ratio=neg_ratio,within_aoi=True,num_pixels=num_pixels,max_windows=max_windows)
            except:
                return self._random_window_dataset(scene=scene,neg_ratio=None,within_aoi=True,num_pixels=num_pixels,max_windows=5)


        
        #Create the Dataset
        ds = ObjectDetectionSlidingWindowGeoDataset(
            scene=scene, # a scene object as created in step 1
            size=msize, # the dimension of the patch
            stride=mstride, # equal to the patch so there is no overlap and no gaps
            out_size=256, # reshape the patch to be 256x256
            # pad_direction="both",
            within_aoi=True 
        )
        return(ds)

        #MMDET expects this method. It is meaningless for us since our data is not structured
        #in the COCO format as it expects. Nevertheless, MMDET requires certain attributes associated
        #with the data. So this method provides them
    def load_annotations(self, ann_file):
        print('LOADING ANNOTATIONS')
        data_infos = []
        for idx in range(len(self.rastervision_dataset)):
            img_info = "nonsense"
            ann_info = "ann nonsense"
            data_infos.append(dict(filename="fakename", width=224, height=224, ann=ann_info))
        return data_infos
    
    def _random_window_dataset(self,scene:Scene,neg_ratio:float,within_aoi:bool,num_pixels:int,max_windows:int):
        ds = ObjectDetectionRandomWindowGeoDataset(
            scene=scene,
            neg_ratio=neg_ratio,
            within_aoi=within_aoi,
            size_lims=(num_pixels-1,num_pixels),
            out_size=num_pixels,
            max_windows=max_windows
        )
        return ds

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = ['gt_bboxes']
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
                   'filename':"none",
                   'ori_filename':"none",
                   'ori_shape' : img.shape,
            "img_shape":img.shape,
            'gt_bboxes' : gt_bboxes,
            'gt_labels' : gt_labels,
            'bbox_fields' : []
        }
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    # def prepare_test_img(self, idx):
    #     img, __ = self.rastervision_dataset[idx]
    #     img = np.transpose(img.numpy())
    #     # r, g, b = img[4, :,:], img[2, :, :], img[1, :, :]

    #     # Combine the bands into a 3-channel RGB image
    #     # rgb_img = np.stack([r, g, b], axis=-1)

    #     #Convert target to MMDetection format
    #     results = {
    #         'img_metas' : {
    #             "img_shape":img.shape,
    #         },
    #         'img' : img
    #     }
    #     self.pre_pipeline(results)
    #     return self.pipeline(results)
    

    def __len__(self):
        return len(self.rastervision_dataset)
       

    def _coco_box(self, label):
        boxes = label.convert_boxes('xywh')
        labels =label.get_field('class_ids')
        return boxes,labels

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    