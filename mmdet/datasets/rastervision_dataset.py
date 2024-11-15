from mmdet.datasets.builder import DATASETS
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
    BufferTransformer
)
from rastervision.pytorch_learner import(
    ObjectDetectionSlidingWindowGeoDataset
)

@DATASETS.register_module()
class RasterVisionDataset(Dataset):
    def __init__(self, image_dir,vector_dir,scene_csv_path,class_config, pipeline, **kwargs):
        super().__init__(**kwargs)
        self.image_dir = image_dir
        self.vector_dir = vector_dir
        self.scene = scene_csv_path
        self.class_config = class_config
        self.pipeline = pipeline
        print("Reading Scene.csv")
        try:
            data_df = pd.read_csv(self.scene)
        except:
            print("Can't read Scene.csv")
        
        labeled_sceneList = [
            self._create_OD_scene(
                os.path.join(self.vector_dir,*row['aoi_path'].split("\\")),
                os.path.join(image_dir,*row['image_path'].split("\\")),
                os.path.join(vector_dir,*row['label_path'].split("\\"))) 
                for index , 
                row in data_df.iterrows()]

        labeled_dataset_list = [
            self._create_OD_dataset(scene)
             for scene in tqdm(labeled_sceneList,desc="Create Labeled Datasets:")]
        self.rastervision.dataset = ConcatDataset(labeled_dataset_list)

    def _create_OD_scene(self, aoi_path, image_path,class_config,label_path=None):
        crs_transformer = RasterioCRSTransformer.from_uri(image_path)

        # Create an extent to clip everything to that is slightly larger than the AOI
        aoiSource = GeoJSONVectorSource(
            aoi_path,crs_transformer,vector_transformers=[BufferTransformer(geom_type='Polygon', default_buf=256)])

        #Extract AOI extent
        myextent=aoiSource.extent

        rasterSource = RasterioSource(
            image_path, #path to the image
            allow_streaming=True, # allow_streaming so we don't have to load the whole image
            bbox=myextent
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
                        default_class_id=class_config.get_class_id('object') #use class config
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

    def _find_patch_size(self, imagery_path):
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

    def _create_OD_dataset(self,scene):
        pixel_size = self._find_patch_size(scene.raster_source.imagery_path)
        patch_ground_size = 256*.3
        #Default patch size
        msize=round(patch_ground_size/pixel_size)
        mstride=msize
        
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

        

    def __len__(self):
        return len(self.rastervision_dataset)
       

    def _coco_box(self, label):
        boxes = label.convert_boxes('xywh')
        labels =label.get_field('class_ids')
        return boxes,labels

    def __getitem__(self, idx):
        img, target = self.rastervision_dataset[idx]

        #Convert target to MMDetection format
        gt_bboxes, gt_labels = self._coco_box(target)
        data={
            'img' : img,
            'gt_bboxes' : gt_bboxes,
            'gt_labels' : gt_labels
        }
        return self.pipeline(data)

    