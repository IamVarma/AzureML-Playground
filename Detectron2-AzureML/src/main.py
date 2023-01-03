# Some basic setup:
# Setup detectron2 logger
import sys, os, distutils.core
import argparse
import wget
import mlflow
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

#verify versions
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# define functions
def main(args):

    mlflow.start_run()

    data_path = args.data_path
    model_output_path = args.model_output
    print(data_path)
    print(os.listdir(data_path))

    train_detectron2(data_path, model_output_path)

def train_detectron2(data_path, model_output_path):

    #register COCO Dataset 
    register_coco_instances("train-data", {}, os.path.join(data_path, "p1_s4_coco_fix1.json"), data_path)
    
    dataset_meta = MetadataCatalog.get("train-data") #add dataset name
    dataset_dicts = DatasetCatalog.get("train-data") #add dataset name

    #base models download
    download_basemodels()

    cfg = get_cfg()
    cfg.merge_from_file("configs/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("train-data",)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10    # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 300   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 18 # dynamically work out number of classes
    # uncomment for semantic segmentation
    #cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(class_labels) # dynamically work out number of classes
    #cfg.MODEL.DEVICE='cpu'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    #checkpointer = DetectionCheckpointer(trainer.model, save_dir='./outputs')
    torch.save(trainer.model.state_dict(), os.path.join(model_output_path, "detectron2_model.pth"))
    print(model_output_path)

def download_basemodels():
    #  download base models
    
    os.makedirs('configs', exist_ok = True)
    wget.download('https://raw.githubusercontent.com/facebookresearch/detectron2/master/configs/Base-RCNN-FPN.yaml','Base-RCNN-FPN.yaml')
    wget.download('https://raw.githubusercontent.com/facebookresearch/detectron2/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml','configs/mask_rcnn_R_50_FPN_3x.yaml')
    print("Base models download complete...")

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_output", type=str)
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":

    # parse args
    args = parse_args()

    # run main function
    main(args)