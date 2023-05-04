# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from train_detectron2_detection import update_path
from ikomia import core, dataprocess
from ikomia.dnn import datasetio, dnntrain
from ikomia.core.task import TaskParam
# Your imports below
from datetime import datetime
# Setup detectron2 logger
import detectron2
import torch
from detectron2.utils.logger import setup_logger

setup_logger()
import os
import copy
from ikomia.core import config as ikcfg

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog
from train_detectron2_detection.utils import MyMapper, MyTrainer, register_datasets
import gc


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainDetectron2DetectionParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        self.cfg["model_name"] = "COCO-Detection/faster_rcnn_R_101_FPN_3x"
        self.cfg["use_custom_cfg"] = False
        self.cfg["config"] = ""
        self.cfg["max_iter"] = 100
        self.cfg["batch_size"] = 2
        self.cfg["input_size"] = 400
        self.cfg["use_pretrained"] = True
        self.cfg["output_folder"] = os.path.dirname(__file__) + "/runs"
        self.cfg["learning_rate"] = 0.0025
        self.cfg["dataset_split_ratio"] = 0.8
        self.cfg["eval_period"] = 50

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["use_custom_cfg"] = eval(param_map["use_custom_cfg"])
        self.cfg["config"] = param_map["config"]
        self.cfg["max_iter"] = int(param_map["max_iter"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["input_size"] = int(param_map["input_size"])
        self.cfg["use_pretrained"] = eval(param_map["use_pretrained"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["learning_rate"] = float(param_map["learning_rate"])
        self.cfg["dataset_split_ratio"] = float(param_map["dataset_split_ratio"])
        self.cfg["eval_period"] = int(param_map["eval_period"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "model_name": self.cfg["model_name"],
            "use_custom_cfg": str(self.cfg["use_custom_cfg"]),
            "config": self.cfg["config"],
            "max_iter": str(self.cfg["max_iter"]),
            "batch_size": str(self.cfg["batch_size"]),
            "input_size": str(self.cfg["input_size"]),
            "use_pretrained": str(self.cfg["use_pretrained"]),
            "output_folder": self.cfg["output_folder"],
            "learning_rate": str(self.cfg["learning_rate"]),
            "dataset_split_ratio": str(self.cfg["dataset_split_ratio"]),
            "eval_period": str(self.cfg["eval_period"])
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainDetectron2Detection(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        self.stop_train = False
        self.epochs_done = None
        self.epochs_todo = None
        # Percentage of training done for display purpose
        self.advancement = 0
        # Create parameters class
        if param is None:
            self.set_param_object(TrainDetectron2DetectionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 100

    def update_progress(self):
        self.epochs_done += 1
        steps = range(self.advancement, int(100 * self.epochs_done / self.epochs_todo))
        for step in steps:
            self.emit_step_progress()
            self.advancement += 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()
        self.stop_train = False
        gc.collect()
        torch.cuda.empty_cache()

        ik_dataset = self.get_input(0)
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")

        # Get parameters :
        param = self.get_param_object()

        tb_logdir = os.path.join(ikcfg.main_cfg["tensorboard"]["log_uri"], str_datetime)

        if not param.cfg["use_custom_cfg"]:
            out_dir = param.cfg["output_folder"] + "/" + str_datetime

            cfg = get_cfg()
            config_path = os.path.join(os.path.dirname(detectron2.__file__), "model_zoo", "configs",
                                       param.cfg["model_name"] + '.yaml')
            cfg.merge_from_file(config_path)
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            if not param.cfg["use_pretrained"]:
                cfg.MODEL.WEIGHTS = None
            else:
                model_zoo.get_checkpoint_url((param.cfg["model_name"] + ".yaml").replace("\\", "/"))
            cfg.INPUT.MAX_SIZE_TEST = param.cfg["input_size"]
            cfg.INPUT.MAX_SIZE_TRAIN = param.cfg["input_size"]
            cfg.SOLVER.IMS_PER_BATCH = param.cfg["batch_size"]
            cfg.SOLVER.BASE_LR = param.cfg["learning_rate"]
            cfg.SOLVER.MAX_ITER = param.cfg["max_iter"]
            cfg.TEST.EVAL_PERIOD = min(param.cfg["eval_period"], cfg.SOLVER.MAX_ITER - 1)
            cfg.SOLVER.STEPS = (int(0.8 * param.cfg["max_iter"]), int(0.9 * param.cfg["max_iter"]))
            cfg.SOLVER.WARMUP_ITERS = 1000
            cfg.DATALOADER.NUM_WORKERS = 0
            cfg.OUTPUT_DIR = out_dir

        else:
            if os.path.isfile(param.cfg["config"]):
                with open(param.cfg["config"], 'r') as file:
                    cfg_data = file.read()
                    cfg = CfgNode.load_cfg(cfg_data)
                    out_dir = cfg.OUTPUT_DIR
            else:
                print("Unable to load config file {}".format(param.cfg["config"]))
                self.end_task_run()

        os.makedirs(out_dir, exist_ok=True)

        # Fixed dataset names
        cfg.DATASETS.TRAIN = ("TrainDetectionDataset",)
        cfg.DATASETS.TEST = ("TestDetectionDataset",)
        register_datasets(ik_dataset.data, param.cfg["dataset_split_ratio"])
        cfg.CLASS_NAMES = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg.CLASS_NAMES)
        cfg.MODEL.RETINANET.NUM_CLASSES = len(cfg.CLASS_NAMES)
        self.advancement = 0
        self.epochs_todo = cfg.SOLVER.MAX_ITER
        self.epochs_done = 0
        with open(os.path.join(out_dir, "config.yaml"), 'w') as f:
            f.write(cfg.dump())
        trainer = MyTrainer(cfg, tb_logdir, self.get_stop_train, self.log_metrics, self.update_progress)
        if param.cfg["use_pretrained"]:
            trainer.resume_or_load(resume=False)  # load MODEL.WEIGHTS
        trainer.train()

        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        # Call end_task_run to finalize process
        self.end_task_run()

    def get_stop_train(self):
        return self.stop_train

    def stop(self):
        super(TrainDetectron2Detection, self).stop()
        self.stop_train = True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainDetectron2DetectionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_detectron2_detection"
        self.info.short_description = "Train for Detectron2 detection models"
        self.info.description = "Train for Detectron2 detection models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.0"
        self.info.icon_path = "icons/detectron2.png"
        self.info.authors = "Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick"
        self.info.article = "Detectron2"
        self.info.journal = ""
        self.info.year = 2019
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://detectron2.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "train, detectron2, object, detection"

    def create(self, param=None):
        # Create process object
        return TrainDetectron2Detection(self.info.name, param)
