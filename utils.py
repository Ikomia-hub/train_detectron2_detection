import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader
import copy
from detectron2.data import detection_utils as utils
from typing import List, Union
import logging
# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.utils.events import TensorboardXWriter
from detectron2.engine.hooks import BestCheckpointer
from detectron2.utils import comm
from detectron2.utils.events import EventStorage
import os
import torch
import random


class MyTrainer(DefaultTrainer):

    def __init__(self, cfg, tb_dir, stop_train, log_metrics, update_progress):
        self.tensorboard_dir = tb_dir
        super().__init__(cfg)
        self.stop_train = stop_train
        self.log_metrics = log_metrics
        self.update_progress = update_progress

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=MyMapper(True, augmentations=[
            T.RandomBrightness(0.9, 1.1),
            T.RandomFlip(prob=0.5),
            T.ResizeShortestEdge(short_edge_length=cfg.INPUT.MAX_SIZE_TRAIN, max_size=cfg.INPUT.MAX_SIZE_TRAIN)
        ], image_format="RGB"))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR)

    def build_writers(self):
        return [TensorboardXWriter(self.tensorboard_dir)]

    def build_hooks(self):
        ret = super().build_hooks()
        ret.append(BestCheckpointer(eval_period=self.cfg.TEST.EVAL_PERIOD,
                                    checkpointer=self.checkpointer,
                                    val_metric="bbox/AP",
                                    mode="max",
                                    file_prefix="model_best"))
        return ret

    def train(self):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        self.iter = self.start_iter

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    self.update_progress()
                    if self.iter % 20 == 0:
                        self.log_metrics(
                            {name.replace("@", "_"): value[0] for name, value in self.storage.latest().items()},
                            step=self.iter)
                    if self.stop_train():
                        logger.info("Training stopped by user at iteration {}".format(self.iter))
                        with open(os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth"), "w") as f:
                            f.write("")
                        self.checkpointer.save("model_final")
                        break
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results


class MyMapper(DatasetMapper):
    def __init__(
            self,
            is_train: bool,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            keypoint_hflip_indices=None):
        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.keypoint_hflip_indices = keypoint_hflip_indices
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # can use other ways to read image
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        # See "Data Augmentation" tutorial for details usage
        auginput = T.AugInput(image)
        transform = self.augmentations(auginput)
        image = torch.from_numpy(auginput.image.transpose(2, 0, 1).copy())
        img_shape = image.shape[1:]

        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
        ]

        return {
            # create the format that the model expects
            "image": image,
            "instances": utils.annotations_to_instances(annos, img_shape),
            "width": img_shape[1],
            "height": img_shape[0]
        }


def register_dataset(dataset_name, images, metadata):
    DatasetCatalog.register(dataset_name, lambda: images)
    MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id = metadata["thing_dataset_id_to_contiguous_id"]

    MetadataCatalog.get(dataset_name).thing_classes = metadata["thing_classes"]


def register_datasets(data, split):
    category_names = data["metadata"]["category_names"]
    thing_classes = [v for k, v in sorted(category_names.items(), key=lambda item: item[0])]
    """thing_dataset_id_to_contiguous_id = None
    if not (min(category_names) == 1 and max(category_names) == len(category_names)):"""
    id_map = {v: i for i, v in enumerate(category_names)}
    thing_dataset_id_to_contiguous_id = id_map
    for sample in data["images"]:
        if "file_name" not in sample.keys():
            sample["file_name"] = sample["filename"]
        for anno in sample["annotations"]:
            anno["bbox_mode"] = BoxMode.XYWH_ABS
            if thing_dataset_id_to_contiguous_id is not None:
                anno["category_id"] = thing_dataset_id_to_contiguous_id[anno["category_id"]]
    random.seed(10)
    random.shuffle(data["images"])
    split_id = int(len(data["images"]) * split)
    train_imgs = data["images"][:split_id]
    test_imgs = data["images"][split_id:]
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    data["metadata"]["thing_classes"] = thing_classes
    data["metadata"]["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    register_dataset("TrainDetectionDataset", train_imgs, data["metadata"])
    register_dataset("TestDetectionDataset", test_imgs, data["metadata"])
    random.seed(0)
