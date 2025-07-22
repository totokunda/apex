import torch
import torchvision
import numpy as np
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
)
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH

try:
    from groundingdino.util.inference import Model
except ImportError:
    import warnings

    warnings.warn(
        "please pip install groundingdino package, or you can refer to models/VACE-Annotators/gdino/groundingdino-0.1.0-cp310-cp310-linux_x86_64.whl"
    )

    # Define a dummy Model class to avoid further import errors if groundingdino is not installed.
    class Model:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GroundingDINO model could not be initialized because the package is not installed."
            )

        def predict_with_classes(self, *args, **kwargs):
            raise ImportError(
                "GroundingDINO model could not be used because the package is not installed."
            )

        def predict_with_caption(self, *args, **kwargs):
            raise ImportError(
                "GroundingDINO model could not be used because the package is not installed."
            )


@preprocessor_registry("gdino")
class GDINOPreprocessor(BasePreprocessor):
    def __init__(
        self,
        config_path,
        model_path,
        save_path=DEFAULT_PREPROCESSOR_SAVE_PATH,
        device="cuda",
        box_threshold=0.25,
        text_threshold=0.2,
        iou_threshold=0.5,
        use_nms=True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            save_path=save_path,
            preprocessor_type=PreprocessorType.IMAGE,
            **kwargs,
        )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        self.use_nms = use_nms
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

        self.model = Model(
            model_config_path=config_path,
            model_checkpoint_path=self.model_path,
            device=self.device,
        )

    def __call__(self, image, classes=None, caption=None, **kwargs):
        pil_image = self._load_image(image)
        image_bgr = np.array(pil_image)[:, :, ::-1]

        class_names = None

        if classes is not None:
            classes = [classes] if isinstance(classes, str) else classes
            detections = self.model.predict_with_classes(
                image=image_bgr,
                classes=classes,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            if detections.class_id is not None:
                class_names = [classes[int(cid)] for cid in detections.class_id]
        elif caption is not None:
            detections, phrases = self.model.predict_with_caption(
                image=image_bgr,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            class_names = (
                phrases.tolist() if isinstance(phrases, np.ndarray) else phrases
            )
        else:
            raise ValueError("Either 'classes' or 'caption' must be provided.")

        if self.use_nms and detections.xyxy is not None and len(detections.xyxy) > 0:
            nms_idx = (
                torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    self.iou_threshold,
                )
                .cpu()
                .numpy()
                .tolist()
            )
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            if detections.class_id is not None:
                detections.class_id = detections.class_id[nms_idx]
            if class_names is not None:
                class_names = [class_names[i] for i in nms_idx]

        boxes = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id

        ret_data = {
            "boxes": boxes.tolist() if boxes is not None else None,
            "confidences": confidences.tolist() if confidences is not None else None,
            "class_ids": class_ids.tolist() if class_ids is not None else None,
            "class_names": class_names,
        }
        return ret_data

    def __str__(self):
        return f"GDINOPreprocessor(model_path={self.model_path})"

    def __repr__(self):
        return self.__str__()
