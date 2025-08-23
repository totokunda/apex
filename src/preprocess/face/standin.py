import torch
import PIL
import numpy as np
import cv2
from typing import Optional
from insightface.app import FaceAnalysis, DEFAULT_MP_NAME
from src.preprocess.base import BasePreprocessor, BaseOutput
from facexlib.parsing import init_parsing_model
from torchvision.transforms.functional import normalize
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
from PIL import Image
import onnxruntime
import glob
from insightface.utils import ensure_available
from insightface.model_zoo import model_zoo
import os.path as osp


class SubFaceAnalysis(FaceAnalysis):
    def __init__(
        self,
        name=DEFAULT_MP_NAME,
        root="~/.insightface",
        allowed_modules=None,
        **kwargs
    ):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = ensure_available("models", name, root=root)
        onnx_files = glob.glob(osp.join(self.model_dir, name, "*.onnx"))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print("model not recognized:", onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print("model ignore:", onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (
                allowed_modules is None or model.taskname in allowed_modules
            ):
                print(
                    "find model:",
                    onnx_file,
                    model.taskname,
                    model.input_shape,
                    model.input_mean,
                    model.input_std,
                )
                self.models[model.taskname] = model
            else:
                print("duplicated model task type, ignore:", onnx_file, model.taskname)
                del model
        assert "detection" in self.models
        self.det_model = self.models["detection"]


def _img2tensor(img: np.ndarray, bgr2rgb: bool = True) -> torch.Tensor:
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


def _pad_to_square(img: np.ndarray, pad_color: int = 255) -> np.ndarray:
    h, w, _ = img.shape
    if h == w:
        return img

    if h > w:
        pad_size = (h - w) // 2
        padded_img = cv2.copyMakeBorder(
            img,
            0,
            0,
            pad_size,
            h - w - pad_size,
            cv2.BORDER_CONSTANT,
            value=[pad_color] * 3,
        )
    else:
        pad_size = (w - h) // 2
        padded_img = cv2.copyMakeBorder(
            img,
            pad_size,
            w - h - pad_size,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[pad_color] * 3,
        )

    return padded_img


class FaceOutput(BaseOutput):
    face: Image.Image
    mask: Image.Image


class FacePreprocessor(BasePreprocessor):
    def __init__(
        self,
        save_path=DEFAULT_PREPROCESSOR_SAVE_PATH,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        providers = (
            ["CUDAExecutionProvider"]
            if self.device.type == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.app = SubFaceAnalysis(
            name="antelopev2", root=save_path, providers=providers
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.parsing_model = init_parsing_model(
            model_name="bisenet", device=self.device
        )
        self.parsing_model.eval()

        print("FaceProcessor initialized successfully.")

    def __call__(
        self,
        image: str | np.ndarray | torch.Tensor | Image.Image,
        resize_to: int = 512,
        border_thresh: int = 10,
        face_crop_scale: float = 1.5,
        extra_input: bool = False,
    ) -> FaceOutput:

        image = self._load_image(image)

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        faces = self.app.get(frame)
        h, w, _ = frame.shape
        image_to_process = None

        if not faces:
            print(
                "[Warning] No face detected. Using the whole image, padded to square."
            )
            image_to_process = _pad_to_square(frame, pad_color=255)
        else:
            largest_face = max(
                faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            x1, y1, x2, y2 = map(int, largest_face.bbox)

            is_close_to_border = (
                x1 <= border_thresh
                and y1 <= border_thresh
                and x2 >= w - border_thresh
                and y2 >= h - border_thresh
            )

            if is_close_to_border:
                print(
                    "[Info] Face is close to border, padding original image to square."
                )
                image_to_process = _pad_to_square(frame, pad_color=255)
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                side = int(max(x2 - x1, y2 - y1) * face_crop_scale)
                half = side // 2

                left = max(cx - half, 0)
                top = max(cy - half, 0)
                right = min(cx + half, w)
                bottom = min(cy + half, h)

                cropped_face = frame[top:bottom, left:right]
                image_to_process = _pad_to_square(cropped_face, pad_color=255)

        image_resized = cv2.resize(
            image_to_process, (resize_to, resize_to), interpolation=cv2.INTER_AREA
        )

        face_tensor = (
            _img2tensor(image_resized, bgr2rgb=True).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            normalized_face = normalize(face_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            parsing_out = self.parsing_model(normalized_face)[0]
            parsing_mask = parsing_out.argmax(dim=1, keepdim=True)

        background_mask_np = (parsing_mask.squeeze().cpu().numpy() == 0).astype(
            np.uint8
        )
        white_background = np.ones_like(image_resized, dtype=np.uint8) * 255
        mask_3channel = cv2.cvtColor(background_mask_np * 255, cv2.COLOR_GRAY2BGR)
        result_img_bgr = np.where(mask_3channel == 255, white_background, image_resized)
        result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
        img_white_bg = PIL.Image.fromarray(result_img_rgb)
        if extra_input:
            # 2. Create image with transparent background (new logic)
            # Create an alpha channel: 255 for foreground (not background), 0 for background
            alpha_channel = (parsing_mask.squeeze().cpu().numpy() != 0).astype(
                np.uint8
            ) * 255

            # Convert the resized BGR image to RGB
            image_resized_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

            # Stack RGB channels with the new alpha channel
            rgba_image = np.dstack((image_resized_rgb, alpha_channel))

            # Create PIL image from the RGBA numpy array
            img_transparent_bg = Image.fromarray(rgba_image, "RGBA")

            return FaceOutput(face=img_white_bg, mask=img_transparent_bg)
        else:
            return FaceOutput(face=img_white_bg)
