from src.helpers.helpers import helpers
import torch
import json
import math
from typing import Dict, Any, Iterable, List, Optional
from PIL import Image
import ujson
from boltons.iterutils import remap
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TextStreamer
from src.helpers.base import BaseHelper
from src.utils.defaults import get_components_path
from src.mixins.cache_mixin import CacheMixin


def clean_json(caption):
    caption["pickascore"] = 1.0
    caption["aesthetic_score"] = 10.0
    caption = prepare_clean_caption(caption)
    return caption

def parse_aesthetic_score(record: dict) -> str:
    ae = record["aesthetic_score"]
    if ae < 5.5:
        return "very low"
    elif ae < 6:
        return "low"
    elif ae < 7:
        return "medium"
    elif ae < 7.6:
        return "high"
    else:
        return "very high"


def parse_pickascore(record: dict) -> str:
    ps = record["pickascore"]
    if ps < 0.78:
        return "very low"
    elif ps < 0.82:
        return "low"
    elif ps < 0.87:
        return "medium"
    elif ps < 0.91:
        return "high"
    else:
        return "very high"


def prepare_clean_caption(record: dict) -> str:
    def keep(p, k, v):
        is_none = v is None
        is_empty_string = isinstance(v, str) and v == ""
        is_empty_dict = isinstance(v, dict) and not v
        is_empty_list = isinstance(v, list) and not v
        is_nan = isinstance(v, float) and math.isnan(v)
        if is_none or is_empty_string or is_empty_list or is_empty_dict or is_nan:
            return False
        return True

    try:
        scores = {}
        if "pickascore" in record:
            scores["preference_score"] = parse_pickascore(record)
        if "aesthetic_score" in record:
            scores["aesthetic_score"] = parse_aesthetic_score(record)

        # Create structured caption dict of original values
        fields = [
            "short_description",
            "objects",
            "background_setting",
            "lighting",
            "aesthetics",
            "photographic_characteristics",
            "style_medium",
            "text_render",
            "context",
            "artistic_style",
        ]

        original_caption_dict = {f: record[f] for f in fields if f in record}

        # filter empty values recursivly (i.e. None, "", {}, [], float("nan"))
        clean_caption_dict = remap(original_caption_dict, visit=keep)

        # Set aesthetics scores
        if "aesthetics" not in clean_caption_dict:
            if len(scores) > 0:
                clean_caption_dict["aesthetics"] = scores
        else:
            clean_caption_dict["aesthetics"].update(scores)

        # Dumps clean structured caption as minimal json string (i.e. no newlines\whitespaces seps)
        clean_caption_str = ujson.dumps(clean_caption_dict, escape_forward_slashes=False)
        return clean_caption_str
    except Exception as ex:
        print("Error: ", ex)
        raise ex


def _collect_images(messages: Iterable[Dict[str, Any]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            image_value = item.get("image")
            if isinstance(image_value, Image.Image):
                images.append(image_value)
            else:
                raise ValueError("Expected PIL.Image for image content in messages.")
    return images


def _strip_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> str:
    if not stop_sequences:
        return text.strip()
    cleaned = text
    for stop in stop_sequences:
        if not stop:
            continue
        index = cleaned.find(stop)
        if index >= 0:
            cleaned = cleaned[:index]
    return cleaned.strip()


@helpers("fibo.prompt_gen")
class PromptGenHelper(BaseHelper, CacheMixin):
    """Inference wrapper using Hugging Face transformers."""

    def __init__(
        self,
        model_path: str,
        save_path: str = get_components_path(),
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super(PromptGenHelper, self).__init__()
        default_processor_kwargs: Dict[str, Any] = {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1024 * 28 * 28,
        }
        processor_kwargs = {**default_processor_kwargs, **(processor_kwargs or {})}
        model_kwargs = model_kwargs or {}
        
        model_path = self._download(model_path, save_path)
        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()
        self.to_device(self.model)

        tokenizer_obj = self.processor.tokenizer
        if tokenizer_obj.pad_token_id is None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
        self._pad_token_id = tokenizer_obj.pad_token_id
        eos_token_id = tokenizer_obj.eos_token_id
        if isinstance(eos_token_id, list) and eos_token_id:
            self._eos_token_id = eos_token_id
        elif eos_token_id is not None:
            self._eos_token_id = [eos_token_id]
        else:
            raise ValueError("Tokenizer must define an EOS token for generation.")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> str:
        prompt_hash = self.hash_prompt(locals())
        cached_prompt, cached_attention_mask = self.load_cached_prompt(prompt_hash)
        if cached_prompt is not None:
            return cached_prompt
        
        tokenizer = self.processor.tokenizer
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_inputs: Dict[str, Any] = {
            "text": [prompt_text],
            "padding": True,
            "return_tensors": "pt",
        }
        images = _collect_images(messages)
        if images:
            processor_inputs["images"] = images
        
        inputs = self.processor(**processor_inputs)
        device = self.model.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._pad_token_id,
        }
        
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Processor did not return input_ids; cannot compute new tokens.")

        with torch.inference_mode():
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generated_ids = self.model.generate(**inputs, **generation_kwargs, streamer=streamer)

        new_token_ids = generated_ids[:, input_ids.shape[-1] :]
        decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        if not decoded:
            return ""
        text = decoded[0]
        stripped_text = _strip_stop_sequences(text, stop)
        json_prompt = json.loads(stripped_text)
        return prepare_clean_caption(json_prompt)