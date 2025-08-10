
from src.preprocess.composition.composition import ReferenceAnythingPreprocessor
from diffusers.utils import export_to_video
import os 

preprocessor = ReferenceAnythingPreprocessor()
test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)

video = "assets/video/conversation.mp4"
mask = "assets/mask/conversation_mask_full.mp4"

RUN_TESTS = [
    "maskbboxtrack",
    "maskpointtrack",
    "bboxtrack"
]

if "salient" in RUN_TESTS:
    output_salient = preprocessor(
        video=video,
        mode="salient",
        return_mask=True
    )

    mask_salient = output_salient.mask

    export_to_video(mask_salient, os.path.join(test_dir, "conversation_salient_mask.mp4"), fps=24)

    print("Finished salient test")
else:
    print("Skipping salient test")
    
if "mask" in RUN_TESTS:
    output_mask = preprocessor(
        video=video,
        mask=mask,
        mode="mask",
        return_mask=True
    )
    mask_mask = output_mask.mask
    export_to_video(mask_mask, os.path.join(test_dir, "conversation_mask_mask.mp4"), fps=24)

    print("Finished mask test")
else:
    print("Skipping mask test")
    
if "bbox" in RUN_TESTS:
    bbox = [0.2, 0.2, 0.8, 0.8]

    output_bbox = preprocessor(
        video=video,
        mode="bbox",
        return_mask=True,
        bbox=bbox
    )
    
    mask_bbox = output_bbox.mask

    export_to_video(mask_bbox, os.path.join(test_dir, "conversation_bbox_1_mask.mp4"), fps=24)  
    print("Finished bbox test 1")  

    bbox = [100, 100, 200, 200]

    output_bbox = preprocessor(
        video=video,
        mode="bbox",
        return_mask=True,
        bbox=bbox
    )

    video_bbox = output_bbox.video
    mask_bbox = output_bbox.mask

    export_to_video(mask_bbox, os.path.join(test_dir, "conversation_bbox_2_mask.mp4"), fps=24)  
    print("Finished bbox test 2")  
else:
    print("Skipping bbox tests")

if "label" in RUN_TESTS:
    output_label = preprocessor(
        video=video,
        mode="label",
        return_mask=True,
        label="person"
    )

    mask_label = output_label.mask

    export_to_video(mask_label, os.path.join(test_dir, "conversation_label_mask.mp4"), fps=24)

    print("Finished label test")
else:
    print("Skipping label test")

if "caption" in RUN_TESTS:
    output_caption = preprocessor(
        video=video,
        mode="caption",
        return_mask=True,
        caption="Two people talking"
    )

    mask_caption = output_caption.mask

    export_to_video(mask_caption, os.path.join(test_dir, "conversation_caption_mask.mp4"), fps=24)
    print("Finished caption test")
else:
    print("Skipping caption test")

if "salientmasktrack" in RUN_TESTS:
    output_salientmasktrack = preprocessor(
        video=video,
        mode="salientmasktrack",
        return_mask=True
    )
    
    mask_salientmasktrack = output_salientmasktrack.mask
    
    export_to_video(mask_salientmasktrack, os.path.join(test_dir, "conversation_salientmasktrack_mask.mp4"), fps=24)
    
    print("Finished salientmasktrack test")
else:
    print("Skipping salientmasktrack test")

if "salientbboxtrack" in RUN_TESTS:
    output_salientbboxtrack = preprocessor(
        video=video,
        mode="salientbboxtrack",
        return_mask=True
    )
    
    mask_salientbboxtrack = output_salientbboxtrack.mask
    
    export_to_video(mask_salientbboxtrack, os.path.join(test_dir, "conversation_salientbboxtrack_mask.mp4"), fps=24)
    
    print("Finished salientbboxtrack test")
else:
    print("Skipping salientbboxtrack test")
    

if "masktrack" in RUN_TESTS:
    output_masktrack = preprocessor(
        video=video,
        mask=mask,
        mode="masktrack",
        return_mask=True
    )
    
    mask_masktrack = output_masktrack.mask
    
    export_to_video(mask_masktrack, os.path.join(test_dir, "conversation_masktrack_mask.mp4"), fps=24)
    
    print("Finished masktrack test")
else:
    print("Skipping masktrack test")
    

if "maskbboxtrack" in RUN_TESTS:
    output_maskbboxtrack = preprocessor(
        video=video,
        mask=mask,
        mode="maskbboxtrack",
        return_mask=True
    )   

    mask_maskbboxtrack = output_maskbboxtrack.mask
    
    export_to_video(mask_maskbboxtrack, os.path.join(test_dir, "conversation_maskbboxtrack_mask.mp4"), fps=24)
    
    print("Finished maskbboxtrack test")
else:
    print("Skipping maskbboxtrack test")

if "maskpointtrack" in RUN_TESTS:
    output_maskpointtrack = preprocessor(
        video=video,
        mask=mask,
        mode="maskpointtrack",
        return_mask=True
    )
    
    mask_maskpointtrack = output_maskpointtrack.mask
    
    export_to_video(mask_maskpointtrack, os.path.join(test_dir, "conversation_maskpointtrack_mask.mp4"), fps=24)
    
    print("Finished maskpointtrack test")
else:
    print("Skipping maskpointtrack test")
    
if "bboxtrack" in RUN_TESTS:
    bbox = [100, 100, 200, 200]
    
    output_bboxtrack = preprocessor(
        video=video,
        bbox=bbox,
        mode="bboxtrack",
        return_mask=True
    )
    
    mask_bboxtrack = output_bboxtrack.mask
    
    export_to_video(mask_bboxtrack, os.path.join(test_dir, "conversation_bboxtrack_mask.mp4"), fps=24)
    
    print("Finished bboxtrack test")
else:
    print("Skipping bboxtrack test")
    


    
    