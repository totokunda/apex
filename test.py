import torch
import ffmpeg
import torchvision
def ffmpeg_i2v(image_path, w=384, h=224, aspect_policy="fit"):
    r = ffmpeg.input("pipe:0", format="image2pipe")
    if aspect_policy == "crop":
        r = r.filter("scale", w, h, force_original_aspect_ratio="increase").filter("crop", w, h)
    elif aspect_policy == "pad":
        r = r.filter("scale", w, h, force_original_aspect_ratio="decrease").filter(
            "pad", w, h, "(ow-iw)/2", "(oh-ih)/2", color="black"
        )
    elif aspect_policy == "fit":
        r = r.filter("scale", w, h)
    else:
        print(f"Unknown aspect policy: {aspect_policy}, using fit as fallback")
        r = r.filter("scale", w, h)
    image_byte = open(image_path, "rb").read()
    try:
        out, _ = r.output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1).run(
            input=image_byte, capture_stdout=True, capture_stderr=True
        )
    except ffmpeg.Error as e:
        print(f"Error occurred: {e.stderr.decode()}")
        raise e

    video = torch.frombuffer(out, dtype=torch.uint8).view(1, h, w, 3)
    return video


def ffmpeg_v2v(video_path, fps, w=384, h=224, prefix_frame=None, prefix_video_max_chunk=5):
    if video_path is None:
        return None
    out, _ = (
        ffmpeg.input(video_path, ss=0, format="mp4")
        .filter("fps", fps=fps)
        .filter("scale", w, h)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", nostdin=None)
        .run(capture_stdout=True, capture_stderr=True)
    )

    video = torch.frombuffer(out, dtype=torch.uint8).view(-1, h, w, 3)

    if prefix_frame is not None:
        return video[:prefix_frame]
    else:
        num_frames_to_read = video.shape[0]
        if num_frames_to_read < fps:
            clip_length = 1
        else:
            PREFIX_VIDEO_MAX_FRAMES = prefix_video_max_chunk * fps
            clip_length = min(num_frames_to_read // fps * fps, PREFIX_VIDEO_MAX_FRAMES)
        return video[-clip_length:]
    
    
    
img = "/workspace/apex/assets/images/IMG_6954.JPG"

img_tensor = ffmpeg_i2v(img).squeeze(0).permute(2, 0, 1)
print(img_tensor.shape)
# save img_tensor as png
torchvision.io.write_png(img_tensor, "img.png")
