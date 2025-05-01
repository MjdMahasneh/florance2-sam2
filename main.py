import os
import torch
import numpy as np
import cv2
import supervision as sv
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Optional, List

from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_DETAILED_CAPTION_TASK, \
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.modes import IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE
from utils.sam import load_sam_image_model, run_sam_inference, load_sam_video_model
from utils.video import generate_unique_name, create_directory, delete_directory

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Constants
VIDEO_SCALE_FACTOR = 0.5
VIDEO_TARGET_DIRECTORY = "tmp"
create_directory(directory_path=VIDEO_TARGET_DIRECTORY)

# Use CUDA if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable optimizations for CUDA
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Load models
print("Loading models...")
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)
SAM_VIDEO_MODEL = load_sam_video_model(device=DEVICE)

# Setup visualization
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)


def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(
    mode: str, 
    image_path: str, 
    text_input: str = None,
    output_path: str = None
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Process an image using either open vocabulary detection or caption grounding.
    
    Args:
        mode: The processing mode (open vocabulary detection or caption grounding)
        image_path: Path to the input image
        text_input: Text prompt for open vocabulary detection (comma-separated classes)
        output_path: Path to save the output image (optional)
    
    Returns:
        Tuple of (processed image, caption)
    """
    print(f"Processing image in {mode} mode...")
    
    # Load image
    image_input = Image.open(image_path).convert("RGB")
    
    if mode == IMAGE_OPEN_VOCABULARY_DETECTION_MODE:
        if not text_input:
            print("Error: Text prompt required for open vocabulary detection.")
            return None, None

        texts = [prompt.strip() for prompt in text_input.split(",")]
        print(f"Detecting objects: {texts}")
        
        detections_list = []
        for text in texts:
            _, result = run_florence_inference(
                model=FLORENCE_MODEL,
                processor=FLORENCE_PROCESSOR,
                device=DEVICE,
                image=image_input,
                task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                text=text
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_input.size
            )
            detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
            detections_list.append(detections)

        detections = sv.Detections.merge(detections_list)
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        output_image = annotate_image(image_input, detections)
        
        if output_path:
            output_image.save(output_path)
            print(f"Saved output to {output_path}")
            
        return output_image, None

    elif mode == IMAGE_CAPTION_GROUNDING_MASKS_MODE:
        print("Generating detailed caption and grounding...")
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_DETAILED_CAPTION_TASK
        )
        caption = result[FLORENCE_DETAILED_CAPTION_TASK]
        print(f"Generated caption: {caption}")
        
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK,
            text=caption
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        output_image = annotate_image(image_input, detections)
        
        if output_path:
            output_image.save(output_path)
            print(f"Saved output to {output_path}")
            
        return output_image, caption
    
    else:
        print(f"Error: Unknown mode {mode}")
        return None, None


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_video(
    video_path: str, 
    text_input: str,
    output_path: str = None
) -> Optional[str]:
    """
    Process a video with open vocabulary detection and mask propagation.
    
    Args:
        video_path: Path to the input video
        text_input: Comma-separated text prompts for object detection
        output_path: Path to save the output video (optional)
        
    Returns:
        Path to the processed video
    """
    print("Processing video...")
    
    if not text_input:
        print("Error: Text prompt required for video processing.")
        return None

    # Get first frame to detect objects
    frame_generator = sv.get_video_frames_generator(video_path)
    frame = next(frame_generator)
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    texts = [prompt.strip() for prompt in text_input.split(",")]
    print(f"Detecting objects: {texts}")
    
    detections_list = []
    for text in texts:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=frame,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=frame.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)
        detections_list.append(detections)

    detections = sv.Detections.merge(detections_list)
    detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)

    if len(detections.mask) == 0:
        print(f"No objects of classes {text_input} found in the first frame of the video.")
        return None

    # Setup temporary directory for frames
    name = generate_unique_name()
    frame_directory_path = os.path.join(VIDEO_TARGET_DIRECTORY, name)
    frames_sink = sv.ImageSink(
        target_dir_path=frame_directory_path,
        image_name_pattern="{:05d}.jpeg"
    )

    # Get video info and scale it
    video_info = sv.VideoInfo.from_video_path(video_path)
    video_info.width = int(video_info.width * VIDEO_SCALE_FACTOR)
    video_info.height = int(video_info.height * VIDEO_SCALE_FACTOR)

    # Split video into frames
    frames_generator = sv.get_video_frames_generator(video_path)
    print("Splitting video into frames...")
    with frames_sink:
        for frame in tqdm(
                frames_generator,
                total=video_info.total_frames
        ):
            frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
            frames_sink.save_image(frame)

    # Initialize SAM video model
    print("Propagating masks through video...")
    inference_state = SAM_VIDEO_MODEL.init_state(
        video_path=frame_directory_path,
        device=DEVICE
    )

    # Add masks to track
    for mask_index, mask in enumerate(detections.mask):
        _, object_ids, mask_logits = SAM_VIDEO_MODEL.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=mask_index,
            mask=mask
        )

    # Set output path
    if output_path is None:
        video_output_path = os.path.join(VIDEO_TARGET_DIRECTORY, f"{name}.mp4")
    else:
        video_output_path = output_path

    # Process video frames
    frames_generator = sv.get_video_frames_generator(video_path)
    masks_generator = SAM_VIDEO_MODEL.propagate_in_video(inference_state)
    
    with sv.VideoSink(video_output_path, video_info=video_info) as sink:
        for frame, (_, tracker_ids, mask_logits) in zip(frames_generator, masks_generator):
            frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
            masks = (mask_logits > 0.0).cpu().numpy().astype(bool)
            if len(masks.shape) == 4:
                masks = np.squeeze(masks, axis=1)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                class_id=np.array(tracker_ids)
            )
            annotated_frame = frame.copy()
            annotated_frame = MASK_ANNOTATOR.annotate(
                scene=annotated_frame, detections=detections)
            annotated_frame = BOX_ANNOTATOR.annotate(
                scene=annotated_frame, detections=detections)
            sink.write_frame(annotated_frame)

    # Clean up
    delete_directory(frame_directory_path)
    print(f"Video processing complete. Output saved to {video_output_path}")
    return video_output_path


def main():
    """
    Example usage of the image and video processing functions.
    Modify these values to process your own files.
    """
    # EXAMPLE IMAGE PROCESSING - OPEN VOCABULARY DETECTION
    # --------------------------------------------------
    image_path = "./samples/image.png"  # Replace with your image path
    text_prompt = "dog, cat"  # Replace with your detection classes
    output_path = "./samples/output.png"  # Replace with your desired output path
    
    ## Uncomment to run image detection
    result_img, _ = process_image(
        mode=IMAGE_OPEN_VOCABULARY_DETECTION_MODE,
        image_path=image_path,
        text_input=text_prompt,
        output_path=output_path
    )
    
    # EXAMPLE IMAGE PROCESSING - CAPTION GROUNDING
    # --------------------------------------------------
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    output_path = "output_caption.jpg"  # Replace with your desired output path
    
    # Uncomment to run caption grounding
    # result_img, caption = process_image(
    #     mode=IMAGE_CAPTION_GROUNDING_MASKS_MODE,
    #     image_path=image_path,
    #     output_path=output_path
    # )
    # print(f"Generated caption: {caption}")
    
    # EXAMPLE VIDEO PROCESSING
    # --------------------------------------------------
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    text_prompt = "player in white outfit, player in black outfit, ball"  # Replace with detection classes
    output_path = "output_video.mp4"  # Replace with your desired output path
    
    # Uncomment to run video processing
    # result_video_path = process_video(
    #     video_path=video_path,
    #     text_input=text_prompt,
    #     output_path=output_path
    # )


if __name__ == "__main__":
    main()
