## Florence-2 + SAM 2 pipeline for Open Vocabulary Object Detection and Caption Grounding

This repository utilizes the Florence-2 and Segment Anything Model 2 (SAM-2) to perform:

- **Open Vocabulary Object Detection**: Detect any object in images by specifying object classes in natural language
- **Caption Grounding with Masks**: Generate detailed captions and ground phrases to regions in the image with segmentation masks
- **Video Object Tracking**: Detect objects in video and track them with propagated segmentation masks


---

## 🔍 Model Information

The script uses two main models:

### Florence-2
Florence-2 is a multimodal large language model that enables:
- Detailed image captioning
- Caption to phrase grounding
- Open vocabulary detection

### Segment Anything Model 2 (SAM-2)
SAM-2 provides precise segmentation masks for detected objects in:
- Images (SAM-2 Image Model)
- Videos (SAM-2 Video Model with temporal propagation)

---

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Required packages (see Installation section)


---


## 🔧 Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/MjdMahasneh/florance2-sam2.git
    cd florance2-sam2
    ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model weights and the `.YAML` config files (find in `hugging-face -> files`, following the links below) and place them in the `checkpoints` and `configs` directories, respectively.
   
   | Model Name    | Link                                                                 |
   |---------------|----------------------------------------------------------------------|
   | SAM base-plus | [Link](https://huggingface.co/facebook/sam2-hiera-base-plus)        |
   | SAM large     | [Link](https://huggingface.co/facebook/sam2-hiera-large)             |
   | SAM small     | [Link](https://huggingface.co/facebook/sam2-hiera-small)             |
   | SAM tiny      | [Link](https://huggingface.co/facebook/sam2-hiera-tiny)              |



---

## 🏗️ Architecture

The tool is organized into several utility modules:

```
├── utils/
│   ├── florence.py  # Florence model loading and inference
│   ├── sam.py       # SAM image and video model utilities
│   ├── modes.py     # Processing mode constants
│   └── video.py     # Video utility functions
├── main.py          # Main script with processing functions
└── tmp/             # Temporary processing directory
```


---

## 💻 Usage Examples

To use, modify `main.py` to include the mode needed, and provide an image and a text prompt that specifies the objects you want to detect. The function will return the processed image with detected objects highlighted.


### Example 1. Image Open Vocabulary Detection



Detect specific objects in an image by providing their class names.

```python
from utils.modes import IMAGE_OPEN_VOCABULARY_DETECTION_MODE

# Example
result_img, _ = process_image(
    mode=IMAGE_OPEN_VOCABULARY_DETECTION_MODE,
    image_path="./samples/image.png",
    text_input="dog, cat, person",
    output_path="./samples/output_detection.png"
)
```

### Example 2. Image Caption Grounding with Masks

Generate a detailed caption for the image and ground the phrases to specific regions with segmentation masks.

```python
from utils.modes import IMAGE_CAPTION_GROUNDING_MASKS_MODE

# Example
result_img, caption = process_image(
    mode=IMAGE_CAPTION_GROUNDING_MASKS_MODE,
    image_path="./samples/scenery.jpg",
    output_path="./samples/output_caption.jpg"
)
print(f"Generated caption: {caption}")
```

### Example 3. Video Object Tracking with Mask Propagation

Detect objects in the first frame of a video and track them through the entire video with segmentation masks.

```python
# Example
result_video_path = process_video(
    video_path="./samples/soccer.mp4",
    text_input="player in white outfit, player in black outfit, ball",
    output_path="./samples/soccer_tracked.mp4"
)
```



---

## 🧩 Extending the Tool

To add new processing modes:
1. Add a new mode constant in `utils/modes.py`
2. Add a new conditional branch in the `process_image()` function
3. Implement the required processing logic

---


### Reference:
- [Florance-2+SAM2 Hugging face Space](https://huggingface.co/spaces/SkalskiP/florence-sam) (Special thanks).
- [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2).
- [How to segment images with SAM-2](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-images-with-sam-2.ipynb).
- [What-is SAM-2](https://blog.roboflow.com/what-is-segment-anything-2/).
- [Segment Anything 2 (SAM 2): Meta AI's Newest Model | Community Q&A (Jul 30)](https://www.youtube.com/watch?v=Dv003fTyO-Y).

  

