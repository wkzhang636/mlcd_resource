# Example 
<img src="updownfunk_mask.gif" alt="output" width="1024"/>  

## How to use
If you just want to use this code, please refer to this sample below
```python
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch
from torchvision import transforms
import subprocess
import os

# video path
video_path = "updownfunk.mp4"
input_dir = "frames"
output_dir = "mask_frames"
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# assert you have ffmpeg installed, mp4 -> jpg
cmd = [
    "ffmpeg",
    "-i", video_path,
    "-vf", "fps=30",    # 30FPS
    "-qscale:v", "1",  
    os.path.join(input_dir, "frame_%04d.jpg") 
]
subprocess.run(cmd)

# model path

model_path = "/root/MLCD-Seg/" # or use your local path
mlcd_seg = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)


# read jpgs
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

for idx, filename in enumerate(image_files, start=1):

    src_path = os.path.join(input_dir, filename)
    seg_img = Image.open(src_path).convert('RGB')

    seg_prompt = "This <video> depicts a group of people dancing.\nCould you provide a segmentation mask for the man in pink suit?"
    pred_mask = mlcd_seg.predict_forward(seg_img, seg_prompt, tokenizer, force_seg=True)

    # Mask visualization
    pred_mask = pred_mask.squeeze(0).cpu()
    pred_mask = (pred_mask > 0.5).float()
    img_tensor = transforms.ToTensor()(seg_img)
    alpha = 0.2  # 20% transparency
    red_mask = torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1).to(img_tensor.device)  # green mask
    black_bg = torch.zeros_like(img_tensor)  # black background
    masked_area = red_mask * alpha + img_tensor * (1 - alpha)
    background = black_bg * alpha + img_tensor * (1 - alpha)
    combined = torch.where(pred_mask.unsqueeze(0).bool(), masked_area, background)
    combined = combined.cpu()  # [3, H, W], CPU

    # Save masked jpgs
    new_name = f"{idx:04d}{os.path.splitext(filename)[1]}"
    dst_path = os.path.join(output_dir, new_name)
    transforms.ToPILImage()(combined.clamp(0, 1)).save(dst_path)

cmd = [
    "ffmpeg",
    "-y",  
    "-framerate", str(30),  # fps
    "-i", os.path.join(output_dir, "%04d.jpg"), 
    "-c:v", "libx264",
    "-crf", str(23), 
    "-pix_fmt", "yuv420p", 
    "-vf", "fps=" + str(23), 
    "updownfunk_mask.mp4"  # output video
]
# jpgs -> mp4    
subprocess.run(cmd, check=True)
```

