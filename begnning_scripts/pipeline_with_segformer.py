import os
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
# LENS 1: SOFTWARE ENGINEERING (Headless enforcement for Org-babel)
matplotlib.use('Agg') 

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

class VideoConfig:
    INPUT_VIDEO = "/home/nate/NextCloud/Roam/Classes/Capstone/data/GOPR1176_1770835361281.MP4"
    OUTPUT_FOLDER = "segformer_stability_results"
    SEG_MODEL_ID = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024" 
    
    # We no longer hardcode CORAL_MAP here. We will pull it from the model config.
    SAMPLE_EVERY_SEC = 5.0
    MAX_FRAMES = 5
    IMG_SIZE = 512 
    INFERENCE_TEMP = 1.5 

# ==========================================
# LENS 2: ARCHITECTURE (Dynamic Label Sourcing)
# ==========================================
def get_stabilized_model():
    print(f"--- Fetching Model & Metadata: {VideoConfig.SEG_MODEL_ID} ---")
    
    model = SegformerForSemanticSegmentation.from_pretrained(VideoConfig.SEG_MODEL_ID)
    processor = SegformerImageProcessor.from_pretrained(VideoConfig.SEG_MODEL_ID)
    
    # SOURCE OF TRUTH: Pull the mapping directly from the model
    # This ensures ID 7 is ALWAYS what the model thinks is 'human'
    id2label = model.config.id2label
    num_classes = len(id2label)
    
    return model, processor, id2label, num_classes

# ==========================================
# LENS 3: COMPUTER VISION (Cleaning)
# ==========================================
def clean_underwater_frame(bgr_frame):
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

# ==========================================
# LENS 4: VISUALIZATION (Metadata-Linked Legend)
# ==========================================
def save_visual_result(rgb, seg, ts, path, id2label, num_classes):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10)) 
    # Use a colormap that spans the ACTUAL number of classes in the model
    cmap = cm.get_cmap('nipy_spectral', num_classes)
    
    axes[0].imshow(rgb)
    axes[0].set_title(f"GoPro Raw Cleaned @ {ts:.1f}s", fontsize=14)
    
    # vmin/vmax ensure the colors are consistent across frames
    axes[1].imshow(seg, cmap=cmap, vmin=0, vmax=num_classes-1)
    
    # LENS 5: METRICS (ID Audit in the Title)
    unique_ids = np.unique(seg)
    axes[1].set_title(f"SegFormer Stable Pred | Unique IDs: {list(unique_ids)}", fontsize=14)
    
    for ax in axes: ax.axis('off')

    # Create legend only for classes actually present in THIS frame
    patches = []
    for i in sorted(unique_ids):
        label_name = id2label.get(int(i), f"Unknown_{i}")
        color = cmap(i / (num_classes - 1))
        patches.append(mpatches.Patch(color=color, label=f"ID {i}: {label_name}"))
    
    fig.legend(handles=patches, loc='center right', title="Dynamic Taxonomy", 
               fontsize='small', frameon=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0, 0.82, 1]) 
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def run_synthesis_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    os.makedirs(VideoConfig.OUTPUT_FOLDER, exist_ok=True)
    
    model, processor, id2label, num_classes = get_stabilized_model()
    model.to(device).eval()
    
    cap = cv2.VideoCapture(VideoConfig.INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    saved = 0
    while saved < VideoConfig.MAX_FRAMES:
        target_frame = int(saved * VideoConfig.SAMPLE_EVERY_SEC * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, bgr = cap.read()
        if not success: break
        
        ts = target_frame / fps
        print(f"Analyzing {ts:.1f}s with Metadata Alignment...", end=" ", flush=True)
        t0 = time.time()
        
        rgb = clean_underwater_frame(bgr)
        h, w, _ = rgb.shape
        
        with torch.no_grad():
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            logits = model(**inputs).logits / VideoConfig.INFERENCE_TEMP
            # Using bilinear/bicubic to match your script's preference
            seg_mask = F.interpolate(logits, size=(h, w), mode="bicubic").argmax(1)[0].cpu().numpy()
        
        out_name = os.path.join(VideoConfig.OUTPUT_FOLDER, f"metadata_frame_{ts:.1f}s.png")
        save_visual_result(rgb, seg_mask, ts, out_name, id2label, num_classes)
        
        print(f"Done ({time.time()-t0:.2f}s)")
        saved += 1
        
    cap.release()
    print(f"\nPipeline complete. Metadata-aligned results in '{VideoConfig.OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    run_synthesis_pipeline()
