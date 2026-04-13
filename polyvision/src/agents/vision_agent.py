import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
import asyncio
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Adjusted imports to work when run via `python3 src/main.py`
from models import FramePayload

class VisionAgent:
    def __init__(self, video_sources: list, inference_temp: float, out_ui_queue: asyncio.Queue, out_iot_queue: asyncio.Queue):
        self.video_sources = video_sources
        self.inference_temp = inference_temp
        self.out_ui_queue = out_ui_queue
        self.out_iot_queue = out_iot_queue
        
        self.model_id = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VisionAgent] Initializing on {self.device}")
        
    def _load_model(self):
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
        self.processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self.id2label = self.model.config.id2label
        self.num_classes = len(self.id2label)
        self.model.to(self.device).eval()
        
    def _clean_frame(self, bgr_frame):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    def _infer(self, rgb):
        h, w, _ = rgb.shape
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits / self.inference_temp
            # Bicubic interpolation as established in earlier script
            seg_mask = F.interpolate(logits, size=(h, w), mode="bicubic").argmax(1)[0].cpu().numpy()
        return seg_mask

    async def run(self):
        print("[VisionAgent] Loading Model...")
        await asyncio.to_thread(self._load_model)
        print(f"[VisionAgent] Model Loaded. Classes: {self.num_classes}")
        
        source_index = 0
        cap = None
        frame_counter = 0
        
        while True:
            if not self.video_sources:
                print("[VisionAgent] CRITICAL: No video sources available. Waiting 10s...")
                await asyncio.sleep(10)
                continue

            current_path = self.video_sources[source_index]
            
            # Connection logic
            if cap is None or not cap.isOpened():
                print(f"[VisionAgent] Attempting connection to source [{source_index}]: {current_path}")
                cap = cv2.VideoCapture(current_path)
                
                if not cap.isOpened():
                    print(f"[VisionAgent] connection failed. Rotating source...")
                    source_index = (source_index + 1) % len(self.video_sources)
                    cap = None
                    await asyncio.sleep(5)
                    continue
                else:
                    print(f"[VisionAgent] Successfully opened: {current_path}")

            success, bgr = cap.read()
            
            if not success:
                print(f"[VisionAgent] Stream ended or connection lost: {current_path}")
                cap.release()
                cap = None
                # Rotate to next source (effectively retries primary after secondary ends/fails)
                source_index = (source_index + 1) % len(self.video_sources)
                await asyncio.sleep(2)
                continue
                
            ts = time.time()
            
            # Offloading heavy CV/ML to thread pool so it doesn't block the Asyncio event loop used by other agents
            rgb = await asyncio.to_thread(self._clean_frame, bgr)
            seg_mask = await asyncio.to_thread(self._infer, rgb)
            
            unique_ids = list(np.unique(seg_mask))
            
            payload = FramePayload(
                timestamp=ts,
                original_bgr=bgr,
                cleaned_rgb=rgb,
                segmentation_mask=seg_mask,
                unique_ids=unique_ids
            )
            
            # Dispatch to downstream queues. If queue is full, we log & drop frame to maintain real-time edge streaming.
            if not self.out_ui_queue.full():
                await self.out_ui_queue.put((payload, self.id2label, self.num_classes))
            else:
                print("[VisionAgent] UI Queue Full. Dropping Frame.")
                
            if not self.out_iot_queue.full():
                # Pass data to IoT bridge
                await self.out_iot_queue.put(payload)
                
            frame_counter += 1
            await asyncio.sleep(0) # Yield control loop
            
        cap.release()
