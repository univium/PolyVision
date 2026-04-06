from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict

@dataclass
class FramePayload:
    """
    The core data packet passed between agents.
    It travels from the Vision Agent -> UI Visualizer / IoT Bridge.
    """
    timestamp: float
    original_bgr: np.ndarray 
    cleaned_rgb: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None
    
    # Optional metadata populated by Vision Agent
    unique_ids: Optional[list] = None
    
    # To be populated by Visual Formatter if publishing image directly to MQTT
    rendered_image_bytes: Optional[bytes] = None
