import asyncio
import os
import sys

# We'll create these modules next
from agents.vision_agent import VisionAgent
from agents.iot_bridge import IOTBridge
from agents.visual_formatter import VisualFormatter

async def main():
    print("--- PolyVision Orchestrator Booting ---")
    
    # 1. Setup Inter-Agent Communication (Asyncio Queues)
    # The maxsize prevents the Vision agent from overwhelming the downstream if it inferences faster than they can process.
    vision_to_ui_queue = asyncio.Queue(maxsize=5)
    vision_to_iot_queue = asyncio.Queue(maxsize=5)
    ui_to_iot_queue = asyncio.Queue(maxsize=5) 
    
    # 2. Extract configuration from environment
    # Fallback helps with local testing outside of Docker if needed
    base_path = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(base_path, "assets", "stock_video.mp4")
    
    input_video = os.getenv("INPUT_VIDEO", default_video)
    inference_temp = float(os.getenv("INFERENCE_TEMP", "1.5"))
    
    print(f"[Orchestrator] Input Source: {input_video}")
    
    # 3. Initialize Agents
    vision_agent = VisionAgent(
        video_path=input_video, 
        inference_temp=inference_temp,
        out_ui_queue=vision_to_ui_queue, 
        out_iot_queue=vision_to_iot_queue
    )
    
    ui_agent = VisualFormatter(
        in_queue=vision_to_ui_queue,
        out_iot_queue=ui_to_iot_queue
    )
    
    iot_agent = IOTBridge(
        in_vision_queue=vision_to_iot_queue,
        in_ui_queue=ui_to_iot_queue
    )
    
    # 4. Start concurrent execution
    try:
        await asyncio.gather(
            vision_agent.run(),
            ui_agent.run(),
            iot_agent.run()
        )
    except Exception as e:
        print(f"[Orchestrator] Unhandled exception in MAS: {e}")
        # Signal shutdown?

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
        sys.exit(0)
