import asyncio
import io
import matplotlib
matplotlib.use('Agg') # Headless execution for Edge devices
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

class VisualFormatter:
    def __init__(self, in_queue: asyncio.Queue, out_iot_queue: asyncio.Queue):
        self.in_queue = in_queue
        self.out_iot_queue = out_iot_queue

    def _render_frame(self, payload, id2label, num_classes):
        # We replicate the legending logic here
        fig, axes = plt.subplots(1, 2, figsize=(16, 8)) 
        cmap = cm.get_cmap('nipy_spectral', num_classes)
        
        axes[0].imshow(payload.cleaned_rgb)
        axes[0].set_title(f"Cleaned Raw @ {payload.timestamp:.1f}s", fontsize=14)
        
        axes[1].imshow(payload.segmentation_mask, cmap=cmap, vmin=0, vmax=num_classes-1)
        axes[1].set_title(f"SegFormer Pred | IDs: {payload.unique_ids}", fontsize=14)
        for ax in axes: ax.axis('off')

        # Create Legend
        patches = []
        for i in sorted(payload.unique_ids):
            label_name = id2label.get(int(i), f"Unknown_{i}")
            color = cmap(i / (num_classes - 1))
            patches.append(mpatches.Patch(color=color, label=f"ID {i}: {label_name}"))
        
        fig.legend(handles=patches, loc='center right', title="Dynamic Taxonomy", 
                   fontsize='small', frameon=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=90) # Lower DPI to save network bandwidth
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    async def run(self):
        print("[VisualFormatter] Started.")
        while True:
            # We receive payload, id2label, num_classes from vision agent
            item = await self.in_queue.get()
            if isinstance(item, tuple):
                payload, id2label, num_classes = item
            else:
                self.in_queue.task_done()
                continue

            try:
                # CPU bound formatting
                img_bytes = await asyncio.to_thread(self._render_frame, payload, id2label, num_classes)
                payload.rendered_image_bytes = img_bytes
                
                # Pass to IOT agent to broadcast to Home Assistant
                if not self.out_iot_queue.full():
                    await self.out_iot_queue.put(payload)
                else:
                    print("[VisualFormatter] IoT Queue full, discarding formatted frame.")
            except Exception as e:
                print(f"[VisualFormatter] Error formatting frame: {e}")
            finally:
                self.in_queue.task_done()
