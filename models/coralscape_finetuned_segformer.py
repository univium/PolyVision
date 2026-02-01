#!/usr/bin/env python3
from dataset import load_dataset
import json
from huggingface_hub import hf_hub_download
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
import torch
from torch import nn
import evaluate



# Checking if CUDA GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loading and spliting the data by pre-sets
ds = load_dataset("EPFL-ECEO/coralscapes")

train_ds = ds["train"]
val_ds   = ds["validation"]  
test_ds  = ds["test"]

# Extracting the number of labels and readable IDs for segmenation later on.
hf_dataset_identifier = "EPFL-ECEO/coralscapes"
repo_id = f"datasets/{hf_dataset_identifier}"
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)

# Data shapeing & lighting conditions variance with batching 
processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)


# Loading the foundational SegFromer model, setting label & ID eqaulivence.  
pretrained_model_name = "nvidia/mit-b3" 
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)

# Setting traning arguments/hyperparameters
training_args = TrainingArguments(
    output_dir="./segformer_coralscapes_b3",
    learning_rate=6e-5,
    num_train_epochs=20,              # adjust based on convergence
    per_device_train_batch_size=3,    # tune to your VRAM (b3 ~8-12GB for 512×512)
    per_device_eval_batch_size=3,
    save_total_limit=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    remove_unused_columns=False,      # keep 'pixel_values', 'labels'
    push_to_hub=False,                # set True later if you want
    report_to="none",                 # or "tensorboard"/"wandb"
    fp16=True,                        # mixed precision on GPU
    dataloader_num_workers=4,
)

# 
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=len(id2label),
        ignore_index=0,
        reduce_labels=processor.do_reduce_labels,
    )
    
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
    return metrics



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

