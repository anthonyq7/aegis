import json
import os

import matplotlib.pyplot as plt
import torch
from dataset import CodeDataset
from peft import PeftModel
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2
)

IN_PATH = "data/processed"
OUT_PATH = "model/results"
os.makedirs(OUT_PATH, exist_ok=True)

#Load original CodeBERT model and fine-tuned LoRA wieghts on top of base model
model = PeftModel.from_pretrained(base_model, "./saved_models/aegis_final")

test_dataset = CodeDataset(f"{IN_PATH}/test.jsonl", tokenizer)

model.eval() #puts model in eval mode (disables dropout, batch norm updates)
predictions = []
true_labels = []

with torch.no_grad(): #toch.nograd() disables gradient computation (saves memory, speeds up inference)
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0) #.unsqueeze adds a batch dimension(model expects batches, not single sample)
        attention_mask = sample["attention_mask"].unsqueeze(0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=-1).item() #Gets class with highest probability, .item() convets tensor to Python number
        predictions.append(pred)
        true_labels.append(sample["labels"].item())

cm = confusion_matrix(true_labels, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Human", "AI-Generated"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Aegis: Code Origin Classification", fontsize=14, pad=15)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT_PATH}/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions) #harmonic mean of precision and recall

results = {
    "model_name": "Aegis",
    "metrics": {
        "accuracy": f"{float(accuracy):.4f}",
        "precision": f"{float(precision):.4f}",
        "recall": f"{float(recall):.4f}",
        "f1": f"{float(f1):.4f}",
    },
    "confusion_matrix": {
        "values": cm.tolist(),
        "labels": ["Human", "AI-Generated"],
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    }
}

with open(f"{OUT_PATH}/model_results.json", "w") as file:
    file.write(json.dumps(results, indent=4))

print(f"Results saved to {OUT_PATH}/model_results.json")
