from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from dataset import CodeDataset
from sklearn.metrics import accuracy_score, f1_score
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2
)

#Load original CodeBERT model and fine-tuned LoRA wieghts on top of base model
model = PeftModel.from_pretrained(base_model, "./saved_models/aegis_final")

test_dataset = CodeDataset("data/processed/test.jsonl", tokenizer)

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

accuracy = accuracy_score(true_labels, predictions) 
f1 = f1_score(true_labels, predictions) #harmonic mean of precision and recall 

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
