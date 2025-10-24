from math import e
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from dataset import CodeDataset
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2  #Tells model to add two classification heads, one for human (class 0) and one for LLM (class 1)
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value", "key"]
)

#Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#load datasets
train_dataset = CodeDataset("data/processed/train.jsonl", tokenizer)
val_dataset = CodeDataset("data/processed/validate.jsonl", tokenizer)

#Training Arguments
training_args = TrainingArguments(
    output_dir="./saved_models/aegis",
    num_train_epochs=3,
    per_device_eval_batch_size=4, 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8, 
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    warmup_ratio=0.1
)

#Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

#Training
print("\nStarting training")
trainer.train()

#Save
print("\nSaving model")
model.save_pretrained("./saved_models/aegis_final")
tokenizer.save_pretrained("./saved_models/aegis_final")

print("Done")


