from math import e
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from dataset import CodeDataset
import torch

#Pre-trained CodeBERT model loaded
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2  #Tells model to add two classification heads, one for human (class 0) and one for LLM (class 1)
)

#Device selection to determine whether to use GPU acceleration or CP
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#LoRA configuration - (technique to efficiently fine-tune by updating small subset of parameters instead of millions)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=4, #Rank of low-rank matrices (lower the rank, fewer parameters to rain)
    lora_alpha=16, #Scaling facotr for LoRA updates
    lora_dropout=0.1, #10% dropout to prevent overfitting
    target_modules=["query", "value", "key"] #which parts of attention layers to apply LoRA
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
    per_device_train_batch_size=4, #4 samples at once during training
    gradient_accumulation_steps=8, #accumulate gradients over 8 minibatches before updating (4*8 is effective batch size)
    learning_rate=1e-4, #how big steos to take whne updating model paramters
    weight_decay=0.01, #L2 regularization to prevent overfitting
    logging_steps=100,
    eval_strategy="epoch", #evalute model after eachh epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    warmup_ratio=0.1 #increase learning rate for first 10% of training
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


