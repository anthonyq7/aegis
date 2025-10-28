import torch, os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

OUT_PATH = "model/results"
READ_FILE_PATH = "data/processed/test.jsonl"
os.makedirs(OUT_PATH, exist_ok=True)

def get_attention_weights(model, tokenizer, code_snippet):

    inputs = tokenizer(
        code_snippet,
        truncation = True,
        padding = "max_length", 
        max_length = 512, 
        return_tensors = "pt"
    )

    #GET model outputs with attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attention = outputs.attentions

    #outputs.attentions is a tuple of attention weights from each layer
    #shape: (batch_size, num_heads, sequence_length, sequence_length)
    return attention, inputs


def visualize_attention(attention, tokens, layer=-1, head=0):
    #attention: tuple of attention tensors from model
    #tokens: list of tokens
    #layer: which transformer layer to visualize (-1 for last layer)
    #head: which attention head to visualize

    #getting the attention weights for specified layer
    attn_weights = attention[layer][0, head].cpu().numpy()
    
    #limiting to first 60 tokens for readability
    max_tokens = 60
    attn_weights = attn_weights[:max_tokens, :max_tokens]
    tokens = tokens[:max_tokens]
    
    #create heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        attn_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        cbar_kws={"label": "Attention Weight"}
    )
    plt.title(f"Attention Weights - Layer {layer}, Head {head} (First {max_tokens} tokens)")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=90, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/attention_weights.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )

    model = PeftModel.from_pretrained(base_model, "./saved_models/aegis_final")

    with open(READ_FILE_PATH, "r") as file:
        code_snippet = json.loads(file.readline()).get("code")

    attention, inputs = get_attention_weights(model, tokenizer, code_snippet)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    visualize_attention(attention, tokens, layer=-1, head=0)

if __name__ == "__main__":
    main()