
import os

import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = "data/raw"
OUT_PATH = "data/processed"

os.makedirs(IN_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

#Returns train split, test split, and validation split
def preprocess() -> None:
    #lines is import since it's json lines file
    human_code_df = pd.read_json(f"{IN_PATH}/human_clean.jsonl", lines=True)
    llm_code_df = pd.read_json(f"{IN_PATH}/llm_code.jsonl", lines=True)

    human_code_df = human_code_df.iloc[:2000]
    llm_code_df = llm_code_df.iloc[:2000]

    #combined df
    combined_df = pd.concat([human_code_df, llm_code_df])
    print(f"Combined df size: {len(combined_df)}")

    #keep track of how many duplicates
    total_duplicates = combined_df.duplicated(subset=["code"]).sum()

    #drop duplicates
    combined_df = combined_df.drop_duplicates(subset=["code"], keep="first")
    print(f"Total number of duplicates: {total_duplicates}")

    #shuffle combined df
    combined_df = combined_df.sample(frac=1, random_state=22).reset_index(drop=True)

    #split into train, test, and validation
    #stratify ensures that there's an even split, i.e. stratified lol
    train, temp = train_test_split(combined_df, test_size=0.2, random_state=22, stratify=combined_df["label"])
    test, validate = train_test_split(temp, test_size=0.5, random_state=22, stratify=temp["label"])

    train.to_json(f"{OUT_PATH}/train.jsonl", orient="records", lines=True)
    test.to_json(f"{OUT_PATH}/test.jsonl", orient="records", lines=True)
    validate.to_json(f"{OUT_PATH}/validate.jsonl", orient="records", lines=True)


    print(f"Train: {len(train)} samples - {(len(train)/len(combined_df))*100:.1f}%")
    print(f"Test: {len(test)} samples - {(len(test)/len(combined_df))*100:.1f}%")
    print(f"Validate: {len(validate)} samples - {(len(validate)/len(combined_df))*100:.1f}%")


def check_balance() -> None:
    for split in ['train', 'validate', 'test']:
        df = pd.read_json(f"{OUT_PATH}/{split}.jsonl", lines=True)

        human = sum(df['label'] == 0)
        llm = sum(df['label'] == 1)
        total = len(df)

        print(f"\n{split.upper()}:")
        print(f"  Total: {total}")
        print(f"  Human: {human} ({human/total*100:.1f}%)")
        print(f"  LLM: {llm} ({llm/total*100:.1f}%)")

if __name__ == "__main__":
    preprocess()
    check_balance()

