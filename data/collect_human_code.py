import json
import os
import random

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
HUGGING_FACE_TOKEN=os.getenv("HUGGING_FACE_TOKEN", "")
random.seed(22)

#authenticate code
login(token=HUGGING_FACE_TOKEN)

IN_PATH = "data/raw/train.jsonl"
PATH = "data/raw"
N_TARGET = 5000
os.makedirs(PATH, exist_ok=True)



def collect_human_code():
    ds = load_dataset("json", data_files=IN_PATH, split="train")
    try:
        count = 0
        with open(f"{PATH}/human_code.jsonl", "w", encoding="utf-8") as file:
            for sample in ds:
                if count >= N_TARGET:
                    break

                question = sample.get("question", "")
                raw_solution = sample.get("solutions", "")

                if not raw_solution or raw_solution.strip() == "":
                    continue

                try:
                    solutions = json.loads(raw_solution)
                    if not solutions or len(solutions) == 0:
                        continue

                    solution = random.choice(solutions)
                except:
                    print("Empty solution")
                    continue

                if not solution:
                    continue

                starter_code = sample.get("starter_code", "")

                sample_metadata = {"question": question, "solution": solution, "starter_code": starter_code, "label": 0}
                file.write(json.dumps(sample_metadata) + "\n")

                count += 1

    except Exception as e:
        raise Exception(f"Failed to collect human samples: {e}")

#splits 2000 human code responses and splits the other half as questions for LLM
def split_code(in_path, out_path, question_path):
    human_list = []
    llm_question_list = []
    raw_list = []
    with open(in_path, "r") as file:
        for sample in file:
            raw_list.append(json.loads(sample))

    random.shuffle(raw_list)
    for entry in raw_list:
        if len(human_list) < 2500:
            to_store = {"code": entry.get("solution"), "label": 0}
            human_list.append(to_store)
        else:
            to_store = {"question": entry.get("question"), "starter_code": entry.get("starter_code")}
            llm_question_list.append(to_store)


    with open(out_path, "w") as file:
        for sample in human_list:
            file.write(json.dumps(sample) + "\n")

    with open(question_path, "w") as file:
        for sample in llm_question_list:
            file.write(json.dumps(sample) + "\n")



def main():
    collect_human_code()
    print("Finished collecting human samples")

if __name__ == "__main__":
    main()
    split_code(f"{PATH}/human_code.jsonl", f"{PATH}/human_clean.jsonl", f"{PATH}/llm_questions.jsonl")








