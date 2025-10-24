import asyncio, os, json, logging
from dotenv import load_dotenv
from matplotlib.image import resample
from openai import AsyncOpenAI
from prompts import MODEL, get_questions, build_message

load_dotenv()
client = AsyncOpenAI()
MAX_OUTPUT_TOKENS = 5000
TIME_SLEEP = 5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def generate_code(prompt, semaphore, max_retries=2):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.responses.create(
                    model=MODEL,
                    input=prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )

                result = response.output_text

                if result and result.strip(): 
                    return result
                else:
                    print(f"Empty response on attempt {attempt + 1}, retrying...")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
            
                if attempt < max_retries - 1:
                    await asyncio.sleep(TIME_SLEEP)
        
        print(f"Failed to generate code after {max_retries} attempts")
        return None

async def generate_all(max_concurrent=10):
    try:
        count = 0
        with open("data/raw/llm_code.jsonl", "a", encoding="utf-8") as file:
            semaphore = asyncio.Semaphore(max_concurrent)
            questions = get_questions("data/raw/llm_questions.jsonl")
            logger.info(f"Starting generation for {len(questions)} questions with max_concurrent={max_concurrent}")

            tasks = []
            for q in questions:
                tasks.append(generate_code(build_message(q.get("question"), q.get("starter_code")), semaphore))

            buffer = []
            for fut in asyncio.as_completed(tasks):
                res = await fut
                if res is not None and not isinstance(res, Exception):
                    buffer.append({"code": res, "label": 1})
                    if len(buffer) >= 10:
                        for x in buffer:
                            file.write(json.dumps(x) + "\n")
                        file.flush()
                        count += len(buffer)
                        buffer.clear()

                        if count % 10 == 0:
                            logger.info(f"Saved {count} snippets to disk")

            if buffer:
                for x in buffer:
                    file.write(json.dumps(x) + "\n")
                file.flush()
                count += len(buffer)
        
        print("Finished generating data")
    except Exception as e:
        print(f"API Error: {e}")

def main():
    asyncio.run(generate_all())


if __name__ == "__main__":
    main()

