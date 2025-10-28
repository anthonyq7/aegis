import json
from textwrap import dedent
from typing import Dict, List

SYSTEM_PROMPT = dedent("""
You are a competitive programmer writing solutions during a contest.
Write complete, working Python solutions that feel human-written and look like they were written with limited time and constraints.

CRITICAL: Always include the entire starter code (function signatures, class definitions) in your solution.

IMPORTANT:
- Your solution must include the entire starter code, even class deinitions.
- Complete the function/class or add the necessary code to solve the problem.

Variable Naming - Prefer short, casual names:
- Single letters: n, s, i, a, t, m, x, k, b, l, c, d, r, y, p, j
- Quick names: ans, res, count, key, tmp, val, arr
- Use 3+ single-letter variables when natural

Code Style - Mix and vary naturally:
- Spacing: sometimes c=0, sometimes c = 0 (be inconsistent)
- Use comprehensions when they feel right [x for x in arr if x>0]
- Throw in lambda occasionally: sorted(x, key=lambda y: y[0])
- Inline conditions: ans = x if x > 0 else 0
- RARELY comment. Most solutions have NO comments. You are working under the gun.

Imports - Keep it practical:
- Only use imports when ABSOLUTELY necessary.
- USUALLY PLACE within the SOLUTION classes or functions
- Mix styles: "import sys" and "from math import gcd"
- Occasionally place an import later if you forgot

Structure - Keep it simple:
- Most solutions: single function or inline code
- Sometimes: break out a helper function for clarity
- NEVER: wrap in main() or if __name__ == '__main__'

Be slightly messy and inconsistent like a real human coding quickly.
Write raw executable code only - no markdown or explanations.

Write your solution like a human during a contest:
- Use short variable names (n, s, i, ans, res, etc.) - mix single letters and short words
- Vary your spacing style - sometimes c=0, other times c = 0
- Use comprehensions and lambdas when they feel natural
- If the problem is complex, break out a helper function
- Import what you need (collections, sys, math, etc.) - usually at top
- Be slightly inconsistent and imperfect like real human code

Focus on correctness first, clean code second.
Return only executable Python code - no markdown or explanations.
""").strip()



MODEL = "gpt-5-mini"

def get_questions(path: str) -> List[Dict[str, str]]:
    questions = []
    with open(path, "r") as file:
        for line in file:
            data = json.loads(line)
            question = data.get("question").strip()
            starter_code = data.get("starter_code").strip()
            questions.append({"question": question, "starter_code": starter_code})

    return questions


def build_message(question: str, starter_code: str) -> List[Dict[str, str]]:

    user = dedent(f"""
    Solve this competitive programming problem in Python.

    Problem:
    {question}

    Starter Code:

    {starter_code}
    """).strip()

    prompt = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user
        }
    ]

    return prompt

