
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

key = os.getenv("GROQ_API_KEY")
if not key:
    raise SystemExit("GROQ_API_KEY is missing.")

try:
    client = Groq(api_key=key, timeout=30.0, max_retries=0)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "Say hello in one short sentence."
            }
        ],
        reasoning_effort="low",
        max_completion_tokens=500,
        include_reasoning=False,
    )

    message = response.choices[0].message
    print("CONTENT:", repr(message.content))
    print("FINISH REASON:", response.choices[0].finish_reason)
    print("USAGE:", response.usage)

except Exception as error:
    print("ERROR:", type(error).__name__)
    print(str(error)[:800])
