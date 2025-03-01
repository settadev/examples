from setta.tasks.fns import SettaInMemoryFn
from groq import Groq

$SETTA_GENERATED_PYTHON

client = Groq(api_key=api_key)
c = {"history": None}


def processChat(p):
    section = p["llama3"]
    userMsg = section["latestChatMessage"]
    if c["history"] is None:
        c["history"] = section["chatHistory"]

    if not userMsg:
        return None

    c["history"].append(
        {
            "role": "user",
            "content": userMsg,
        }
    )

    chat_completion = client.chat.completions.create(
        messages=c["history"],
        model="llama-3.3-70b-versatile",
    )

    c["history"].append(
        {"role": "assistant", "content": chat_completion.choices[0].message.content}
    )

    return [
        {
            "name": "test chat",
            "type": "chatHistory",
            "value": c["history"],
        }
    ]


fn = SettaInMemoryFn(fn=processChat)

print("done!")
