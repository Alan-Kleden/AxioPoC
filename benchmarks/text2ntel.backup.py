import pandas as pd
from convokit import Corpus

cmv = Corpus(filename="data/convokit/cmv/winning-args-corpus/winning-args-corpus")

rows = []
for conv in cmv.iter_conversations():
    outcome = conv.meta.get("delta_awarded")  # True/False si persuasion
    for utt in conv.iter_utterances():
        rows.append({
            "thread_id": conv.id,
            "text": utt.text,
            "timestamp": utt.meta.get("timestamp") or utt.meta.get("created_utc"),
            "outcome": outcome
        })
pd.DataFrame(rows).to_csv("data/convokit/cmv/cmv.csv", index=False)
print("OK -> data/convokit/cmv/cmv.csv")
