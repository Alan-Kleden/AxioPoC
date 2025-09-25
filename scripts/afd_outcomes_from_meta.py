from convokit import Corpus, download
import csv
print("Chargement corpus (cache local)…")
c = Corpus(filename=download("wiki-articles-for-deletion-corpus"))

def get_outcome(conv):
    m = getattr(conv, "meta", {}) or {}
    # clé fiable trouvée par la probe
    return str(m.get("outcome_label", "") or "")

rows=[]
for cid in c.get_conversation_ids():
    conv = c.get_conversation(cid)
    rows.append({"thread_id": cid, "outcome": get_outcome(conv)})

with open(r"data/wikipedia/afd/afd_threads.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["thread_id","outcome"]); w.writeheader(); w.writerows(rows)
print("OK -> data/wikipedia/afd/afd_threads.csv | n=", len(rows))
