from convokit import Corpus, download
from collections import Counter
c = Corpus(filename=download("wiki-articles-for-deletion-corpus"))
keys = Counter(); samples = {}
for cid in c.get_conversation_ids():
    m = getattr(c.get_conversation(cid), "meta", {}) or {}
    for k,v in m.items():
        keys[k]+=1
        if k not in samples and v: samples[k]=str(v)[:120]
print("Top keys:", keys.most_common(20))
print("Samples:")
for k,v in samples.items(): print("-",k,":",v)
