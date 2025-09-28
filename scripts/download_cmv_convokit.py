# -*- coding: utf-8 -*-
"""
Télécharge le corpus Change My View (CMV) de ConvoKit et affiche le dossier où
sont écrits `utterances.jsonl` et `conversations.json`.
"""
from convokit import Corpus, download
from pathlib import Path

name_candidates = [
    "change-my-view-corpus",
    "change-my-view",
    "reddit-cmv-corpus",
]

cachedir = Path.home() / ".convokit" / "saved-corpora"

for name in name_candidates:
    try:
        print(f"Trying: {name} …")
        path = download(name)  # télécharge si absent, sinon renvoie le chemin
        c = Corpus(path)       # force la création des fichiers jsonl si besoin
        print("OK:", path)
        print("Saved in:", cachedir)
        break
    except Exception as e:
        print(f"  -> failed for {name}: {e}")
else:
    raise SystemExit("Aucun nom de corpus CMV connu n'a fonctionné.")
