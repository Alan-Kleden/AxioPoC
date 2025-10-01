# scripts/normalize_thread_ids.py
import argparse
import pandas as pd

def t3_to_int(x: str) -> int:
    s = str(x)
    if s.startswith("t3_"): s = s[3:]
    return int(s, 36)

def int_to_t3(x: int) -> str:
    # encode entier -> base36 + préfixe t3_
    n = int(x)
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0: b36 = "0"
    else:
        b = []
        while n:
            n, r = divmod(n, 36)
            b.append(chars[r])
        b36 = "".join(reversed(b))
    return "t3_" + b36

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="CSV input (doit contenir thread_id)")
    ap.add_argument("--out", required=True, help="CSV output")
    ap.add_argument("--mode", required=True, choices=["cmv_t3_to_int", "int_to_t3"], help="Conversion à effectuer")
    ap.add_argument("--col", default="thread_id", help="Nom de la colonne id (default: thread_id)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if args.col not in df.columns:
        raise SystemExit(f"[ERR] Colonne {args.col!r} absente de {args.inp}")

    if args.mode == "cmv_t3_to_int":
        df[args.col] = df[args.col].astype(str).map(t3_to_int)
        # IMPORTANT: s'assurer que c'est bien int
        df[args.col] = df[args.col].astype("int64")
    else:
        # int -> t3_<base36> (si jamais tu veux l'autre sens)
        df[args.col] = df[args.col].astype("int64").map(int_to_t3)

    df.to_csv(args.out, index=False)
    print(f"[OK] {args.mode} -> {args.out} | rows={len(df)}")

if __name__ == "__main__":
    main()
