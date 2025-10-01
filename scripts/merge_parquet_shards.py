# scripts/merge_parquet_shards.py
import argparse, glob, os, sys
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help='Pattern glob des shards, ex: "G:\\Mon Drive\\AxioPoC\\artifacts_xfer\\afd_msg_emb_shard_*.parquet"')
    ap.add_argument("--out", required=True, help='Parquet de sortie, ex: "H:\\AxioPoC\\artifacts_xfer\\afd_msg_emb.parquet"')
    args = ap.parse_args()

    files = sorted(glob.glob(args.shards))
    if not files:
        print(f"[ERR] Aucun shard trouvé pour: {args.shards}", file=sys.stderr)
        sys.exit(1)

    print(f"[merge] {len(files)} shard(s) -> {args.out}")

    # Lis le schéma du 1er shard
    first_pf = pq.ParquetFile(files[0])
    schema = first_pf.schema_arrow

    # Writer Parquet unique (même schéma)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = pq.ParquetWriter(args.out, schema, compression="zstd", version="2.6")  # zstd compact/rapide (pyarrow>=12)

    total_rows = 0
    for i, fp in enumerate(files, 1):
        pf = pq.ParquetFile(fp)
        nrg = pf.num_row_groups
        shard_rows = 0
        for rg in range(nrg):
            tbl = pf.read_row_group(rg)
            writer.write_table(tbl)
            shard_rows += tbl.num_rows
        total_rows += shard_rows
        print(f"[merge] {i}/{len(files)} | +{shard_rows} rows (cum={total_rows})")

    writer.close()
    print(f"[OK] Merge -> {args.out} | rows={total_rows}")

    # Sanity: re-ouvre vite fait pour afficher shape
    pf_out = pq.ParquetFile(args.out)
    rows = sum(pf_out.metadata.row_group(i).num_rows for i in range(pf_out.metadata.num_row_groups))
    cols = pf_out.schema_arrow.num_fields
    print(f"[check] out shape ≈ ({rows}, {cols})")

if __name__ == "__main__":
    main()
