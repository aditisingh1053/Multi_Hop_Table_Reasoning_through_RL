"""Extract WikiTables-WithLinks tables and passages straight from the git object store.

Several file names in the repository contain characters that Windows does not allow
(':', '*', '"', ...), so `git checkout` fails on Windows. We avoid the working tree
entirely and stream every blob with `git cat-file --batch`.

Output: data/processed/wikitables.jsonl, one line per table:
    {"table_id": ..., "table": {...tables_tok json...}, "passages": {...request_tok json...}}
"""
import argparse
import json
import subprocess
from pathlib import Path


def list_blobs(repo: Path, prefix: str):
    out = subprocess.run(
        ["git", "-c", "core.quotepath=off", "ls-tree", "-r", "-z", "HEAD", prefix],
        cwd=repo, check=True, capture_output=True,
    ).stdout.decode("utf-8")
    blobs = {}
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        _, kind, sha = meta.split()
        if kind == "blob" and path.endswith(".json"):
            table_id = path[len(prefix) + 1:-len(".json")]
            blobs[table_id] = sha
    return blobs


def read_blobs(repo: Path, shas):
    """Yield (sha, bytes) for each sha using one long-lived `git cat-file --batch`."""
    proc = subprocess.Popen(["git", "cat-file", "--batch"], cwd=repo,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for sha in shas:
            proc.stdin.write((sha + "\n").encode())
            proc.stdin.flush()
            header = proc.stdout.readline().decode().split()
            assert header[0] == sha and header[1] == "blob", header
            size = int(header[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)  # trailing newline
            yield sha, data
    finally:
        proc.stdin.close()
        proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="data/raw/WikiTables-WithLinks")
    ap.add_argument("--out", default="data/processed/wikitables.jsonl")
    args = ap.parse_args()
    repo = Path(args.repo)
    tables = list_blobs(repo, "tables_tok")
    passages = list_blobs(repo, "request_tok")
    print(f"tables: {len(tables)}  passage files: {len(passages)}")

    by_sha = {}
    wanted = list(tables.values()) + list(passages.values())
    for sha, data in read_blobs(repo, wanted):
        by_sha[sha] = data

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for table_id, sha in sorted(tables.items()):
            table = json.loads(by_sha[sha].decode("utf-8"))
            psg = json.loads(by_sha[passages[table_id]].decode("utf-8")) if table_id in passages else {}
            f.write(json.dumps({"table_id": table_id, "table": table, "passages": psg}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} tables to {args.out}")


if __name__ == "__main__":
    main()
