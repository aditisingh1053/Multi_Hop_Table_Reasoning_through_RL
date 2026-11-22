"""Write gpu_bundle.zip (code + configs only, a few hundred KB) for upload to Kaggle / Colab.

The archive contains a top-level folder `mhtrl/`, as GPU_INSTRUCTIONS.md expects.
Usage: python scripts/make_gpu_bundle.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "scripts", "configs", "tests"]
FILES = ["requirements.txt", "README.md", "GPU_INSTRUCTIONS.md"]
EXTRA = ["results/test_qids_seen_table.json"]   # used by the report for the unseen-table split


def main():
    out = ROOT / "gpu_bundle.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE:
            for f in sorted((ROOT / d).rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                    z.write(f, Path("mhtrl") / f.relative_to(ROOT))
                    n += 1
        for f in FILES + EXTRA:
            if (ROOT / f).exists():
                z.write(ROOT / f, Path("mhtrl") / f)
                n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
