"""Download the real Swiss Jura geochemistry dataset from a fixed gstat commit."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


URL = (
    "https://raw.githubusercontent.com/r-spatial/gstat/"
    "c4612f3abcdb58b79f5639c2562d8314ae657ce8/data/jura.rda"
)
EXPECTED = "f1d9e8a7e6686aa7473a1ce452fffe74f0c45a3e2ba464e35a9163c0d447d845"


def main() -> None:
    payload = urlopen(URL, timeout=60).read()
    actual = sha256(payload).hexdigest()
    if actual != EXPECTED:
        raise RuntimeError(f"Checksum mismatch: {actual}")
    target = Path(__file__).resolve().parents[1] / "data" / "jura.rda"
    target.write_bytes(payload)
    print(f"Downloaded {target.name} ({len(payload)} bytes, sha256={actual})")
    print("Read in R with data(jura), or in Python with pyreadr plus pandas/polars.")


if __name__ == "__main__":
    main()
