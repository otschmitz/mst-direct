"""Download the MIT-licensed DMS benchmark inputs with checksum validation."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


FILES = {
    "reference_data.mat": (
        "https://raw.githubusercontent.com/leandrofgr/DMS/"
        "898db6dcf2a2eef99ef82e1c664d17e1f5012510/datasets/reference_data.mat",
        "cf829cb8c51049cd2fc118eedcaa5b799b2b5ff1ee88385558f246e4dca9b524",
    ),
    "HardData_ReferenceModel_size100_range20.mat": (
        "https://raw.githubusercontent.com/leandrofgr/DMS/"
        "898db6dcf2a2eef99ef82e1c664d17e1f5012510/datasets/"
        "HardData_ReferenceModel_size100_range20.mat",
        "0d5ce154da118d06c9a5b1f4f4c9003b8cc68854ea8c7394508863a683e24f37",
    ),
}


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "data"
    target.mkdir(exist_ok=True)
    for name, (url, expected) in FILES.items():
        payload = urlopen(url, timeout=60).read()
        actual = sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: {actual}")
        (target / name).write_bytes(payload)
        print(f"Downloaded {name} ({len(payload)} bytes, sha256={actual})")


if __name__ == "__main__":
    main()
