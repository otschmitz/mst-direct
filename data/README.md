# DMS benchmark data

Run `python scripts/download_dms_data.py`. The script downloads two files from
the exact public DMS commit used in the audit and validates SHA-256 checksums.

Source: https://github.com/leandrofgr/DMS at commit
`898db6dcf2a2eef99ef82e1c664d17e1f5012510`.

The upstream repository is MIT licensed. Retain its copyright and licence when
redistributing files. These are synthetic benchmark data, not field data.

For the proposed real-data experiment, run
`python scripts/download_jura_data.py`. This retrieves `jura.rda` from a fixed
`gstat` commit and validates its SHA-256 checksum. Redistribution rights must be
confirmed before committing the downloaded file to a public repository.
