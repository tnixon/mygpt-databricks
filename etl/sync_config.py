"""Copy the per-dataset config CSVs into the UC config volume.

Runs as the first task of the DABs ETL job (before 03_sections_and_config.sql,
which reads these CSVs via read_files). The bundle syncs etl/config/*.csv into the
job's working tree; this task lands them on the governed volume so the config is a
single editable source of truth.
"""

from __future__ import annotations

import glob
import os
import shutil

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "config")
VOLUME_DIR = os.environ.get("MYGPT_CONFIG_VOLUME", "/Volumes/structural_biology/mygpt/config")


def main() -> None:
    os.makedirs(VOLUME_DIR, exist_ok=True)
    for src in glob.glob(os.path.join(LOCAL_DIR, "*.csv")):
        dest = os.path.join(VOLUME_DIR, os.path.basename(src))
        shutil.copyfile(src, dest)
        print(f"copied {os.path.basename(src)} -> {dest}")


if __name__ == "__main__":
    main()
