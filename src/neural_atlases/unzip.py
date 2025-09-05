from zipfile import ZipFile
from pathlib import Path
import shutil

zip_path   = Path("data/data.zip")        # the archive on disk
out_dir    = Path("data/blackswan")       # where you want the files to land
wanted_dir = "nla_share/blackswan/blackswan/"               # folder *inside* the ZIP (MUST end with “/”)

with ZipFile(zip_path) as zf:
    for member in zf.namelist():              # iterate over every entry
        if member.startswith(wanted_dir):     # keep only the desired folder tree
            dest = out_dir / member           # where this entry should go on disk

            # make sure parent directories exist
            dest.parent.mkdir(parents=True, exist_ok=True)

            # if the member is a directory, just create it and continue
            if member.endswith("/"):
                dest.mkdir(exist_ok=True)
                continue

            # otherwise copy the file’s bytes
            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
