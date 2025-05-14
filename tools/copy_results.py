import os
from pathlib import Path
from socket import gethostname
from subprocess import run

"""
Copy all results from node "/scr-ssd" to "/viscam/projects using rsync"
This should be run on the compute node where /scr or /scr-ssd is mounted
"""

node_prefixes = [
    "viscam1", "viscam2", "viscam3", "viscam4", "viscam5", "viscam6", "viscam7", "viscam8", "viscam9", "viscam10",
    "viscam11", "viscam12", "viscam13",
]
bwlimit = "0"


def copy_results(src_dir, dst_dir, verbose=False):
    """Copy data from scr_dir to dst_dir in bulks, both should be at same level of dataset directory"""
    assert gethostname().split(".")[0] in node_prefixes, f"This script should be run on a compute node: {node_prefixes}"
    src_dir = str(src_dir).rstrip('/') + '/'
    dst_dir = str(dst_dir).rstrip('/')
    os.makedirs(dst_dir, exist_ok=True)
    command = ["rsync", "-avz", str(src_dir), str(dst_dir), "--bwlimit", bwlimit, "--remove-source-files"]
    if verbose:
        command.extend(["--progress"])
    run(command, check=True)


if __name__ == "__main__":
    src_dir = Path("/scr-ssd/briannlz/video_object_processing/data/tmp/")
    dst_dir = Path("data/tmp/")
    copy_results(src_dir, dst_dir, verbose=True)
    print("Done", flush=True)
