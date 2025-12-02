import os
import fnmatch
import subprocess
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

UIROBOT_CANDIDATES = [
    r"%ProgramFiles%\UiPath\Studio\UiRobot.exe",
    r"%ProgramFiles%\UiPath\UiRobot.exe",
    r"%USERPROFILE%\AppData\Local\UiPath\UiRobot.exe",
    r"%USERPROFILE%\AppData\Local\UiPath\UiPath.Agent.exe",
]

def find_uirobot():
    for candidate in UIROBOT_CANDIDATES:
        p = os.path.expandvars(candidate)
        if os.path.exists(p):
            logging.info(f"Found UiRobot at: {p}")
            return p
    # fallback: search PATH
    for path in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(path) / "UiRobot.exe"
        if p.exists():
            logging.info(f"Found UiRobot in PATH: {p}")
            return str(p)
    raise FileNotFoundError("UiRobot.exe not found. Install UiPath Robot/Assistant or set a correct path.")

def find_extracted_package_folder(process_name: str) -> Path:
    """
    Finds the extracted package folder under %LOCALAPPDATA%\UiPath\Packages.
    Returns the most-recent version folder Path.
    """
    base = Path(os.path.expandvars(r"%LOCALAPPDATA%\UiPath\Packages"))
    if not base.exists():
        raise FileNotFoundError(f"UiPath Packages folder not found: {base}")
    matches = [p for p in base.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, f"{process_name}*")]
    if not matches:
        raise FileNotFoundError(f"No package folder for '{process_name}' in {base}")
    # pick newest top-level match, then newest subfolder if present
    best = max(matches, key=lambda p: p.stat().st_mtime)
    subdirs = [d for d in best.iterdir() if d.is_dir()]
    if subdirs:
        version_folder = max(subdirs, key=lambda p: p.stat().st_mtime)
        return version_folder
    return best

def deploy_json_and_run_local(source_json_path: Path, process_name: str, uirobot_path: str = None,
                              target_filename: str = "workflow.json", wait=True):
    """
    Copies source_json_path -> <package_folder>/<target_filename> and runs UiRobot -p process_name.
    """
    uirobot_path = uirobot_path or find_uirobot()
    package_folder = find_extracted_package_folder(process_name)
    target = package_folder / target_filename
    # copy
    with open(source_json_path, "rb") as fr, open(target, "wb") as fw:
        fw.write(fr.read())
    logging.info(f"Copied {source_json_path} -> {target}")

    # run UiRobot
    cmd = [uirobot_path, "-p", process_name]
    logging.info("Running UiRobot cmd: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if wait:
        out, err = proc.communicate()
        logging.info("UiRobot returncode=%s", proc.returncode)
        if out:
            logging.info("stdout: %s", out.strip())
        if err:
            logging.warning("stderr: %s", err.strip())
        return proc.returncode, out, err
    return proc
