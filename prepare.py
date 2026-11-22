"""One-off preparation (needs internet): download the official yolo11l.pt next to run.py."""
import os

from ultralytics import YOLO

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
YOLO("yolo11l.pt")  # downloads the official weights into the current directory
assert os.path.exists(os.path.join(here, "yolo11l.pt"))
print("yolo11l.pt ready")
