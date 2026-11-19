"""One pinned, private Tracker release shared by live and offline execution."""
import json
from pathlib import Path

RECEIPT = Path(__file__).resolve().parents[4] / 'configs/supermimic/trackers.json'


def tracker_release():
    return {row['robot']: row for row in json.loads(RECEIPT.read_text())['checkpoints']}


def tracker_checkpoint(robot, shared):
    entry = tracker_release()[robot]
    return Path(shared) / entry['relative_path']


def tracker_release_id():
    return json.loads(RECEIPT.read_text())["release_id"]
