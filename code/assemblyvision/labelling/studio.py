"""Label Studio project configs and the reading of its JSON exports.

Two passes, so every keypoint belongs to exactly one person:
  1. boxes: one task per sampled frame, a box around every person;
  2. keypoints: one task per box (a crop with margin), the 17 COCO keypoints of
     that person. Each joint has a visible and an occluded label, so marking a
     joint is one click; a joint that cannot be placed is left out.
Frames and crops are never uploaded into Label Studio: tasks point at the frame
server, so the only thing Label Studio stores (and exports) is label JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..datasets.mapping import COCO17

BOX_LABEL = "person"
OCCLUDED = " (occluded)"
_COLORS = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
           "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000")


def boxes_config() -> str:
    return f"""<View>
  <Header value="Draw a box around every person, including partly visible ones (guide, section 3)."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="box" toName="image" strokeWidth="2">
    <Label value="{BOX_LABEL}" background="#ff4d4d"/>
  </RectangleLabels>
  <Choices name="skip" toName="image" showInline="true">
    <Choice value="frame unusable (blur, cut, black)"/>
  </Choices>
</View>
"""


def keypoints_config() -> str:
    labels = []
    for name, color in zip(COCO17, _COLORS):
        labels.append(f'    <Label value="{name}" background="{color}"/>')
        labels.append(f'    <Label value="{name}{OCCLUDED}" background="{color}" selectedColor="#000000"/>')
    return ("<View>\n"
            '  <Header value="Mark the 17 keypoints of the person in the middle of the crop (guide, section 4).'
            ' Use the occluded label when the joint is hidden but its position is clear; leave it out otherwise."/>\n'
            '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>\n'
            '  <KeyPointLabels name="kp" toName="image" strokeWidth="2" smart="false">\n'
            + "\n".join(labels) +
            "\n  </KeyPointLabels>\n"
            '  <Choices name="skip" toName="image" showInline="true">\n'
            '    <Choice value="not the person in the middle, or too small to label"/>\n'
            "  </Choices>\n</View>\n")


def write_configs(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, text in (("boxes.xml", boxes_config()), ("keypoints.xml", keypoints_config())):
        (folder / name).write_text(text)
        out.append(folder / name)
    return out


def _author(ann: dict, names: dict | None = None) -> str:
    """The labeller: Label Studio 1.23 exports completed_by as the user id (an int);
    names maps ids to names (eval_pose_labels --labellers)."""
    c = ann.get("completed_by")
    key = str(c.get("email") or c.get("id")) if isinstance(c, dict) else str(c)
    if isinstance(c, dict) and names and str(c.get("id")) in names:
        return names[str(c.get("id"))]
    return (names or {}).get(key, key)


def load_export(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("a Label Studio JSON export is a list of tasks")
    return data


def _skipped(results: list[dict]) -> bool:
    return any(r.get("type") == "choices" and r.get("value", {}).get("choices") for r in results)


def parse_boxes(tasks: list[dict], names: dict | None = None) -> list[dict]:
    """Box pass export -> one record per (task, annotation): frame, labeller and boxes in
    frame pixels [x1, y1, x2, y2]. Cancelled or skipped annotations are left out."""
    out = []
    for t in tasks:
        d = t.get("data", {})
        for ann in t.get("annotations", []):
            res = ann.get("result", [])
            if ann.get("was_cancelled") or _skipped(res):
                continue
            boxes = []
            for r in res:
                if r.get("type") != "rectanglelabels":
                    continue
                v, W, H = r["value"], r["original_width"], r["original_height"]
                x1, y1 = v["x"] / 100 * W, v["y"] / 100 * H
                boxes.append([x1, y1, x1 + v["width"] / 100 * W, y1 + v["height"] / 100 * H])
            out.append({"frame_name": d.get("frame_name"), "frame_idx": d.get("frame_idx"), "group": d.get("group"),
                        "overlap": bool(d.get("overlap")), "labeller": _author(ann, names), "boxes": boxes,
                        "width": d.get("width"), "height": d.get("height")})
    return out


def parse_keypoints(tasks: list[dict], names: dict | None = None) -> list[dict]:
    """Keypoint pass export -> one record per (crop task, annotation): the person's
    keypoints in FRAME pixels, xy [17,2] and visibility [17] (2 visible, 1 occluded,
    0 not labelled), the crop's box and the labeller."""
    out = []
    for t in tasks:
        d = t.get("data", {})
        for ann in t.get("annotations", []):
            res = ann.get("result", [])
            if ann.get("was_cancelled") or _skipped(res):
                continue
            xy = [[0.0, 0.0] for _ in COCO17]
            vis = [0] * len(COCO17)
            for r in res:
                if r.get("type") != "keypointlabels":
                    continue
                label = r["value"]["keypointlabels"][0]
                occ = label.endswith(OCCLUDED)
                name = label[:-len(OCCLUDED)] if occ else label
                if name not in COCO17:
                    continue
                k = COCO17.index(name)
                cx = r["value"]["x"] / 100 * r["original_width"]
                cy = r["value"]["y"] / 100 * r["original_height"]
                xy[k] = [d["crop_x0"] + cx / d["crop_scale"], d["crop_y0"] + cy / d["crop_scale"]]
                vis[k] = 1 if occ else 2
            out.append({"frame_name": d["frame_name"], "frame_idx": d.get("frame_idx"), "box": d["box"],
                        "box_id": d["box_id"], "overlap": bool(d.get("overlap")), "labeller": _author(ann, names),
                        "xy": xy, "vis": vis})
    return out
