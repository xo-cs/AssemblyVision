"""AvError and the error codes and texts of LLD section 7.

Only the codes used so far are listed; the rest are added with their modules.
Texts are English for now and move to i18n/<lang>.toml with the interface.
"""
from __future__ import annotations

# code: (effect, text). Effects as in LLD 7.2.
MESSAGES: dict[str, tuple[str, str]] = {
    "AV-ING-001": ("Run", "The video {name} could not be read. Check that the file is complete and is an mp4 or avi file."),
    "AV-ING-002": ("Run", "The file type {ext} is not supported. Use an mp4 or avi file."),
    "AV-ING-003": ("Run", "The video could not be converted to a constant frame rate. The ffmpeg message is in the log of the ingest stage."),
    "AV-ING-004": ("Run", "Not enough free disk space to convert the video: {free_gb} GB are free and {need_gb} GB are needed."),
    "AV-ING-005": ("Run", "The video is only {dur} s long. At least {min} s are needed to split it into training, validation and test parts."),
    "AV-SHT-001": ("Notice", "Camera cuts could not be detected. The whole video is treated as one handheld shot, so there are no floor paths or heatmap."),
    "AV-SHT-002": ("Notice", "The shot setting is not valid: {detail}."),
    "AV-POS-001": ("Stage", "The pose model {name} could not be loaded. Check that the file exists in the models folder and was copied completely."),
    "AV-POS-002": ("Stage", "The GPU has not enough free memory for model variant {variant}. {action}"),
    "AV-POS-003": ("Item", "Pose estimation failed on frame {frame}. The frame is left empty and the run continues."),
    "AV-POS-004": ("Item", "No worker was found in the first {s} s of shot {shot}. Skip this shot or choose another."),
    "AV-POS-005": ("Refused", "No subject worker was chosen for shot {shot}. Click a worker on the picture, or skip the shot."),
    "AV-TRK-001": ("Notice", "The tracked worker was lost for {s} s at {t} s in shot {shot}. The track is split there and counted in the coverage."),
    "AV-TRK-002": ("Item", "No worker could be tracked for at least {min} s in shot {shot}."),
    "AV-TRK-003": ("Refused", "The chosen point is not on a detected worker in shot {shot}. Click inside a box."),
    "AV-RUN-001": ("Refused", "The video {name} is not in the input folder."),
    "AV-STO-003": ("Stage", "The data file {file} is damaged or has an unexpected format: {detail}."),
    "AV-STO-004": ("Run", "{what} could not be saved: {detail}. No partial result is shown as complete."),
    "AV-STO-005": ("Refused", "The path {path} is outside the data folder and was refused."),
    "AV-CFG-001": ("Refused", "Unknown setting {key} in {file}. Check the spelling."),
    "AV-CFG-002": ("Refused", "The setting {key} = {value} is not valid. Allowed: {range}."),
    "AV-CFG-003": ("Refused", "The configuration file {file} could not be read."),
    "AV-SYS-001": ("Stage", "An unexpected error ({type}) stopped the stage {stage}. The details are in the stage log."),
}


class _Keep(dict):
    """Leaves unknown {fields} visible instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_message(code: str, **fields) -> str:
    return MESSAGES[code][1].format_map(_Keep(fields))


class AvError(Exception):
    """An expected error with an AV-XXX-nnn code.

    fields fill the braces of the text; detail is extra information for the
    log, and also fills {detail} in the texts that have it (AV-STO-003, 004).
    """

    def __init__(self, code: str, detail: str = "", **fields):
        if code not in MESSAGES:
            raise ValueError(f"unknown error code {code}")
        self.code = code
        self.fields = fields
        self.detail = detail
        self.effect = MESSAGES[code][0]
        self.text = format_message(code, detail=detail, **fields)
        super().__init__(f"{code}: {self.text}")

    def as_record(self) -> dict:
        """Entry for the manifest notices and errors."""
        rec = {"code": self.code, "effect": self.effect, "text": self.text}
        if self.detail:
            rec["detail"] = self.detail
        return rec
