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
    "AV-SEG-001": ("Notice", "Engine B is not available because no trained weights were found. Engine A is used."),
    "AV-SEG-002": ("Stage", "Engine {engine} returned steps that are not valid for shot {shot}: {detail}. No steps were stored for this shot."),
    "AV-SEG-003": ("Notice", "No step definitions are set. The footage is cut at pauses and the pieces are named segment_1, segment_2 and so on."),
    "AV-MOT-001": ("Item", "{metric} of step {step} is not reported: only {cov} percent of frames had the keypoints it needs."),
    "AV-MOT-002": ("Item", "{metric} of step {step} is not reported: the step is too short ({dur} s)."),
    "AV-MOT-004": ("Item", "The metric {metric} failed for step {step}. The other metrics are not affected."),
    "AV-EVL-001": ("Stage", "The label file {file} is missing or not in the expected format: {detail}."),
    "AV-EVL-004": ("Item", "There are no labelled {kind} in the {split} part, so no score was calculated."),
    "AV-RUN-001": ("Refused", "The video {name} is not in the input folder."),
    "AV-RUN-004": ("Run", "The analysis process stopped without a result. The run log shows the last step it finished."),
    "AV-RUN-006": ("Notice", "{what} {value} is not supported. {default} is used instead."),
    "AV-RUN-007": ("Refused", "This run is already being processed."),
    "AV-LIV-001": ("Refused", "No camera image arrived. Allow camera access in the browser and open the page as http://127.0.0.1:{port}."),
    "AV-LIV-002": ("Refused", "A live session is already running. Stop it first."),
    "AV-LIV-003": ("Notice", "The template {name} could not be read ({detail}). Live mode cuts segments at pauses instead."),
    "AV-LIV-004": ("Notice", "{text}"),
    "AV-LIV-005": ("Refused", "The live pose model could not be loaded: {free} MB free on GPU 1, {need} MB needed."),
    "AV-LIV-006": ("Notice", "Live processing reaches only {fps} frames per second, so the overlay is behind the camera."),
    "AV-LIV-007": ("Notice", "The recording could not be written ({detail}). Live analysis continues without it."),
    "AV-LIV-008": ("Refused", "This session has no recording, so it cannot be analysed as a batch run."),
    "AV-LIV-009": ("Refused", "Click inside a person's box to select who is analysed."),
    "AV-APP-001": ("Refused", "Only mp4 and avi files can be uploaded."),
    "AV-APP-002": ("Refused", "Select or upload a video first."),
    "AV-APP-003": ("Refused", "The run {run_id} was not found. Refresh the list."),
    "AV-APP-004": ("Refused", "Click a worker in the picture, choose a box number, or skip the shot."),
    "AV-APP-005": ("Refused", "The interface was not started. It may only listen on this computer or on a private network address, and share links are not allowed."),
    "AV-APP-006": ("Notice", "{what} {value} is not supported. {default} is used instead."),
    "AV-APP-007": ("Refused", "This page is read only. Start an analysis from the normal interface."),
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
