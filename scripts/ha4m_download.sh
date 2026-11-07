#!/usr/bin/env bash
# HA4M download in order, one paced stream (Week 10). Run it in the tmux window "dl":
#   tmux new-window -d -n dl ~/AssemblyVision/scripts/ha4m_download.sh
# 0. wait while another HA4M download holds the lock (it finishes first)
# 1. annotations of every video (done videos are skipped, failed ones retried, up to 3 passes)
# 2. the domain gap frames: 27 s at every 3rd frame of IDU001V001 and IDU023V001 (about 1.5 GB)
# Meanwhile the Engine B data cache is extended every 10 min with the videos that arrived.
# The RGB route (pilot and 24 video subset) is cancelled. Progress and HTTP 429 waits go
# to logs/ha4m_download.log; every step resumes per video or per frame.
set -u
source ~/AssemblyVision/av.sh >/dev/null 2>&1
cd ~/AssemblyVision/code
LOG=~/AssemblyVision/logs/ha4m_download.log
LOCK=~/AssemblyVision/data/public/datasets/ha4m/.download.lock
DONE=$(mktemp -u ~/AssemblyVision/tmp/ha4m_dl_done.XXXX)
note() { echo "$(date '+%F %T') INFO ha4m_download.sh: $*" | tee -a "$LOG"; }
X="python -m assemblyvision.tools.fetch_dataset ha4m"
build() { python -m assemblyvision.tools.engine_b_data ha4m 2>&1 | tee -a "$LOG"; }

while ! flock -n "$LOCK" true; do sleep 60; done          # an earlier pass still running
note "start: annotations, then the domain gap frames (one stream); Engine B cache every 10 min"
( while [ ! -e "$DONE" ]; do build >/dev/null; sleep 600; done ) &
builder=$!

ok=0
for pass in 1 2 3; do
    $X annotations --videos all
    code=$?
    if [ $code -eq 0 ]; then ok=1; break; fi
    if [ $code -eq 2 ]; then note "annotations refused (exit 2), stopping"; touch "$DONE"; exit 2; fi
    note "annotations pass $pass ended with failures; next pass in 10 min"
    sleep 600
done
touch "$DONE"; wait $builder 2>/dev/null; rm -f "$DONE"
build
if [ $ok -ne 1 ]; then note "annotations still incomplete after 3 passes"; fi
note "domain gap frames next"
python -m assemblyvision.tools.ha4m_domain_gap fetch
note "domain gap fetch ended with exit $?; the YOLO run waits for approval (GPU job)"
