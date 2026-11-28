# Pose labelling guide (AssemblyVision, Week 10)

## 1. Data rules (read first, every session)
- **No screenshots**, no photos of the screen, no screen recording.
- **No downloads of frames.** Do not save, copy or drag an image out of the browser. Label Studio holds no images and can export only the label JSON. Only Khurshid exports.
- **Use the frames for this labelling task only.** Do not show, describe or discuss their content outside the team.
- **After each session the browser cache is cleared** and the tunnel is closed (Khurshid does both, see section 2).

## 2. Access
Everything runs on the server. Labelling happens only in the browser **on Khurshid's laptop**, through **his** SSH tunnel. There are no server accounts for labellers.
1. **Khurshid starts** Label Studio (`scripts/labelstudio.sh start qisens`) and the frame server (`python -m assemblyvision.tools.label_frames_server --run <run>`). Both listen on 127.0.0.1 only.
2. **He opens the tunnel** from his laptop: `ssh -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081 <server>`.
3. **You log in at http://127.0.0.1:8080 with your own Label Studio login** (Khurshid, Asel and Kazi each have one), so every label records who made it. Log out when you finish.
4. **Khurshid closes the session:** he logs everyone out, stops both servers, closes the tunnel and clears the laptop browser's cache (cached images and files, all time).

## 3. Pass 1: boxes (project "boxes")
- **Your tasks:** label the tasks of your group, A or B. Tasks marked `overlap` are labelled by both of you.
- **Draw one box per person**, including people who are partly hidden or cut by the image border. A box goes from the top of the head to the lowest visible point, and from the leftmost to the rightmost visible point.
- **Skip only people too small to recognise** (about under 30 px tall) and reflections or people on screens.
- **Mark a frame "unusable"** only if it is blurred, cut or black.

## 4. Pass 2: keypoints (project "keypoints")
- **One crop per box.** Label only the person in the **middle** of the crop. If the crop does not show one, mark it "not the person in the middle".
- **Mark the 17 COCO keypoints.** **Left and right are the worker's own side**: the worker's left shoulder is usually on the right of the image when they face the camera.

  | # | keypoint | # | keypoint | # | keypoint |
  | --- | --- | --- | --- | --- | --- |
  | 0 | nose | 6 | right_shoulder | 12 | right_hip |
  | 1 | left_eye | 7 | left_elbow | 13 | left_knee |
  | 2 | right_eye | 8 | right_elbow | 14 | right_knee |
  | 3 | left_ear | 9 | left_wrist | 15 | left_ankle |
  | 4 | right_ear | 10 | right_wrist | 16 | right_ankle |
  | 5 | left_shoulder | 11 | left_hip | | |

  ```
         (1)(0)(2)          worker facing the camera
       (3)       (4)        image left = worker's RIGHT
     (6)-----------(5)
     |               |
    (8)             (7)
    |                 |
  (10)    (12)(11)   (9)
           |    |
          (14) (13)
           |    |
          (16) (15)
  ```
- **Visible:** use the plain label (for example `left_wrist`) and click the joint centre.
- **Occluded but clear:** if the joint is hidden (behind a part, a tool or another person) but you can place it within about one hand width, use the `(occluded)` label, for example `left_wrist (occluded)`.
- **Not placeable:** if the joint is outside the image or you cannot tell where it is, **leave it out**.
- **Where to click:**
  - joints: on the joint centre, not the skin edge;
  - hips: on the visible hip point at the side of the pelvis (not the belt buckle);
  - ankles: on the ankle bone;
  - ears: on the ear, even under a cap, or occluded under a helmet.
- **Zoom in** (mouse wheel) for wrists and ankles. About 30 s per person is normal.

## 5. Quality
- **Check before submitting:** count the boxes (pass 1), and in pass 2 check that left and right are not swapped.
- **Overlap tasks:** do not discuss them with the other labeller before both are done. They measure our agreement.
- **Unsure cases:** if you are unsure about a case, skip the task and note its frame number for Khurshid. Do not guess.

## 6. Measurement labels in the app (Labels page)

For the team's accounts only (the mentor's account does not see this page). Open **Labels** in the app, choose the task
and the run, press **Open**. Label alone and do not look at the system's result first. The rules of section 1 apply: no
screenshots, no downloads (the page has no download button), clear the browser cache after a QiSens session.

**Frames (100 frames), three questions per frame:**
1. **Trunk: two clicks, not a guess.** Nobody can judge 20, 45 or 60 degrees by eye, so the angle is measured from your
   clicks. Click (1) the **midpoint of the hips** (halfway between the two hip joints), then (2) the **midpoint of the
   shoulders** (halfway between the two shoulder joints). The page draws the line between them and a thin vertical line
   through the hip point, and shows the angle in degrees and its band: B0 under 20, B1 20 to 45, B2 45 to 60, B3 over 60.
   If a point is off, press **Redo clicks**. If the hips or the shoulders cannot be seen, press **Trunk: cannot see**.
   The angle is computed exactly like the system's (the line from the hip midpoint to the shoulder midpoint against the
   vertical of the image), so it is an angle in the image: bending toward or away from the camera looks smaller.
2. **Are both hands working?** yes, no, or cannot see.
3. **Gaze direction:** towards the hands or the work, left, right, up, away from the camera, or cannot see. Left and right
   are as seen in the image.

**Wrists (50 frames):** click the worker's left wrist (their own left), then the right one; or mark one not visible.

**Events (one 2 minute clip):** play the clip and press **Cycle start** when the same work cycle begins again, **Tool
pickup** when a hand closes on a tool, **Object placement** when a hand lets go of a placed object.

Asel and Kazi each do every item, so the agreement between them can be reported.

