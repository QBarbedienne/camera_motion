#!/usr/bin/env python3

import sys
from pathlib import Path
import cv2
import numpy as np
import imutils
import time
import src.ExecMode as em
import src.Compute as sl

em.ExecMode = em.MODE_PROD

#sl.BGS_HISTSIZ = 500
#sl.BGS_VARTHRES = 16
#sl.BGS_DOSHDW = True

  # FFD_CREAT_THRES bas ==> len(kp) haut
  # FFD_CREAT_THRES haut ==> len(kp) bas
sl.FFD_CREAT_THRES = 10
sl.FFD_CREAT_NONMAXSUP = True
  # cv2.FAST_FEATURE_DETECTOR_TYPE_5_8
  # cv2.FAST_FEATURE_DETECTOR_TYPE_7_12
  # cv2.FAST_FEATURE_DETECTOR_TYPE_9_16 # default value
sl.FFD_CREAT_TYPE = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
sl.FDD_DETECT_MINFREQ = 3

sl.MAX_MOVLEN = 50
#sl.MINAVG_MOVLEN = 2
sl.MINAVG_MOVLEN = 0

sl.MINTHRES_ANGCNT = 0.9
sl.MINMOV_AREACNT = 5
sl.MINPTS_VAL = 2

FRAME_TEMPO_TEST = 1

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if len(sys.argv) != 2:
  print("Usage: Test_Slam.py <vdfile>")
  exit(1)

vdfile = sys.argv[1]

cap = cv2.VideoCapture(vdfile)
if not cap.isOpened():
    print("Cannot open <" + vdfile + ">")
    exit()

vdbsn = Path(vdfile).stem
print("### Setting default parameters ###")
sl.Y_POS_ORIG = 1
sl.X_POS_ORIG = 0.5
sl.A_POS_ORIG = 1
sl.RL_FACTOR = 0.15
sl.RR_FACTOR = 0.15 
sl.T_FACTOR = 0.3

if em.ExecMode == em.MODE_PROD:   tempo = 1
elif em.ExecMode == em.MODE_TEST: tempo = FRAME_TEMPO_TEST
else:                            tempo = 0
fps = []
iseq = 0
while(True):
  
  init_timer = time.perf_counter()
  iseq += 1
  ret, img_bgr = cap.read()
  if not ret:
    print("Exiting ...")
    break
  
  img_bgr = imutils.resize(img_bgr, width=min(1280, img_bgr.shape[1]))

  sl.Compute_pos(img_bgr)

  if em.ExecMode == em.MODE_DEV:
    print("Tapez une touche pour continuer...")
  key = cv2.waitKey(1)
  if key == ord('q'):
    print("Bye ! :-)")
    break

  if key == ord('r') or iseq == 1:
    if iseq != 1:
      print('reboot')
    sl.reboot_cst()

    FRAME_TEMPO_TEST = 1
  if em.ExecMode == em.MODE_TEST and key == ord(' '):
    print("Tapez une touche pour continuer...")
    cv2.waitKey(0)
  if len(fps) < 100 : 
      fps.append(1/(time.perf_counter()-init_timer))
  else: 
      fps.append(1/(time.perf_counter()-init_timer))
      del fps[0]


print('Mean fps :', np.mean(fps))
print('Median fps:', np.median(fps))
sl.Compute_pos(None, vdfile)

cap.release()
cv2.destroyAllWindows()