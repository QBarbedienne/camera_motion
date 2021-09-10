import cv2
import numpy as np
import src.ExecMode as em
import src.StatPerf as sp
if em.ExecMode == em.MODE_DEV:
  from matplotlib import pyplot as plt

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
  #-----------------------------------------------------------------------------
  # Paramètres par defaut modifiables
  # (ATTENTION: le positionnement dans le script appelant prévaut)
  #-----------------------------------------------------------------------------
#BGS_HISTSIZ = 500
#BGS_VARTHRES = 16
#BGS_DOSHDW = True

def reboot_cst():
  global BGS, FFD, CYCLE, IM_YSIZ, IM_XSIZ
  global IM_FFD_PREV, PT_PREV, MOV_HISTO
  global IM_CAM, Y_CAM, X_CAM, A_CAM
  global A_POS_ORIG, Y_POS_ORIG, X_POS_ORIG

  FFD_CREAT_THRES = 50
  FFD_CREAT_NONMAXSUP = True
  # cv2.FAST_FEATURE_DETECTOR_TYPE_5_8
  # cv2.FAST_FEATURE_DETECTOR_TYPE_7_12
  FFD_CREAT_TYPE = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16 # default value
  FDD_DETECT_MINFREQ = 10

  MAX_MOVLEN = 50
  MINAVG_MOVLEN = 2
  MINTHRES_ANGCNT = 0.5
  MINMOV_AREACNT = 5
  MINPTS_VAL = 2

  Y_POS_ORIG = 0.5
  X_POS_ORIG = 0.5
  A_POS_ORIG = 1

  RL_FACTOR = 0.01
  RR_FACTOR = 0.01 
  T_FACTOR = 0.3

    #-----------------------------------------------------------------------------
    # Constantes
    #-----------------------------------------------------------------------------
  ARMT_YSIZ = 3
  ARMT_XSIZ = 3

  ARMT_Y1 = 0; ARMT_Y2 = 1; ARMT_X1 = 2; ARMT_X2 = 3
  PTTAB_X1 = 0; PTTAB_Y1 = 1; PTTAB_X2 = 2; PTTAB_Y2 = 3; 

  NB_PTS = 4
  NB_DIR = 8
  #NB_DIR = 16
  ANG_PTSSEP = 360 / NB_DIR

  NB_MVT = 12
  MVT_TU = 0
  MVT_TD = 1
  MVT_TL = 2
  MVT_TR = 3
  MVT_TF = 4
  MVT_TB = 5
  MVT_RF = 6
  MVT_RB = 7
  MVT_RL = 8
  MVT_RR = 9
  MVT_RH = 10
  MVT_RA = 11

  MVT_DEF_TAB = np.zeros((NB_MVT, NB_PTS), np.uint8)
  MVT_NAM_TAB = [None, None, None, None, None, None, None, None, None, None, None, None, ]

  if NB_DIR == 8:
    MVT_NAM_TAB[MVT_TU] = 'TRS UP'
    MVT_DEF_TAB[MVT_TU] = [ 2,  2,  2,  2]
    MVT_NAM_TAB[MVT_TD] = 'TRS DOWN'
    MVT_DEF_TAB[MVT_TD] = [ 6,  6,  6,  6]
    MVT_NAM_TAB[MVT_TL] = 'TRS LEFT'
    MVT_DEF_TAB[MVT_TL] = [ 4,  4,  4,  4]
    MVT_NAM_TAB[MVT_TR] = 'TRS RIGHT'
    MVT_DEF_TAB[MVT_TR] = [ 0,  0,  0,  0]
    MVT_NAM_TAB[MVT_TF] = 'TRS FORWARD'
    MVT_DEF_TAB[MVT_TF] = [ 5,  7,  1,  3]
    MVT_NAM_TAB[MVT_TB] = 'TRS BACK'
    MVT_DEF_TAB[MVT_TB] = [ 1,  3,  5,  7]
    MVT_NAM_TAB[MVT_RF] = 'ROT FORWARD'
    MVT_DEF_TAB[MVT_RF] = [ 5,  7,  5,  7]
    MVT_NAM_TAB[MVT_RB] = 'ROT BACK'
    MVT_DEF_TAB[MVT_RB] = [ 1,  3,  1,  3]
    MVT_NAM_TAB[MVT_RL] = 'ROT LEFT'
    MVT_DEF_TAB[MVT_RL] = [ 5,  3,  5,  3]
    MVT_NAM_TAB[MVT_RR] = 'ROT RIGHT'
    MVT_DEF_TAB[MVT_RR] = [ 1,  7,  1,  7]
    MVT_NAM_TAB[MVT_RH] = 'ROT HORAIRE'
    MVT_DEF_TAB[MVT_RH] = [ 7,  1,  3,  5]
    MVT_NAM_TAB[MVT_RA] = 'ROT ANTIHORAIRE'
    MVT_DEF_TAB[MVT_RA] = [ 3,  5,  7,  1]

  if NB_DIR == 16:
    MVT_NAM_TAB[MVT_TU] = 'TRS UP'
    MVT_DEF_TAB[MVT_TU] = [ 4,  4,  4,  4]
    MVT_NAM_TAB[MVT_TD] = 'TRS DOWN'
    MVT_DEF_TAB[MVT_TD] = [12, 12, 12, 12]
    MVT_NAM_TAB[MVT_TL] = 'TRS LEFT'
    MVT_DEF_TAB[MVT_TL] = [ 8,  8,  8,  8]
    MVT_NAM_TAB[MVT_TR] = 'TRS RIGHT'
    MVT_DEF_TAB[MVT_TR] = [ 0,  0,  0,  0]
    MVT_NAM_TAB[MVT_TF] = 'TRS FORWARD'
    MVT_DEF_TAB[MVT_TF] = [10, 14,  2,  6]
    MVT_NAM_TAB[MVT_TB] = 'TRS BACK'
    MVT_DEF_TAB[MVT_TB] = [ 2,  6, 10, 14]
    MVT_NAM_TAB[MVT_RF] = 'ROT FORWARD'
    MVT_DEF_TAB[MVT_RF] = [11, 13, 11, 13]
    MVT_NAM_TAB[MVT_RB] = 'ROT BACK'
    MVT_DEF_TAB[MVT_RB] = [ 3,  5,  3,  5]
    MVT_NAM_TAB[MVT_RL] = 'ROT LEFT'
    MVT_DEF_TAB[MVT_RL] = [ 9,  7,  9,  7]
    MVT_NAM_TAB[MVT_RR] = 'ROT RIGHT'
    MVT_DEF_TAB[MVT_RR] = [ 1, 15,  1, 15]
    MVT_NAM_TAB[MVT_RH] = 'ROT HORAIRE'
    MVT_DEF_TAB[MVT_RH] = [13,  1,  5,  9]
    MVT_NAM_TAB[MVT_RA] = 'ROT ANTIHORAIRE'
    MVT_DEF_TAB[MVT_RA] = [ 7, 11, 15,  3]

  CYCLE = 0
  FFD = None
  BGS = None
  IM_YSIZ = None
  IM_XSIZ = None
  ARMT = None
  AR_YSIZ = None
  AR_XSIZ = None

  IM_FFD_PREV = None
  PT_PREV = None

  MOV_HISTO = np.zeros((ARMT_YSIZ, ARMT_XSIZ, 0, 2), np.uint16)

  IM_CAM = None
  Y_CAM = None
  X_CAM = None
  A_CAM = None
  H_CAM = np.zeros((0, 2), np.uint16)

FFD_CREAT_THRES = 50
FFD_CREAT_NONMAXSUP = True
# cv2.FAST_FEATURE_DETECTOR_TYPE_5_8
# cv2.FAST_FEATURE_DETECTOR_TYPE_7_12
FFD_CREAT_TYPE = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16 # default value
FDD_DETECT_MINFREQ = 10

MAX_MOVLEN = 50
MINAVG_MOVLEN = 2
MINTHRES_ANGCNT = 0.5
MINMOV_AREACNT = 5
MINPTS_VAL = 2

Y_POS_ORIG = 0.5
X_POS_ORIG = 0.5
A_POS_ORIG = 1

RL_FACTOR = 0.01
RR_FACTOR = 0.01 
T_FACTOR = 0.3

  #-----------------------------------------------------------------------------
  # Constantes
  #-----------------------------------------------------------------------------
ARMT_YSIZ = 3
ARMT_XSIZ = 3

ARMT_Y1 = 0; ARMT_Y2 = 1; ARMT_X1 = 2; ARMT_X2 = 3
PTTAB_X1 = 0; PTTAB_Y1 = 1; PTTAB_X2 = 2; PTTAB_Y2 = 3; 

NB_PTS = 4
NB_DIR = 8
#NB_DIR = 16
ANG_PTSSEP = 360 / NB_DIR

NB_MVT = 12
MVT_TU = 0
MVT_TD = 1
MVT_TL = 2
MVT_TR = 3
MVT_TF = 4
MVT_TB = 5
MVT_RF = 6
MVT_RB = 7
MVT_RL = 8
MVT_RR = 9
MVT_RH = 10
MVT_RA = 11

MVT_DEF_TAB = np.zeros((NB_MVT, NB_PTS), np.uint8)
MVT_NAM_TAB = [None, None, None, None, None, None, None, None, None, None, None, None, ]

if NB_DIR == 8:
  MVT_NAM_TAB[MVT_TU] = 'TRS UP'
  MVT_DEF_TAB[MVT_TU] = [ 2,  2,  2,  2]
  MVT_NAM_TAB[MVT_TD] = 'TRS DOWN'
  MVT_DEF_TAB[MVT_TD] = [ 6,  6,  6,  6]
  MVT_NAM_TAB[MVT_TL] = 'TRS LEFT'
  MVT_DEF_TAB[MVT_TL] = [ 4,  4,  4,  4]
  MVT_NAM_TAB[MVT_TR] = 'TRS RIGHT'
  MVT_DEF_TAB[MVT_TR] = [ 0,  0,  0,  0]
  MVT_NAM_TAB[MVT_TF] = 'TRS FORWARD'
  MVT_DEF_TAB[MVT_TF] = [ 5,  7,  1,  3]
  MVT_NAM_TAB[MVT_TB] = 'TRS BACK'
  MVT_DEF_TAB[MVT_TB] = [ 1,  3,  5,  7]
  MVT_NAM_TAB[MVT_RF] = 'ROT FORWARD'
  MVT_DEF_TAB[MVT_RF] = [ 5,  7,  5,  7]
  MVT_NAM_TAB[MVT_RB] = 'ROT BACK'
  MVT_DEF_TAB[MVT_RB] = [ 1,  3,  1,  3]
  MVT_NAM_TAB[MVT_RL] = 'ROT LEFT'
  MVT_DEF_TAB[MVT_RL] = [ 5,  3,  5,  3]
  MVT_NAM_TAB[MVT_RR] = 'ROT RIGHT'
  MVT_DEF_TAB[MVT_RR] = [ 1,  7,  1,  7]
  MVT_NAM_TAB[MVT_RH] = 'ROT HORAIRE'
  MVT_DEF_TAB[MVT_RH] = [ 7,  1,  3,  5]
  MVT_NAM_TAB[MVT_RA] = 'ROT ANTIHORAIRE'
  MVT_DEF_TAB[MVT_RA] = [ 3,  5,  7,  1]

if NB_DIR == 16:
  MVT_NAM_TAB[MVT_TU] = 'TRS UP'
  MVT_DEF_TAB[MVT_TU] = [ 4,  4,  4,  4]
  MVT_NAM_TAB[MVT_TD] = 'TRS DOWN'
  MVT_DEF_TAB[MVT_TD] = [12, 12, 12, 12]
  MVT_NAM_TAB[MVT_TL] = 'TRS LEFT'
  MVT_DEF_TAB[MVT_TL] = [ 8,  8,  8,  8]
  MVT_NAM_TAB[MVT_TR] = 'TRS RIGHT'
  MVT_DEF_TAB[MVT_TR] = [ 0,  0,  0,  0]
  MVT_NAM_TAB[MVT_TF] = 'TRS FORWARD'
  MVT_DEF_TAB[MVT_TF] = [10, 14,  2,  6]
  MVT_NAM_TAB[MVT_TB] = 'TRS BACK'
  MVT_DEF_TAB[MVT_TB] = [ 2,  6, 10, 14]
  MVT_NAM_TAB[MVT_RF] = 'ROT FORWARD'
  MVT_DEF_TAB[MVT_RF] = [11, 13, 11, 13]
  MVT_NAM_TAB[MVT_RB] = 'ROT BACK'
  MVT_DEF_TAB[MVT_RB] = [ 3,  5,  3,  5]
  MVT_NAM_TAB[MVT_RL] = 'ROT LEFT'
  MVT_DEF_TAB[MVT_RL] = [ 9,  7,  9,  7]
  MVT_NAM_TAB[MVT_RR] = 'ROT RIGHT'
  MVT_DEF_TAB[MVT_RR] = [ 1, 15,  1, 15]
  MVT_NAM_TAB[MVT_RH] = 'ROT HORAIRE'
  MVT_DEF_TAB[MVT_RH] = [13,  1,  5,  9]
  MVT_NAM_TAB[MVT_RA] = 'ROT ANTIHORAIRE'
  MVT_DEF_TAB[MVT_RA] = [ 7, 11, 15,  3]

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
def ShowImg(title, img, mode):
#-------------------------------------------------------------------------------
#  if em.ExecMode == em.MODE_PROD:
#    return
    
  cv2.imshow(title, img)

  if em.ExecMode == em.MODE_DEV:
    tempo = 0
    print("### TYPE A KEY ! ###")
  else: # MODE_TEST
    tempo = 1
  cv2.waitKey(tempo)

#-------------------------------------------------------------------------------
def ShowCurve(curv, suptitle):
#-------------------------------------------------------------------------------
  if em.ExecMode != em.MODE_DEV: return

  xsize = curv.size
  xaxe = np.arange(0, xsize, 1)

  plt.figure(suptitle)
  plt.subplot(111)
  plt.plot(xaxe, curv)
  plt.show()

#-------------------------------------------------------------------------------
def PrintDev(lib, val):
#-------------------------------------------------------------------------------
  if em.ExecMode != em.MODE_DEV: return
  if val is not None: print(lib, val)
  else: print(lib)
  return
  
#-------------------------------------------------------------------------------
def InitAreaMat(imysiz, imxsiz):
#-------------------------------------------------------------------------------
  global ARMT, AR_YSIZ, AR_XSIZ
  
  ARMT = np.zeros((ARMT_YSIZ, ARMT_XSIZ, 4), np.uint16)
  AR_YSIZ = imysiz // ARMT_YSIZ
  AR_XSIZ = imxsiz // ARMT_XSIZ
  y = 0
  while y < ARMT_YSIZ:
    x = 0
    while x < ARMT_XSIZ:
      ARMT[y][x][ARMT_Y1] = y * AR_YSIZ
      ARMT[y][x][ARMT_Y2] = (y + 1) * AR_YSIZ
      ARMT[y][x][ARMT_X1] = x * AR_XSIZ
      ARMT[y][x][ARMT_X2] = (x + 1) * AR_XSIZ
      x += 1
    y += 1
  PrintDev("ARMT", ARMT)
  return

#-------------------------------------------------------------------------------
def DrawCam(libmov):
#-------------------------------------------------------------------------------
  global IM_CAM, A_CAM, H_CAM

  crclrad = 50
  IM_CAM[:,:] = 0
  posgridtab = np.linspace(0, IM_YSIZ - 4, 4, True)
  for pos in posgridtab:
    pos = int(pos)
    cv2.line(IM_CAM, (0, pos), (IM_XSIZ - 1, pos), 127, 2)
  posgridtab = np.linspace(0, IM_XSIZ - 4, 4, True)
  for pos in posgridtab:
    pos = int(pos)
    cv2.line(IM_CAM, (pos, 0), (pos, IM_YSIZ - 1), 127, 2)

  for x, y in H_CAM:
    cv2.circle(IM_CAM, (x, y), 3, 123, -1)
    
  cv2.circle(IM_CAM, (X_CAM, Y_CAM), crclrad, 255, 1)
 
  if A_CAM > 360: A_CAM = A_CAM - 360
  if A_CAM < 0:   A_CAM = 360 - A_CAM
  avgang = A_CAM
  if avgang > 180: avgang -= 360
  avgang *= (np.pi / 180)
  linx = int(X_CAM + (np.cos(avgang) * crclrad))
  liny = int(Y_CAM - (np.sin(avgang) * crclrad))
  cv2.line(IM_CAM, (X_CAM, Y_CAM), (linx, liny), 255, 5)

  IM_CAM[0:20][:] = 0
  cv2.putText(IM_CAM, libmov, (0, 15),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

  H_CAM = np.append(H_CAM, [[X_CAM, Y_CAM]], 0)
 
  ShowImg("CAM", IM_CAM, em.MODE_TEST)

#-------------------------------------------------------------------------------
def ShowMovesCurves(vdfile):
#-------------------------------------------------------------------------------
    def smooth(y, box_pts):
      box = np.ones(box_pts)/box_pts
      y_smooth = np.convolve(y, box, mode='same')
      return y_smooth

    smoothval = 15
    if MOV_HISTO.shape[2] < smoothval:
      return
      
    plt.figure(vdfile, figsize = (24, 15))
    xaxe = np.arange(0, MOV_HISTO.shape[2], 1)
    y = 0
    while y < ARMT_YSIZ:
      x = 0
      while x < ARMT_XSIZ:
        lenangtab = MOV_HISTO[y][x].ravel(order = 'F').reshape(2, -1)
        plt.subplot(ARMT_YSIZ, ARMT_YSIZ, y * ARMT_XSIZ + x + 1)

        lentab = np.int16(lenangtab[0])
        plt.plot(xaxe, smooth(lentab, smoothval), label = 'Intensité')

        angtab = np.float64(\
                np.where(lenangtab[1] > 180, lenangtab[1] - 360, lenangtab[1]))
        angtab *= (np.pi / 180)
        
        sintab = np.int16(np.sin(angtab) * MAX_MOVLEN)
        plt.plot(xaxe, smooth(sintab, smoothval), label = 'sin(theta)')
        
        costab = np.int16(np.cos(angtab) * MAX_MOVLEN)
        plt.plot(xaxe, smooth(costab, smoothval), label = 'cos(theta)')

        plt.legend()
        plt.title('Area %d %d'%(y, x));
        x += 1
      y += 1

    plt.tight_layout()
    plt.show()
    return

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

CYCLE = 0
FFD = None
BGS = None
IM_YSIZ = None
IM_XSIZ = None
ARMT = None
AR_YSIZ = None
AR_XSIZ = None

IM_FFD_PREV = None
PT_PREV = None

MOV_HISTO = np.zeros((ARMT_YSIZ, ARMT_XSIZ, 0, 2), np.uint16)

IM_CAM = None
Y_CAM = None
X_CAM = None
A_CAM = None
H_CAM = np.zeros((0, 2), np.uint16)

#-------------------------------------------------------------------------------
def Compute_pos(im_bgr, vdfile = None):
#-------------------------------------------------------------------------------
  global BGS, FFD, CYCLE, IM_YSIZ, IM_XSIZ
  global IM_FFD_PREV, PT_PREV, MOV_HISTO
  global IM_CAM, Y_CAM, X_CAM, A_CAM
  global A_POS_ORIG, Y_POS_ORIG, X_POS_ORIG

    #---------------------------------------------------------------------------
    # Appel final: affichage graphe historique rotations/amplitudes déplacements
    #---------------------------------------------------------------------------
  if im_bgr is None:
    ShowMovesCurves(vdfile)
    return

    #---------------------------------------------------------------------------
    # Nouveau cycle
    #---------------------------------------------------------------------------
  CYCLE += 1
  PrintDev("CYCLE", CYCLE)
    
    #---------------------------------------------------------------------------
    # 1er cycle : initialisation
    #---------------------------------------------------------------------------
  if CYCLE == 1:
        #-----------------------------------------------------------------------
        # Création objet BackgroundSubtractorMOG2 : 
        # Instancié ici pour prendre en compte la modification de paramètres
        # par le scrip appelant
        #-----------------------------------------------------------------------
#    BGS = cv2.createBackgroundSubtractorMOG2(BGS_HISTSIZ, BGS_VARTHRES,
#                                                                    BGS_DOSHDW)
        #-----------------------------------------------------------------------
        # Création objet FastFeatureDetector : 
        # Instancié ici pour prendre en compte la modification de paramètres
        # par le scrip appelant
        #-----------------------------------------------------------------------
    FFD = cv2.FastFeatureDetector_create(FFD_CREAT_THRES, FFD_CREAT_NONMAXSUP,
                                                                FFD_CREAT_TYPE)
        #-----------------------------------------------------------------------
        # Taille de l'image et matrice des zones
        #-----------------------------------------------------------------------
    IM_YSIZ, IM_XSIZ = im_bgr.shape[:2]
    InitAreaMat(IM_YSIZ, IM_XSIZ)

        #-----------------------------------------------------------------------
        # Image et position initiale de la camera
        #-----------------------------------------------------------------------
    IM_CAM = np.zeros((IM_YSIZ, IM_XSIZ), np.uint8)
    Y_CAM = int((IM_YSIZ - 1) * Y_POS_ORIG)
    X_CAM = int((IM_XSIZ - 1) * X_POS_ORIG)
    A_CAM = int(90 * A_POS_ORIG)
    DrawCam("=== Init ===")
  try:
    ShowImg("im_bgrud", im_bgr, em.MODE_TEST)
  except:
    pass
    #---------------------------------------------------------------------------
    # Début SLAM
    #---------------------------------------------------------------------------
  sp.SP_BegStep("Slam")
    
        #-----------------------------------------------------------------------
        # Suppression background
        #-----------------------------------------------------------------------
#  im_ffd_curr = BGS.apply(im_bgrud)
#  ShowImg("No background", im_ffd_curr, em.MODE_TEST)
  
        #-----------------------------------------------------------------------
        # Conversion image en NB
        #-----------------------------------------------------------------------
  im_ffd_curr = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

        #-----------------------------------------------------------------------
        # Détection des KPs sur la frame précédente
        # Principe : la méthode "Detect" est appelée pour la détection initiale
        # des KPs puis la méthode "calcOpticalFlowPyrLK" pour la détection des 
        # déplacements. "calcOptical" retourne la map des KPs trouvés sur la
        # frame courante. Cette map peut être utilisée comme map précédente
        # pour le cycle suivant. Mais du fait des mouvements de nombreux KPs
        # disparaissent au fil des frames : il faut alors rappeller "Detect"
        # régulièrement pour réinitialiser la map des KPs.
        #-----------------------------------------------------------------------
  if PT_PREV is None or CYCLE % FDD_DETECT_MINFREQ == 1:
    if CYCLE == 1: im = im_ffd_curr
    else: im = IM_FFD_PREV
        # mask:	Mask specifying where to look for keypoints (optional). 
    PrintDev("==> FFD.detect", None)
    kpprev = FFD.detect(im, mask=None)
    PT_PREV = cv2.KeyPoint_convert(kpprev)

        #-----------------------------------------------------------------------
        # 1er cycle: pas de détection possible
        #-----------------------------------------------------------------------
  if CYCLE == 1:
    IM_FFD_PREV = im_ffd_curr
    return

        #-----------------------------------------------------------------------
        # Nombre de KPs trouvés sur la frame précédente
        #-----------------------------------------------------------------------
  kpprevcnt = len(PT_PREV)
  if kpprevcnt <= 0:
    IM_FFD_PREV = im_ffd_curr
    PT_PREV = None
    PrintDev("=== No previous keypoint found ===", None)
    return
  PrintDev("kpprev count", kpprevcnt)

        #-----------------------------------------------------------------------
        # Détection des mouvements des KPs entre frame précédente/courante
        #-----------------------------------------------------------------------
  ptcurrtab, status, err = cv2.calcOpticalFlowPyrLK(IM_FFD_PREV, im_ffd_curr,
                                                                  PT_PREV, None)
        #-----------------------------------------------------------------------
        # Nombre de mouvements trouvés (appairages KPs ok)
        #-----------------------------------------------------------------------
  movokcnt = np.count_nonzero(status)
  if movokcnt <= 0:
    IM_FFD_PREV = im_ffd_curr
    PT_PREV = None
    PrintDev("=== No movement found ===", None)
    return
  PrintDev("status ok count", movokcnt)
    
        #-----------------------------------------------------------------------
        # Suppressions des KPs non appairables
        #-----------------------------------------------------------------------
  status = np.broadcast_to(status, (status.shape[0], 2))
  ptprevtab = PT_PREV[status == [1, 1]].reshape(-1, 2)
  ptcurrtab = ptcurrtab[status == [1, 1]].reshape(-1, 2)
    
        #-----------------------------------------------------------------------
        # Fusion tables des KPs précédents/courants
        # KPPREV = [[x1a, y1a], [x1b, y1b], ...]
        # KPCURR = [[x2a, y2a], [x2b, y2b], ...]
        # ===>
        # KPTAB = [[x1a, x1b, x1c, ...],
        #          [y1a, y1b, y1c, ...],
        #          [x2a, x2b, x2c, ...],
        #          [y2a, y2b, y2c, ...]]
        #-----------------------------------------------------------------------
  pttab = np.int16(
                np.hstack((ptprevtab, ptcurrtab)).
                ravel(order = 'F').
                reshape(4, -1))
  #PrintDev("pttab", pttab)

        #-----------------------------------------------------------------------
        # Mémorisation datas pour le cycle suivant
        #-----------------------------------------------------------------------
  IM_FFD_PREV = im_ffd_curr
  PT_PREV = ptcurrtab
    
        #-----------------------------------------------------------------------
        # Calcul tables des composantes globales à partir de la table des KPs
        #-----------------------------------------------------------------------
            #-------------------------------------------------------------------
            # Calcul mouvements X et Y
            #-------------------------------------------------------------------
  xmovtab = pttab[PTTAB_X1] - pttab[PTTAB_X2]
              # Inverse l'axe des Y de bas en haut
  ymovtab = pttab[PTTAB_Y2] - pttab[PTTAB_Y1]
    
            #-------------------------------------------------------------------
            # Calcul amplitudes mouvements
            #-------------------------------------------------------------------
  lmovtab = np.around(np.hypot(xmovtab, ymovtab), 2)
    
            #-------------------------------------------------------------------
            # Calcul angles mouvements en ° (tronqués à l'unité)
            #-------------------------------------------------------------------
  amovtab = np.int64(np.arctan2(ymovtab, xmovtab) * 180 / np.pi)
  amovtab = np.where(amovtab < 0, 360 + amovtab, amovtab)

            #-------------------------------------------------------------------
            # Calcul areas d'appartenance des mouvements
            #-------------------------------------------------------------------
  aowntab = np.zeros((ARMT_YSIZ, ARMT_XSIZ, movokcnt), np.bool_)
  y = 0
  while y < ARMT_YSIZ:
    x = 0
    while x < ARMT_XSIZ:
      cond  = ((pttab[PTTAB_X1] >= ARMT[y][x][ARMT_X1]) & \
               (pttab[PTTAB_X1] <  ARMT[y][x][ARMT_X2]) & \
               (pttab[PTTAB_Y1] >= ARMT[y][x][ARMT_Y1]) & \
               (pttab[PTTAB_Y1] <  ARMT[y][x][ARMT_Y2])) \
               | \
              ((pttab[PTTAB_X2] >= ARMT[y][x][ARMT_X1]) & \
               (pttab[PTTAB_X2] <  ARMT[y][x][ARMT_X2]) & \
               (pttab[PTTAB_Y2] >= ARMT[y][x][ARMT_Y1]) & \
               (pttab[PTTAB_Y2] <  ARMT[y][x][ARMT_Y2]))
      np.copyto(aowntab[y][x], cond)
      x += 1
    y += 1

        #-----------------------------------------------------------------------
        # Calcul conditions d'extraction des composantes pour chaque area
        #-----------------------------------------------------------------------  y = 0
  condarea = np.zeros((ARMT_YSIZ, ARMT_XSIZ, movokcnt), np.bool_)
  movareacnt = np.zeros((ARMT_YSIZ, ARMT_XSIZ), np.int64)
  
  y = 0
  while y < ARMT_YSIZ:
    x = 0
    while x < ARMT_XSIZ:
      #print("Conditions area (%d, %d)"%(y, x))
            #-------------------------------------------------------------------
            # Condition appartenance à l'area
            #-------------------------------------------------------------------
      np.copyto(condarea[y][x], aowntab[y][x])
      #print("\tappartenance area : %d"%(np.count_nonzero(condarea[y][x])))
            #-------------------------------------------------------------------
            # Ajout condition sur l'amplidute maxi des mouvements
            # (rejète les mouvements parasites de grande longeur et 
            # non significatifs retournés par "CalcOptical")
            #-------------------------------------------------------------------
      condlen = lmovtab <= MAX_MOVLEN
      condarea[y][x] &= condlen
      #print("\tlongueur maxi : %d"%(np.count_nonzero(condarea[y][x])))
      
            #-------------------------------------------------------------------
            # Ajout condition sur pertinence angles des mouvements
            # (garde les mouvements dont les angles se rapprochent en nombre
            # de la majorité)
            #-------------------------------------------------------------------
              # Tous les mouvements sélectionnés précédemment
      agareatab = np.extract(condarea[y][x], amovtab)
              # Histogramme comptage par angle
      agareahsttab = np.bincount(agareatab, minlength=360)
              # Nb de mouvements pour l'angle le + représentatif
      agareacntmax = np.amax(agareahsttab)
              # Seuil minimal comptage pour la prise en compte des mouvements
      movcntthres = int(agareacntmax * MINTHRES_ANGCNT)
              # Condition de filtrage des mouvements supérieurs au seuil
      condcnt = agareahsttab > movcntthres # 360
              # Application de la condition sur la table globale des angles
      condang = condcnt[amovtab]
              # Application condition sur mouvements sélectionnés précédemment
      condarea[y][x] &= condang
      #print("\tangle pertinent : %d"%(np.count_nonzero(condarea[y][x])))
      
            #-------------------------------------------------------------------
            # Comptage des mouvements retenus pour l'area
            #-------------------------------------------------------------------
      movareacnt[y][x] = np.count_nonzero(condarea[y][x])
      
            #-------------------------------------------------------------------
            # Pas assez de mouvements : ignore l'area
            #-------------------------------------------------------------------
      if movareacnt[y][x] < MINMOV_AREACNT:
        #print("\tREJECT: count mini %d < %d"%(movareacnt[y][x], MINMOV_AREACNT))
        movareacnt[y][x] = 0
        x += 1
        continue

            #-------------------------------------------------------------------
            # Amplitude mouvements trop faible : ignore l'area
            #-------------------------------------------------------------------
      lntab = np.extract(condarea[y][x], lmovtab)
      avglen = np.average(lntab)
      if avglen < MINAVG_MOVLEN:
        #print("\tREJECT: avglen mini %.2f < %.2f"%(avglen, MINAVG_MOVLEN))
        movareacnt[y][x] = 0
        x += 1
        continue

      x += 1
    y += 1

        #-----------------------------------------------------------------------
        # Dessine l'image des traces mouvements
        #-----------------------------------------------------------------------
  if False:
    im_move = np.zeros((IM_YSIZ, IM_XSIZ), np.uint8)
  
    y = 0
    while y < ARMT_YSIZ:
      x = 0
      while x < ARMT_XSIZ:
              #-------------------------------------------------------------------
              # Extractions composantes retenues pour l'area
              #-------------------------------------------------------------------
        x1tab = np.extract(condarea[y][x], pttab[PTTAB_X1])
        y1tab = np.extract(condarea[y][x], pttab[PTTAB_Y1])
        x2tab = np.extract(condarea[y][x], pttab[PTTAB_X2])
        y2tab = np.extract(condarea[y][x], pttab[PTTAB_Y2])
        lntab = np.extract(condarea[y][x], lmovtab)
        agtab = np.extract(condarea[y][x], amovtab)
  
              #-------------------------------------------------------------------
              # Dessine les contours de l'area
              #-------------------------------------------------------------------
        x1 = ARMT[y][x][ARMT_X1]; x2 = ARMT[y][x][ARMT_X2]
        y1 = ARMT[y][x][ARMT_Y1]; y2 = ARMT[y][x][ARMT_Y2]
        cv2.line(im_move, (x1, y1), (x2, y1), 255)
        cv2.line(im_move, (x1, y2), (x2, y2), 255)
        cv2.line(im_move, (x1, y1), (x1, y2), 255)
        cv2.line(im_move, (x2, y1), (x2, y2), 255)
        
              #-------------------------------------------------------------------
              # Aucun mouvement retenu : ne dessine rien
              #-------------------------------------------------------------------
        if movareacnt[y][x] == 0:
          x += 1
          continue
          
              #-------------------------------------------------------------------
              # Dessine les traits représentant les mouvements
              #-------------------------------------------------------------------
        imov = 0
        while imov < movareacnt[y][x]:
          xx1 = x1tab[imov]; yy1 = y1tab[imov]
          xx2 = x2tab[imov]; yy2 = y2tab[imov]
          imov += 1
          cv2.line(im_move, (xx1, yy1), (xx2, yy2), 255)
  
              #-------------------------------------------------------------------
              # Affiche l'angle et l'amplitude moyenne des mouvements
              #-------------------------------------------------------------------
        im_move[y1:y1+20][x1+2:x2-2] = 0
        cv2.putText(im_move,
            "len=%.2f ang=%.2f"%(np.average(lntab), np.average(agtab)),
            (x1 + 2, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        x += 1
      y += 1
  
    ShowImg("Moves", im_move, em.MODE_TEST)

        #-----------------------------------------------------------------------
        # Dessine l'image tableau de bord
        #-----------------------------------------------------------------------
  if False:
    im_dbrd = np.zeros((IM_YSIZ, IM_XSIZ), np.uint8)
  
    y = 0
    while y < ARMT_YSIZ:
      x = 0
      while x < ARMT_XSIZ:
              #-------------------------------------------------------------------
              # Extractions composantes retenues pour l'area
              #-------------------------------------------------------------------
        x1tab = np.extract(condarea[y][x], pttab[PTTAB_X1])
        y1tab = np.extract(condarea[y][x], pttab[PTTAB_Y1])
        x2tab = np.extract(condarea[y][x], pttab[PTTAB_X2])
        y2tab = np.extract(condarea[y][x], pttab[PTTAB_Y2])
        lntab = np.extract(condarea[y][x], lmovtab)
        agtab = np.extract(condarea[y][x], amovtab)
  
              #-------------------------------------------------------------------
              # Dessine les contours de l'area
              #-------------------------------------------------------------------
        x1 = ARMT[y][x][ARMT_X1]; x2 = ARMT[y][x][ARMT_X2]
        y1 = ARMT[y][x][ARMT_Y1]; y2 = ARMT[y][x][ARMT_Y2]
        cv2.line(im_dbrd, (x1, y1), (x2, y1), 255)
        cv2.line(im_dbrd, (x1, y2), (x2, y2), 255)
        cv2.line(im_dbrd, (x1, y1), (x1, y2), 255)
        cv2.line(im_dbrd, (x2, y1), (x2, y2), 255)
  
              #-------------------------------------------------------------------
              # Aucun mouvement retenu : ne dessine rien
              #-------------------------------------------------------------------
        if movareacnt[y][x] == 0:
          x += 1
          continue
          
              #-------------------------------------------------------------------
              # Représentation de l'angle moyen
              #-------------------------------------------------------------------
        crclecart = 10
        crclrad = (AR_YSIZ // 2)
        crclxcent = x1 + crclrad
        crclycent = y1 + crclrad
        crclrad -= crclecart
        cv2.circle(im_dbrd, (crclxcent, crclycent), crclrad, 255, 1)
        
        avgangdeg = avgangrad = np.average(agtab)
        # if avgangrad > 180: avgangrad -= 360
        avgangrad *= (np.pi / 180)
        linx = int(crclxcent + (np.cos(avgangrad) * crclrad))
        liny = int(crclycent - (np.sin(avgangrad) * crclrad))
        cv2.line(im_dbrd, (crclxcent, crclycent), (linx, liny), 255, 5)
        
              #-------------------------------------------------------------------
              # Représentation de l'amplitude moyenne
              #-------------------------------------------------------------------
        avglen = np.average(lntab)
        spleft = AR_XSIZ - ((crclrad * 2) + crclecart)
        linx = int(x1 + (crclrad * 2) + (spleft / 2))
        liny = int(crclycent + crclrad)
        liny2 = int(liny - ((crclrad * 2) / MAX_MOVLEN) * avglen)
        cv2.line(im_dbrd, (linx, liny), (linx, liny2), 255, 5)
        
        x += 1
      y += 1
  
    #ShowImg("Dash Board", im_dbrd, em.MODE_TEST)

        #-----------------------------------------------------------------------
        # Alimentation de l'historique des mouvements
        #-----------------------------------------------------------------------
  mov_histo = np.zeros((ARMT_YSIZ, ARMT_XSIZ, 1, 2), np.uint16)
  
  y = 0
  while y < ARMT_YSIZ:
    x = 0
    while x < ARMT_XSIZ:
      if movareacnt[y][x] == 0:
        x += 1
        continue

      lntab = np.extract(condarea[y][x], lmovtab)
      agtab = np.extract(condarea[y][x], amovtab)

      avglen = np.average(lntab)
      avgang = np.average(agtab)
      mov_histo[y][x][0] = [avglen, avgang]

      x += 1
    y += 1

  MOV_HISTO = np.append(MOV_HISTO, mov_histo, 2)

        #-----------------------------------------------------------------------
        # Calcul angle et positions de la camera
        #-----------------------------------------------------------------------
            #-------------------------------------------------------------------
            # Fonction retournant le nb pts, amplitude et angle moyen d'une area
            #-------------------------------------------------------------------
  def AvgLenAngPt(y, x):
    ptcnt = avglen = avgang = 0
    if movareacnt[y][x] > 0:
      ptcnt = movareacnt[y][x]
      lntab = np.extract(condarea[y][x], lmovtab)
      agtab = np.extract(condarea[y][x], amovtab)
      avglen = np.average(lntab)
      avgang = np.average(agtab)

    return ptcnt, avglen, avgang
    
            #-------------------------------------------------------------------
            # Nb pts, amplitude et angle moyen des 4 areas coins de la frame
            #-------------------------------------------------------------------
  CNT = 0; LEN = 1; ANG = 2; FPT = 3
  ptdirtab = np.zeros((NB_PTS, FPT + NB_DIR), np.float)
  ptdirtab[0][CNT], ptdirtab[0][LEN], ptdirtab[0][ANG] = \
                                      AvgLenAngPt(0, ARMT_XSIZ - 1)
  ptdirtab[1][CNT], ptdirtab[1][LEN], ptdirtab[1][ANG] = \
                                      AvgLenAngPt(0, 0)
  ptdirtab[2][CNT], ptdirtab[2][LEN], ptdirtab[2][ANG] = \
                                      AvgLenAngPt(ARMT_YSIZ - 1, 0)
  ptdirtab[3][CNT], ptdirtab[3][LEN], ptdirtab[3][ANG] = \
                                      AvgLenAngPt(ARMT_YSIZ - 1, ARMT_XSIZ - 1)

            #-------------------------------------------------------------------
            # Compte le nb points dont l'ampl. et l'angle ont pu être calculés
            #-------------------------------------------------------------------
  ptokcnt = 0; idxpt = 0
  while idxpt < NB_PTS:
    if ptdirtab[idxpt][CNT] > 0: ptokcnt += 1
    idxpt += 1
  PrintDev("ptokcnt", ptokcnt)

            #-------------------------------------------------------------------
            # Pas assez de points valides : pas de déplacement camera possible
            #-------------------------------------------------------------------
  if ptokcnt < MINPTS_VAL:
    return
            #-------------------------------------------------------------------
            # Affection amplitude direction pour chaque point
            #-------------------------------------------------------------------
  idxpt = 0
  while idxpt < NB_PTS:
    if ptdirtab[idxpt][CNT] == 0:
      idxpt += 1
      continue
        # détermine l'index de la direction seuil bas à partir de l'angle
    diridx = int(ptdirtab[idxpt][ANG] / ANG_PTSSEP)
        # reliquat angle restant à partir de la direction seuil bas
    anglft = ptdirtab[idxpt][ANG] - (diridx * ANG_PTSSEP)
        # coefficient de proximité de la direction seuil bas
        # avec l'angle effectif (0 => 1)
    cfproxlow = (1 - (anglft / ANG_PTSSEP))
        # coefficient de proximité de la direction suivante
        # avec l'angle effectif (0 => 1)
    cfproxnxt = (1 - cfproxlow)
        # distribution de l'amplitude du mouvement pour la direction seuil bas
        # et la direction suivante
    lenmvtlow = ptdirtab[idxpt][LEN] * cfproxlow
    lenmvtnxt = ptdirtab[idxpt][LEN] * cfproxnxt
        # positionne l'amplitude pour la direction seuil bas et suivante
    ptdirtab[idxpt][FPT + diridx] = lenmvtlow
          # cas direction seuil bas < 360 et suivante > 0
    if diridx == NB_DIR - 1:
      ptdirtab[idxpt][FPT + 0] = lenmvtnxt
    else:
      ptdirtab[idxpt][FPT + (diridx + 1)] = lenmvtnxt
    idxpt += 1

  #print("ptdirtab", ptdirtab)
            #-------------------------------------------------------------------
            # Affection amplitude pour les points de chaque type de mouvement
            #-------------------------------------------------------------------
  mvtptstab = np.zeros((NB_MVT, NB_PTS), np.float)
  idxmvt = 0
  while idxmvt < NB_MVT:
    idxpt = 0
    while idxpt < NB_PTS:
      if ptdirtab[idxpt][CNT] == 0:
        idxpt += 1
        continue
      mvtptstab[idxmvt][idxpt] = \
                              ptdirtab[idxpt][FPT + MVT_DEF_TAB[idxmvt][idxpt]]
      idxpt += 1
        # mouvement avec au moins N pts valorisés
    #if np.count_nonzero(mvtptstab[idxmvt]) >= MINPTS_VAL:
    #  print("=== %s ==="%(MVT_NAM_TAB[idxmvt]))
    #  print("%02f %02f"%(mvtptstab[idxmvt][1], mvtptstab[idxmvt][0]))
    #  print("%02f %02f"%(mvtptstab[idxmvt][2], mvtptstab[idxmvt][3]))

    idxmvt += 1

  #print("mvtptstab", mvtptstab)
            #-------------------------------------------------------------------
            # Calcul déplacement camera
            #-------------------------------------------------------------------
  mvtavgtab = np.zeros(NB_MVT, np.float)
              #-----------------------------------------------------------------
              # Calcul amplitude moyenne globale des mouvements
              #-----------------------------------------------------------------
  for idxmvt in [MVT_TL, MVT_TR, MVT_TF, MVT_TB, MVT_RL, MVT_RR]:
        # Pas assez de points valides pour le mouvement : ignoré
    if np.count_nonzero(mvtptstab[idxmvt]) < MINPTS_VAL:
      continue
        # Amplitude moyenne globale du mouvement
    mvtavgtab[idxmvt] = np.sum(mvtptstab[idxmvt]) / ptokcnt

              #-----------------------------------------------------------------
              # Suppression des mouvements avec des directions antagonistes
              #-----------------------------------------------------------------
  if mvtavgtab[MVT_TL] > 0 and mvtavgtab[MVT_TR] > 0:
    mvtavgtab[MVT_TL] = mvtavgtab[MVT_TR] = 0
  if mvtavgtab[MVT_TF] > 0 and mvtavgtab[MVT_TB] > 0:
    mvtavgtab[MVT_TF] = mvtavgtab[MVT_TB] = 0
  if mvtavgtab[MVT_RL] > 0 and mvtavgtab[MVT_RR] > 0:
    mvtavgtab[MVT_RL] = mvtavgtab[MVT_RR] = 0

              #-----------------------------------------------------------------
              # Priorisation des rotations si rotation/translation simultanées
              #-----------------------------------------------------------------
#  if mvtavgtab[MVT_RL] > 0 and mvtavgtab[MVT_TL] > 0:
#    mvtavgtab[MVT_TL] = 0
#  if mvtavgtab[MVT_RR] > 0 and mvtavgtab[MVT_TR] > 0:
#    mvtavgtab[MVT_TR] = 0

  mvtavgtab[MVT_RL] += mvtavgtab[MVT_TL]
  mvtavgtab[MVT_RR] += mvtavgtab[MVT_TR]
  mvtavgtab[MVT_TL] = 0
  mvtavgtab[MVT_TR] = 0
  #mvtavgtab[MVT_TB] = 0
  
              #-----------------------------------------------------------------
              # Calcul déplacements
              #-----------------------------------------------------------------
  libmov = ""  
  if mvtavgtab[MVT_RL] > 0:
    A_CAM += mvtavgtab[MVT_RL] * RL_FACTOR
    libmov = libmov + "%s [%.2f]"% \
                            (MVT_NAM_TAB[MVT_RL], mvtavgtab[MVT_RL] * RL_FACTOR)
  if mvtavgtab[MVT_RR] > 0:
    A_CAM -= mvtavgtab[MVT_RR] * RR_FACTOR
    libmov = libmov + "%s [%.2f]"% \
                            (MVT_NAM_TAB[MVT_RR], mvtavgtab[MVT_RR] * RR_FACTOR)

  if A_CAM > 360: A_CAM = A_CAM - 360
  if A_CAM < 0:   A_CAM = 360 - A_CAM
  libmov = "A [%.2f] "%(A_CAM) + libmov
  
  for mvtrs, agtrs in [(MVT_TF, 0), (MVT_TB, 180), (MVT_TR, -90), (MVT_TL, 90)]:
    if mvtavgtab[mvtrs] == 0 or np.abs(mvtavgtab[mvtrs]) < 0.1:
      continue
    else:
      print(mvtavgtab[mvtrs])
    acam = A_CAM + agtrs
    if acam > 360:  acam = acam - 360
    if acam < 0:    acam = 360 - acam
    acam *= (np.pi / 180)
    X_CAM = int(np.around((X_CAM + (np.cos(acam) * mvtavgtab[mvtrs]) * T_FACTOR)))
    Y_CAM = int(np.around((Y_CAM - (np.sin(acam) * mvtavgtab[mvtrs]) * T_FACTOR)))
    # print(mvtavgtab)
    # print(MVT_NAM_TAB)
    libmov = libmov + " %s [%.2f]"% \
                            (MVT_NAM_TAB[mvtrs], mvtavgtab[mvtrs] * T_FACTOR)
    print(libmov)
            #-------------------------------------------------------------------
            # Affichage position camera
            #-------------------------------------------------------------------
  DrawCam(libmov)

  sp.SP_DumpStep()

  return
  
  
