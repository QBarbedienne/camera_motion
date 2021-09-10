import sys
import cv2
import numpy as np
import imutils
import time 

class GetMotion():
    def __init__(self):

        ## Variables 
        self.idxpt = 0
        self.FFD = None

        self.FEATURE_DETECTOR = None
        self.IM_YSIZE = None
        self.IM_XSIZE = None
        self.ARMT = None
        self.AR_YSIZ = None
        self.AR_XSIZ = None
        self.IM_FFD_PREV = None
        self.PT_PREV = None
        self.ARMT_YSIZ = 3
        self.ARMT_XSIZ = 3
        self.MOV_HISTO = np.zeros((self.ARMT_YSIZ, self.ARMT_XSIZ, 0, 2), np.uint16)

        self.IM_Cam = None
        self.Y_Cam = None
        self.X_Cam = None
        self.A_Cam = None
        self.H_Cam = np.zeros((0, 2), np.uint16)
        self.itteration = 0

        ## constantes
        
        self.FFD_CREAT_THRES = 50
        self.FFD_CREAT_NONMAXSUP = True
        # self.FFD_CREAT_TYPE = cv2.FAST_FEATURE_DETECTOR_TYPE_5_8
        # self.FFD_CREAT_TYPE = cv2.FAST_FEATURE_DETECTOR_TYPE_7_12
        self.FFD_CREAT_TYPE = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16 # default value
        self.FDD_DETECT_MINFREQ = 10

        self.MAX_MOVLEN = 50
        self.MINAVG_MOVLEN = 2
        self.MINTHRES_ANGCNT = 0.5
        self.MINMOV_AREACNT = 5
        self.MINPTS_VAL = 2

        self.Y_POS_ORIG = 0.5
        self.X_POS_ORIG = 0.5
        self.A_POS_ORIG = 1

        self.RL_FACTOR = 0.01
        self.RR_FACTOR = 0.01 
        self.T_FACTOR = 0.3

        self.ARMT_YSIZ = 3
        self.ARMT_XSIZ = 3

        self.ARMT_Y1 = 0
        self.ARMT_Y2 = 1
        self.ARMT_X1 = 2
        self.ARMT_X2 = 3
        self.PTTAB_X1 = 0
        self.PTTAB_Y1 = 1
        self.PTTAB_X2 = 2
        self.PTTAB_Y2 = 3

        self.NB_PTS = 4
        self.NB_DIR = 8
        self.ANG_PTSSEP = 360 / self.NB_DIR

        self.MVT_TU = 0
        self.MVT_TD = 1
        self.MVT_TL = 2
        self.MVT_TR = 3
        self.MVT_TF = 4
        self.MVT_TB = 5
        self.MVT_RF = 6
        self.MVT_RB = 7
        self.MVT_RL = 8
        self.MVT_RR = 9
        self.MVT_RH = 10
        self.MVT_RA = 11
        self.NB_MVT = 12

        self.MVT_DEF_TAB = np.zeros((self.NB_MVT, self.NB_PTS), np.uint8)
        self.MVT_NAM_TAB = [None, None, None, None, None, None, None, None, None, None, None, None, ]

        self.MVT_NAM_TAB[self.MVT_TU] = 'TRS UP'
        self.MVT_DEF_TAB[self.MVT_TU] = [ 2,  2,  2,  2]
        self.MVT_NAM_TAB[self.MVT_TD] = 'TRS DOWN'
        self.MVT_DEF_TAB[self.MVT_TD] = [ 6,  6,  6,  6]
        self.MVT_NAM_TAB[self.MVT_TL] = 'TRS LEFT'
        self.MVT_DEF_TAB[self.MVT_TL] = [ 4,  4,  4,  4]
        self.MVT_NAM_TAB[self.MVT_TR] = 'TRS RIGHT'
        self.MVT_DEF_TAB[self.MVT_TR] = [ 0,  0,  0,  0]
        self.MVT_NAM_TAB[self.MVT_TF] = 'TRS FORWARD'
        self.MVT_DEF_TAB[self.MVT_TF] = [ 5,  7,  1,  3]
        self.MVT_NAM_TAB[self.MVT_TB] = 'TRS BACK'
        self.MVT_DEF_TAB[self.MVT_TB] = [ 1,  3,  5,  7]
        self.MVT_NAM_TAB[self.MVT_RF] = 'ROT FORWARD'
        self.MVT_DEF_TAB[self.MVT_RF] = [ 5,  7,  5,  7]
        self.MVT_NAM_TAB[self.MVT_RB] = 'ROT BACK'
        self.MVT_DEF_TAB[self.MVT_RB] = [ 1,  3,  1,  3]
        self.MVT_NAM_TAB[self.MVT_RL] = 'ROT LEFT'
        self.MVT_DEF_TAB[self.MVT_RL] = [ 5,  3,  5,  3]
        self.MVT_NAM_TAB[self.MVT_RR] = 'ROT RIGHT'
        self.MVT_DEF_TAB[self.MVT_RR] = [ 1,  7,  1,  7]
        self.MVT_NAM_TAB[self.MVT_RH] = 'ROT HORAIRE'
        self.MVT_DEF_TAB[self.MVT_RH] = [ 7,  1,  3,  5]
        self.MVT_NAM_TAB[self.MVT_RA] = 'ROT ANTIHORAIRE'
        self.MVT_DEF_TAB[self.MVT_RA] = [ 3,  5,  7,  1]

        self.libmov = "" 
    def init_area_mat(self, imysiz, imxsiz):
    
        self.ARMT = np.zeros((self.ARMT_YSIZ, self.ARMT_XSIZ, 4), np.uint16)
        self.AR_YSIZ = imysiz // self.ARMT_YSIZ
        self.AR_XSIZ = imxsiz // self.ARMT_XSIZ
        y = 0
        while y < self.ARMT_YSIZ:
            x = 0
            while x < self.ARMT_XSIZ:
                self.ARMT[y][x][self.ARMT_Y1] = y * self.AR_YSIZ
                self.ARMT[y][x][self.ARMT_Y2] = (y + 1) * self.AR_YSIZ
                self.ARMT[y][x][self.ARMT_X1] = x * self.AR_XSIZ
                self.ARMT[y][x][self.ARMT_X2] = (x + 1) * self.AR_XSIZ
                x += 1
            y += 1

    def do_init(self, im_bgr):
        
        # Création objet FastFeatureDetector : Instancié ici pour prendre en compte la modification de paramètres par le scrip appelant
        self.FFD = cv2.FastFeatureDetector_create(self.FFD_CREAT_THRES, self.FFD_CREAT_NONMAXSUP, self.FFD_CREAT_TYPE)

        # Taille de l'image et matrice des zones
        IM_YSIZ, IM_XSIZ = im_bgr.shape[:2]
        self.init_area_mat(IM_YSIZ, IM_XSIZ)

        # position initiale de la camera
        self.IM_CAM = np.zeros((IM_YSIZ, IM_XSIZ), np.uint8)
        self.Y_Cam = int((IM_YSIZ - 1) * self.Y_POS_ORIG)
        self.X_Cam = int((IM_XSIZ - 1) * self.X_POS_ORIG)
        self.A_Cam = int(90 * self.A_POS_ORIG)
        self.itteration += 1

    def AvgLenAngPt(self, y, x, movareacnt, condarea, lmovtab, amovtab):
        # Fonction retournant le nb pts, amplitude et angle moyen d'une area
        ptcnt = avglen = avgang = 0
        if movareacnt[y][x] > 0:
            ptcnt = movareacnt[y][x]
            lntab = np.extract(condarea[y][x], lmovtab)
            agtab = np.extract(condarea[y][x], amovtab)
            avglen = np.average(lntab)
            avgang = np.average(agtab)

        return ptcnt, avglen, avgang

    def does_it_move(self, im_bgr):

        if self.itteration ==0:
            self.do_init(im_bgr)

        im_ffd_curr = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

        if self.PT_PREV is None or self.itteration % self.FDD_DETECT_MINFREQ == 1:
            if self.itteration == 1: im = im_ffd_curr
            else: im = self.IM_FFD_PREV
                # mask:	Mask specifying where to look for keypoints (optional).
            kpprev = self.FFD.detect(im, mask=None)
            self.PT_PREV = cv2.KeyPoint_convert(kpprev)
        if self.itteration == 1:
            self.IM_FFD_PREV = im_ffd_curr
            return

        kpprevcnt = len(self.PT_PREV)
        if kpprevcnt <= 0:
            self.IM_FFD_PREV = im_ffd_curr
            self.PT_PREV = None
            print('No previous keypoint found')
            return

        ptcurrtab, status, err = cv2.calcOpticalFlowPyrLK(self.IM_FFD_PREV, im_ffd_curr, self.PT_PREV, None)
        
        movokcnt = np.count_nonzero(status)
        if movokcnt <= 0:
            self.IM_FFD_PREV = im_ffd_curr
            self.PT_PREV = None
            print("No movement found")
            return

        status = np.broadcast_to(status, (status.shape[0], 2))
        ptprevtab = self.PT_PREV[status == [1, 1]].reshape(-1, 2)
        ptcurrtab = ptcurrtab[status == [1, 1]].reshape(-1, 2)

        pttab = np.int16(np.hstack((ptprevtab, ptcurrtab)).ravel(order = 'F').reshape(4, -1))

        self.IM_FFD_PREV = im_ffd_curr
        self.PT_PREV = ptcurrtab


        # Calcul mouvements X et Y
        xmovtab = pttab[self.PTTAB_X1] - pttab[self.PTTAB_X2]
        # Inverse l'axe des Y de bas en haut
        ymovtab = pttab[self.PTTAB_Y2] - pttab[self.PTTAB_Y1]

        # Calcul amplitudes mouvements
        lmovtab = np.around(np.hypot(xmovtab, ymovtab), 2)
        # Calcul angles mouvements en ° (tronqués à l'unité)
        amovtab = np.int64(np.arctan2(ymovtab, xmovtab) * 180 / np.pi)
        amovtab = np.where(amovtab < 0, 360 + amovtab, amovtab)

        # Calcul areas d'appartenance des mouvements
        aowntab = np.zeros((self.ARMT_YSIZ, self.ARMT_XSIZ, movokcnt), np.bool_)
        y = 0
        while y < self.ARMT_YSIZ:
            x = 0
            while x < self.ARMT_XSIZ:
                cond  = ((pttab[self.PTTAB_X1] >= self.ARMT[y][x][self.ARMT_X1]) & \
                        (pttab[self.PTTAB_X1] <  self.ARMT[y][x][self.ARMT_X2]) & \
                        (pttab[self.PTTAB_Y1] >= self.ARMT[y][x][self.ARMT_Y1]) & \
                        (pttab[self.PTTAB_Y1] <  self.ARMT[y][x][self.ARMT_Y2])) \
                        | \
                        ((pttab[self.PTTAB_X2] >= self.ARMT[y][x][self.ARMT_X1]) & \
                        (pttab[self.PTTAB_X2] <  self.ARMT[y][x][self.ARMT_X2]) & \
                        (pttab[self.PTTAB_Y2] >= self.ARMT[y][x][self.ARMT_Y1]) & \
                        (pttab[self.PTTAB_Y2] <  self.ARMT[y][x][self.ARMT_Y2]))
                np.copyto(aowntab[y][x], cond)
                x += 1
            y += 1
        
        # Calcul conditions d'extraction des composantes pour chaque area

        condarea = np.zeros((self.ARMT_YSIZ, self.ARMT_XSIZ, movokcnt), np.bool_)
        movareacnt = np.zeros((self.ARMT_YSIZ, self.ARMT_XSIZ), np.int64)

        y = 0
        while y < self.ARMT_YSIZ:
            x = 0
            while x < self.ARMT_XSIZ:
                
                # Condition appartenance à l'area
                np.copyto(condarea[y][x], aowntab[y][x])
                
                # Ajout condition sur l'amplidute maxi des mouvements (rejète les mouvements parasites de grande longeur et non significatifs retournés par "CalcOptical")
                condlen = lmovtab <= self.MAX_MOVLEN
                condarea[y][x] &= condlen
                #print("\tlongueur maxi : %d"%(np.count_nonzero(condarea[y][x])))
                
                # Ajout condition sur pertinence angles des mouvements (garde les mouvements dont les angles se rapprochent en nombre de la majorité)

                # Tous les mouvements sélectionnés précédemment
                agareatab = np.extract(condarea[y][x], amovtab)
                # Histogramme comptage par angle
                agareahsttab = np.bincount(agareatab, minlength=360)
                # Nb de mouvements pour l'angle le + représentatif
                agareacntmax = np.amax(agareahsttab)
                # Seuil minimal comptage pour la prise en compte des mouvements
                movcntthres = int(agareacntmax * self.MINTHRES_ANGCNT)
                # Condition de filtrage des mouvements supérieurs au seuil
                condcnt = agareahsttab > movcntthres # 360
                # Application de la condition sur la table globale des angles
                condang = condcnt[amovtab]
                # Application condition sur mouvements sélectionnés précédemment
                condarea[y][x] &= condang
                
                #print("\tangle pertinent : %d"%(np.count_nonzero(condarea[y][x])))
                
                # Comptage des mouvements retenus pour l'area
                movareacnt[y][x] = np.count_nonzero(condarea[y][x])
                
                # Pas assez de mouvements : ignore l'area
                if movareacnt[y][x] < self.MINMOV_AREACNT:
                    movareacnt[y][x] = 0
                    x += 1
                    continue

                # Amplitude mouvements trop faible : ignore l'area
                lntab = np.extract(condarea[y][x], lmovtab)
                avglen = np.average(lntab)
                if avglen < self.MINAVG_MOVLEN:
                    movareacnt[y][x] = 0
                    x += 1
                    continue

                x += 1
            y += 1


        # Alimentation de l'historique des mouvements
        mov_histo = np.zeros((self.ARMT_YSIZ, self.ARMT_XSIZ, 1, 2), np.uint16)
        
        y = 0
        while y < self.ARMT_YSIZ:
            x = 0
            while x < self.ARMT_XSIZ:
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

        MOV_HISTO = np.append(self.MOV_HISTO, mov_histo, 2)

        # Nb pts, amplitude et angle moyen des 4 areas coins de la frame
        self.CNT = 0
        self.LEN = 1
        self.ANG = 2
        self.FPT = 3
        self.ptdirtab = np.zeros((self.NB_PTS, self.FPT + self.NB_DIR), np.float64)
        self.ptdirtab[0][self.CNT], self.ptdirtab[0][self.LEN], self.ptdirtab[0][self.ANG] = self.AvgLenAngPt(0, self.ARMT_XSIZ - 1, movareacnt, condarea, lmovtab, amovtab)
        self.ptdirtab[1][self.CNT], self.ptdirtab[1][self.LEN], self.ptdirtab[1][self.ANG] = self.AvgLenAngPt(0, 0, movareacnt, condarea, lmovtab, amovtab)
        self.ptdirtab[2][self.CNT], self.ptdirtab[2][self.LEN], self.ptdirtab[2][self.ANG] = self.AvgLenAngPt(self.ARMT_YSIZ - 1, 0, movareacnt, condarea, lmovtab, amovtab)
        self.ptdirtab[3][self.CNT], self.ptdirtab[3][self.LEN], self.ptdirtab[3][self.ANG] = self.AvgLenAngPt(self.ARMT_YSIZ - 1, self.ARMT_XSIZ - 1, movareacnt, condarea, lmovtab, amovtab)

        # Compte le nb points dont l'ampl. et l'angle ont pu être calculés
        self.ptokcnt = 0
        self.idxpt = 0
        while self.idxpt < self.NB_PTS:
            if self.ptdirtab[self.idxpt][self.CNT] > 0: self.ptokcnt += 1
            self.idxpt += 1

        # Pas assez de points valides : pas de mouvement caméra possible
        if self.ptokcnt < self.MINPTS_VAL:
            return False
        else:
            self.compute_movement()
            return True

    def compute_movement(self):
        # Affection amplitude direction pour chaque point
        self.idxpt = 0
        while self.idxpt < self.NB_PTS:
            if self.ptdirtab[self.idxpt][self.CNT] == 0:
                self.idxpt += 1
                continue
            # détermine l'index de la direction seuil bas à partir de l'angle
            diridx = int(self.ptdirtab[self.idxpt][self.ANG] / self.ANG_PTSSEP)
                # reliquat angle restant à partir de la direction seuil bas
            anglft = self.ptdirtab[self.idxpt][self.ANG] - (diridx * self.ANG_PTSSEP)
                # coefficient de proximité de la direction seuil bas
                # avec l'angle effectif (0 => 1)
            cfproxlow = (1 - (anglft / self.ANG_PTSSEP))
                # coefficient de proximité de la direction suivante
                # avec l'angle effectif (0 => 1)
            cfproxnxt = (1 - cfproxlow)
                # distribution de l'amplitude du mouvement pour la direction seuil bas
                # et la direction suivante
            lenmvtlow = self.ptdirtab[self.idxpt][self.LEN] * cfproxlow
            lenmvtnxt = self.ptdirtab[self.idxpt][self.LEN] * cfproxnxt
                # positionne l'amplitude pour la direction seuil bas et suivante
            self.ptdirtab[self.idxpt][self.FPT + diridx] = lenmvtlow
                # cas direction seuil bas < 360 et suivante > 0
            if diridx == self.NB_DIR - 1:
                self.ptdirtab[self.idxpt][self.FPT + 0] = lenmvtnxt
            else:
                self.ptdirtab[self.idxpt][self.FPT + (diridx + 1)] = lenmvtnxt
            self.idxpt += 1


        # Affection amplitude pour les points de chaque type de mouvement
        mvtptstab = np.zeros((self.NB_MVT, self.NB_PTS), np.float64)
        idxmvt = 0
        while idxmvt < self.NB_MVT:
            idxpt = 0
            while idxpt < self.NB_PTS:
                if self.ptdirtab[idxpt][self.CNT] == 0:
                    idxpt += 1
                    continue
                mvtptstab[idxmvt][idxpt] = self.ptdirtab[idxpt][self.FPT + self.MVT_DEF_TAB[idxmvt][idxpt]]
                idxpt += 1
                # mouvement avec au moins N pts valorisés
            #if np.count_nonzero(mvtptstab[idxmvt]) >= MINPTS_VAL:
            #  print("=== %s ==="%(MVT_NAM_TAB[idxmvt]))
            #  print("%02f %02f"%(mvtptstab[idxmvt][1], mvtptstab[idxmvt][0]))
            #  print("%02f %02f"%(mvtptstab[idxmvt][2], mvtptstab[idxmvt][3]))

            idxmvt += 1

        # Calcul déplacement camera
        mvtavgtab = np.zeros(self.NB_MVT, np.float64)
        # Calcul amplitude moyenne globale des mouvements
        for idxmvt in [self.MVT_TL, self.MVT_TR, self.MVT_TF, self.MVT_TB, self.MVT_RL, self.MVT_RR]:
            # Pas assez de points valides pour le mouvement : ignoré
            # if np.count_nonzero(mvtptstab[idxmvt]) < self.MINPTS_VAL:
            #     continue
            # Amplitude moyenne globale du mouvement
            mvtavgtab[idxmvt] = np.sum(mvtptstab[idxmvt]) / self.ptokcnt

        # Suppression des mouvements avec des directions antagonistes
        if mvtavgtab[self.MVT_TL] > 0 and mvtavgtab[self.MVT_TR] > 0:
            mvtavgtab[self.MVT_TL] = mvtavgtab[self.MVT_TR] = 0
        if mvtavgtab[self.MVT_TF] > 0 and mvtavgtab[self.MVT_TB] > 0:
            mvtavgtab[self.MVT_TF] = mvtavgtab[self.MVT_TB] = 0
        if mvtavgtab[self.MVT_RL] > 0 and mvtavgtab[self.MVT_RR] > 0:
            mvtavgtab[self.MVT_RL] = mvtavgtab[self.MVT_RR] = 0

        mvtavgtab[self.MVT_RL] += mvtavgtab[self.MVT_TL]
        mvtavgtab[self.MVT_RR] += mvtavgtab[self.MVT_TR]
        mvtavgtab[self.MVT_TL] = 0
        mvtavgtab[self.MVT_TR] = 0

        self.libmov = ""  
        if mvtavgtab[self.MVT_RL] > 0:
            self.A_Cam += mvtavgtab[self.MVT_RL] * self.RL_FACTOR
            self.libmov = self.libmov + "%s [%.2f]"% \
                                    (self.MVT_NAM_TAB[self.MVT_RL], mvtavgtab[self.MVT_RL] * self.RL_FACTOR)
        if mvtavgtab[self.MVT_RR] > 0:
            self.A_Cam -= mvtavgtab[self.MVT_RR] * self.RR_FACTOR
            self.libmov = self.libmov + "%s [%.2f]"% \
                                    (self.MVT_NAM_TAB[self.MVT_RR], mvtavgtab[self.MVT_RR] * self.RR_FACTOR)

        if self.A_Cam > 360: self.A_Cam = self.A_Cam - 360
        if self.A_Cam < 0:   self.A_Cam = 360 - self.A_Cam
        self.libmov = "A [%.2f] "%(self.A_Cam) + self.libmov
        for mvtrs, agtrs in [(self.MVT_TF, 0), (self.MVT_TB, 180), (self.MVT_TR, -90), (self.MVT_TL, 90)]:
            if mvtavgtab[mvtrs] == 0:
                
                continue
            acam = self.A_Cam + agtrs
            if acam > 360:  acam = acam - 360
            if acam < 0:    acam = 360 - acam
            acam *= (np.pi / 180)
            
            self.X_Cam = int(np.around((self.X_Cam + (np.cos(acam) * mvtavgtab[mvtrs]) * self.T_FACTOR)))
            self.Y_Cam = int(np.around((self.Y_Cam - (np.sin(acam) * mvtavgtab[mvtrs]) * self.T_FACTOR)))
            self.libmov = self.libmov + " %s [%.2f]"% \
                                    (self.MVT_NAM_TAB[mvtrs], mvtavgtab[mvtrs] * self.T_FACTOR)
