# -*- coding: utf-8 -*-
# プログラム実行に必要なライブラリを読み込む
import cv2
import os
import numpy  as np
from decimal import Decimal, ROUND_DOWN
from skimage.feature import hog
np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# "特徴量選択プログラム(getFeatureVal_hatanaka.py)"の処理開始
def getFeatureVal_hatanaka(exe_cnt, img, eye_center, eye_size, train_dlib, pattern):
    # print(f'"特徴量選択プログラム(getFeatureVal_hatanaka.py)"の処理を開始します({exe_cnt}回目)。')
    # 【入力】
    # img : 入力画像（カラー）uint8型
    # eye_center : 目の中心座標 [右目の垂直座標、右目の水平座標、左目の垂直座標、左目の水平座標]
    # eye_size : 目のサイズ[縦、横]
    # pattern : 変換する特徴量パターン
    """
    1:HL, 2:I, 3:RGB, 4:HOG, 5:LBP, 6:H, 7:HL/I ,8:HL/RGB,
    9:HL/HOG, 10:I/RGB, 11:I/HOG, 12:RGB/HOG, 13: HL/I/RGB,
    14:HL/I/HOG, 15:HL/RGB/HOG, 16:I/RGB/HOG, 17:HL/I/RGB/HOG/LBP/H,
    18:Veg, 19:DLIB, 20:HL/I/DLIB, 21:HL/I/HOG/DLIB
    """
    # 【出力】
    # X : PLSに用いる入力値を返す（データ点数 N ×　データ１点当たりの次元 d )
    # X_breakdown : Xの特徴量サイズの内訳

    eyeR_center_y    = int(eye_center[0])
    eyeR_center_x    = int(eye_center[1])
    eyeL_center_y    = int(eye_center[2])
    eyeL_center_x    = int(eye_center[3])
  
    eyeH             = int(eye_size[0])
    eyeW             = int(eye_size[1])
  
    eyeH_half        = int(np.divide(eyeH, 2))
    eyeW_half        = int(np.divide(eyeW, 2))
    eyeH_half        = int(Decimal(eyeH_half).quantize(Decimal('0'), rounding=ROUND_DOWN))
    eyeW_half        = int(Decimal(eyeW_half).quantize(Decimal('0'), rounding=ROUND_DOWN))

    eyeR_leftup_y    = int(np.subtract(eyeR_center_y, eyeH_half)) # 右目の左上垂直座標
    eyeR_leftup_x    = int(np.subtract(eyeR_center_x, eyeW_half)) # 右目の左上水平座標
    eyeL_leftup_y    = int(np.subtract(eyeL_center_y, eyeH_half)) # 左目の左上垂直座標
    eyeL_leftup_x    = int(np.subtract(eyeL_center_x, eyeW_half)) # 左目の左上水平座標
    
    eyeR_img         = img[eyeR_leftup_y : eyeR_leftup_y + eyeH - 1, eyeR_leftup_x : eyeR_leftup_x + eyeW - 1, :]
    eyeL_img         = img[eyeL_leftup_y : eyeL_leftup_y + eyeH - 1, eyeL_leftup_x : eyeL_leftup_x + eyeW - 1, :]
    
    # RGB------------------------------------------------------------------
    if (pattern == 3) or (pattern == 8) or (pattern == 10) or (pattern == 12) or (pattern == 13) or (pattern == 15) or (pattern == 16) or (pattern == 17):
        eyeR_R       = float(eyeR_img[: , : , 0])
        eyeR_G       = float(eyeR_img[: , : , 1])
        eyeR_B       = float(eyeR_img[: , : , 2])
        eyeR_R_tf    = eyeR_R.reshape(-1, 1)
        eyeR_G_tf    = eyeR_G.reshape(-1, 1)
        eyeR_B_tf    = eyeR_B.reshape(-1, 1)

        eyeL_R       = float(eyeL_img[: , : , 0])
        eyeL_G       = float(eyeL_img[: , : , 1])
        eyeL_B       = float(eyeL_img[: , : , 2])
        eyeL_R_tf    = eyeL_R.reshape(-1, 1)
        eyeL_G_tf    = eyeL_G.reshape(-1, 1)
        eyeL_B_tf    = eyeL_B.reshape(-1, 1)
   
    # 白黒画像(輝度値I)------------------------------------------------------
    if (pattern != 3) or (pattern != 6) or (pattern != 19):
        img_gray     = cv2.cvtColor(img,      cv2.COLOR_RGB2GRAY)
        eyeR_gray    = cv2.cvtColor(eyeR_img, cv2.COLOR_RGB2GRAY)
        eyeL_gray    = cv2.cvtColor(eyeL_img, cv2.COLOR_RGB2GRAY)
        
        eyeR_gray_tf = eyeR_gray.reshape(-1, 1)
        eyeL_gray_tf = eyeL_gray.reshape(-1, 1)
   
    # HSV画像(色相H)---------------------------------------------------------------
    if (pattern == 6) or (pattern == 17):
        eyeR_hsv     = cv2.cvtColor(eyeR_img, cv2.COLOR_RGB2HSV)
        eyeR_h       = eyeR_hsv[:, :, 0]
        eyeL_hsv     = cv2.cvtColor(eyeL_img, cv2.COLOR_RGB2HSV)
        eyeL_h       = eyeL_hsv[:, :, 0]
        eyeR_h_tf    = eyeR_h.reshape(-1, 1)
        eyeL_h_tf    = eyeL_h.reshape(-1, 1)
   
    # Haar-LikeとLBP-------------------------------------------------------
    HL_R  = np.zeros((eyeH, eyeW))
    HL_L  = np.zeros((eyeH, eyeW))
    LBP_R = np.zeros((eyeH, eyeW))
    LBP_L = np.zeros((eyeH, eyeW))
    
    i = 0
    while i < eyeW:
        j = 0
        while j < eyeH:
            # 注目座標（目の左上から順に）
            R_y = int(eyeR_leftup_y) + j - 1
            R_x = int(eyeR_leftup_x) + i - 1
            
            L_y = int(eyeL_leftup_y) + j - 1
            L_x = int(eyeL_leftup_x) + i - 1
        
            # Haar-Like ---------------------------------------------------
            if (pattern == 1) or (pattern == 7) or (pattern == 8) or (pattern == 9) or (pattern == 13) or (pattern == 14) or (pattern == 15) or (pattern == 17) or (pattern == 20) or (pattern == 21):
                # 水平方向差分 : 左-右(A-B)

                # 右目
                tmpA        = img_gray[R_y - eyeH_half : R_y + eyeH_half - 1, R_x - eyeW_half : R_x - 1]
                tmpB        = img_gray[R_y - eyeH_half : R_y + eyeH_half - 1, R_x : R_x + eyeW_half - 1]
                tmpA_tf     = np.reshape(tmpA, (-1, 1))
                tmpB_tf     = np.reshape(tmpB, (-1, 1))
                tmpA_numel  = tmpA_tf.shape[0]
                tmpB_numel  = tmpB_tf.shape[0]
                tmpA_result = np.sum(tmpA_tf, axis=0) / tmpA_numel
                tmpB_result = np.sum(tmpB_tf, axis=0) / tmpB_numel
                
                HL_R[j, i]  = np.subtract(tmpA_result, tmpB_result)
                HL_R_tf     = HL_R.reshape(-1, 1)
            
                # 左目
                tmpA        = img_gray[L_y - eyeH_half : L_y + eyeH_half - 1, L_x - eyeW_half : L_x - 1]
                tmpB        = img_gray[L_y - eyeH_half : L_y + eyeH_half - 1, L_x : L_x + eyeW_half - 1]
                tmpA_tf     = np.reshape(tmpA, (-1, 1))
                tmpB_tf     = np.reshape(tmpB, (-1, 1))
                tmpA_numel  = tmpA_tf.shape[0]
                tmpB_numel  = tmpB_tf.shape[0]
                tmpA_result = np.sum(tmpA_tf, axis=0) / tmpA_numel
                tmpB_result = np.sum(tmpB_tf, axis=0) / tmpB_numel

                HL_L[j, i]  = np.subtract(tmpA_result, tmpB_result)
                HL_L_tf     = HL_L.reshape(-1, 1)

            # LBP----------------------------------------------------------
            if (pattern == 5) or (pattern == 17):
                LBP_prm = 1

                # 右目
                tmpC        = img_gray[R_y - LBP_prm : R_y + LBP_prm, R_x - LBP_prm : R_x + LBP_prm]
                LBP_R[j, i] = tmpC.detectAndCompute(tmpC, LBP_prm)
                LBP_R_tf    = LBP_R.reshape(-1, 1)

                # 左目
                tmpC        = img_gray[L_y - LBP_prm : L_y + LBP_prm, L_x - LBP_prm : L_x + LBP_prm]
                LBP_L[j, i] = tmpC.detectAndCompute(tmpC, LBP_prm)
                LBP_L_tf    = LBP_L.reshape(-1, 1)
            j += 1
        i += 1
    
    # HOG特徴量-------------------------------------------------------------
    if (pattern == 4) or (pattern == 9) or (pattern == 11) or (pattern == 12) or (pattern == 14) or (pattern == 15) or (pattern == 16) or (pattern == 17) or (pattern == 21):
        HOG_R                      = hog(eyeR_gray, orientations=16, pixels_per_cell=(5, 10))
        HOG_R_T                    = np.conjugate(HOG_R.T)
        HOG_R_T                    = np.reshape(HOG_R_T, (-1, 1))

        HOG_L                      = hog(eyeL_gray, orientations=16, pixels_per_cell=(5, 10))
        HOG_L_T                    = np.conjugate(HOG_L.T)
        HOG_L_T                    = np.reshape(HOG_L_T, (-1, 1))

    # haar-likeの真ん中だけ(視線値Veg)---------------------------------------
    if pattern == 18:
        # 注目座標（目の左上から順に）
        R_y                        = eyeR_center_y
        R_x                        = eyeR_center_x
        L_y                        = eyeL_center_y
        L_x                        = eyeL_center_x
        tmpA                       = img_gray[R_y - eyeH_half : R_y + eyeH_half - 1, R_x - eyeW_half : R_x - 1]
        tmpB                       = img_gray[R_y - eyeH_half : R_y + eyeH_half - 1, R_x : R_x + eyeW_half - 1]
        tmpA_tf                    = tmpA.reshape(-1, 1)
        tmpB_tf                    = tmpB.reshape(-1, 1)
        tmpA_numel                 = tmpA_tf.shape[0]
        tmpB_numel                 = tmpB_tf.shape[0]
        tmpA_result                = np.sum(tmpA_tf, axis=0) / tmpA_numel
        tmpB_result                = np.sum(tmpB_tf, axis=0) / tmpB_numel

        HL_R2                      = np.subtract(tmpA_result, tmpB_result)

        # 左目
        tmpA                       = img_gray[L_y - eyeH_half : L_y + eyeH_half - 1, L_x - eyeW_half : L_x - 1]
        tmpB                       = img_gray[L_y - eyeH_half : L_y + eyeH_half - 1, L_x : L_x + eyeW_half - 1]
        tmpA_tf                    = tmpA.reshape(-1, 1)
        tmpB_tf                    = tmpB.reshape(-1, 1)
        tmpA_numel                 = tmpA_tf.shape[0]
        tmpB_numel                 = tmpB_tf.shape[0]
        tmpA_result                = np.sum(tmpA_tf, asix=0) / tmpA_numel
        tmpB_result                = np.sum(tmpB_tf, asix=0) / tmpB_numel

        HL_L2                      = np.subtract(tmpA_result, tmpB_result)
    
    # dlib顔器官検出による特徴点(DLIB)---------------------------------------
    if (pattern == 19) or (pattern == 20) or (pattern == 21):
        DLIB                       = train_dlib
        DLIB_len                   = len(DLIB)
        DLIB_tf                    = np.resize(DLIB, (DLIB_len, 1))

    # if pattern
    if pattern == 1:
        X           = np.concatenate([HL_R_tf, HL_L_tf])
        X_breakdown = ['HL_R', len(HL_R_tf), 'HL_L', len(HL_L_tf)]
    elif pattern == 2:
        X           = np.concatenate([eyeR_gray_tf, eyeL_gray_tf])
        X_breakdown = [['gray_R', len(eyeR_gray)], ['gray_L', len(eyeL_gray)]]
    elif pattern == 3:
        X           = np.concatenate([eyeR_R_tf, eyeR_G_tf, eyeR_B_tf, eyeL_R_tf, eyeL_G_tf, eyeL_B_tf])
        X_breakdown = [['eyeR_R', len(eyeR_R)], ['eyeR_G', len(eyeR_G)], ['eyeR_B', len(eyeR_B)], ['eyeL_R', len(eyeL_R)], ['eyeL_G', len(eyeL_G)], ['eyeL_B', len(eyeL_B)]]
    elif pattern == 4:
        X           = np.concatenate([HOG_R_T, HOG_L_T])
        X_breakdown = [['HOG_R', len(HOG_R)], ['HOG_L', len(HOG_L)]]
    elif pattern == 5:
        X           = np.concatenate([LBP_R_tf, LBP_L_tf])
        X_breakdown = [['LBP_R', len(LBP_R)], ['LBP_L', len(LBP_L)]]
    elif pattern == 6:
        X           = np.concatenate([eyeR_h_tf, eyeL_h_tf])
        X_breakdown = [['eyeR_h', len(eyeR_h)], ['eyeL_h', len(eyeL_h)]]
    elif pattern == 7:
        X           = np.concatenate([HL_R_tf, HL_L_tf, eyeR_gray_tf, eyeL_gray_tf])
        X_breakdown = []
    elif pattern == 8:
        X           = np.concatenate([HL_R_tf, HL_L_tf, eyeR_R_tf, eyeR_G_tf, eyeR_B_tf, eyeL_R_tf, eyeL_G_tf, eyeL_B_tf])
        X_breakdown = []
    elif pattern == 9:
        X           = np.concatenate([HL_R_tf, HL_L_tf, HOG_R_T, HOG_L_T])
        X_breakdown = []
    elif pattern == 10:
        X           = np.concatenate([eyeR_gray_tf, eyeL_gray_tf, eyeR_G_tf, eyeR_B_tf, eyeL_R_tf, eyeL_G_tf, eyeL_B_tf])
        X_breakdown = []
    elif pattern == 11:
        X           = np.concatenate([eyeR_gray_tf, eyeL_gray_tf, HOG_R_T, HOG_L_T])
        X_breakdown = []
    elif pattern == 12:
        X           = np.concatenate([eyeR_G_tf, eyeR_B_tf, eyeL_R_tf, eyeL_G_tf, eyeL_B_tf, HOG_R_T, HOG_L_T])
        X_breakdown = []
    elif pattern == 13:
        X           = np.concatenate([HL_R_tf, HL_L_tf, eyeR_gray_tf, eyeL_gray_tf, eyeR_G_tf, eyeR_B_tf, eyeL_R_tf, eyeL_G_tf, eyeL_B_tf])
        X_breakdown = []
    elif pattern == 14:
        X           = np.concatenate([HL_R_tf, HL_L_tf, eyeR_gray_tf, eyeL_gray_tf, HOG_R_T, HOG_L_T])
        X_breakdown = []
    elif pattern == 15:
        X           = np.concatenate([HL_R_tf, HL_L_tf, eyeR_G_tf, eyeR_B_tf, eyeL_R_tf, eyeL_G_tf, eyeL_B_tf, HOG_R_T, HOG_L_T])
        X_breakdown = []
    elif pattern == 16:
        X           = np.concatenate([eyeR_gray_tf, eyeL_gray_tf, eyeR_G_tf, eyeR_B_tf, eyeL_R_tf, eyeL_G_tf, eyeL_B_tf, HOG_R_T, HOG_L_T])
        X_breakdown = []
    elif pattern == 17:
        X           = np.concatenate([HL_R_tf, HL_L_tf, eyeR_gray_tf, eyeL_gray_tf, eyeR_R_tf, eyeL_R_tf, eyeR_G_tf, eyeL_G_tf, eyeR_B_tf, eyeL_B_tf, LBP_R_tf, LBP_L_tf, eyeR_h_tf, eyeL_h_tf, HOG_R_T, HOG_L_T])
        X_breakdown = [['HL_R', len(HL_R_tf)], ['HL_L', len(HL_L_tf)], ['gray_R', len(eyeR_gray_tf)], ['gray_L', len(eyeL_gray_tf)], ['eyeR_R', len(eyeR_R_tf)], ['eyeL_R', len(eyeL_R_tf)], ['eyeR_G', len(eyeR_G_tf)], ['eyeL_G', len(eyeL_G_tf)], ['eyeR_B', len(eyeR_B_tf)], ['eyeL_B', len(eyeL_B_tf)], ['LBP_R', len(LBP_R)], ['LBP_L', len(LBP_L)], ['eyeR_h', len(eyeR_h)], ['eyeL_h', len(eyeL_h)], ['HOG_R', len(HOG_R)], ['HOG_L', len(HOG_L)]]
    elif pattern == 18:
        X           = np.concatenate([HL_R2, HL_L2])
        X_breakdown = []
    elif pattern == 19:
        X           = np.concatenate([DLIB_tf])
        X_breakdown = [['DLIB', len(DLIB_tf)]]
    elif pattern == 20:
        X           = np.concatenate([HL_R_tf, HL_L_tf, eyeR_gray_tf, eyeL_gray_tf, DLIB_tf])
        X_breakdown = []
    elif pattern == 21:
        X           = np.concatenate([HL_R_tf, HL_L_tf, eyeR_gray_tf, eyeL_gray_tf, HOG_R_T, HOG_L_T, DLIB_tf])
        X           = np.ravel(X)
        X_breakdown = [['HL_R', len(HL_R_tf)], ['HL_L', len(HL_L_tf)], ['gray_R', len(eyeR_gray_tf)], ['gray_L', len(eyeL_gray_tf)], ['HOG_R', len(HOG_R)], ['HOG_L', len(HOG_L)], ['DLIB', len(DLIB_tf)]]
  
    # function終端
    # print('"特徴量選択プログラム(getFeatureVal_hatanaka.py)"の処理を終了します。')
    return [X, X_breakdown]