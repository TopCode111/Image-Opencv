# -*- coding: utf-8 -*-
# プログラム実行に必要なライブラリを読み込む
import os
import sys
import cv2
import time
import shutil
import datetime
import itertools
import pandas as pd
import numpy  as np
import numpy.matlib
from numpy import linalg as LA
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 外部ファイルを読み込む
import getFeatureVal_hatanaka
import myPLS2_vip


# コマンドライン引数の正誤を判定
def cmdline_check():
    # print('"cmdline_check関数"の処理に移行します。')

    sys_args     = sys.argv
    argv_length  = len(sys_args)
    default_path = "/data_2020/kado/kado_gameVer_2"
    """
    sys_args[1] : 座標成分の行列名(pre_round:予測値, preY:真値, diff:誤差)
    sys_args[2] : 実行結果の保存先フォルダ(デフォルト値は"/data_2020/kado/kado_gameVer_2")
    """

    # コマンドライン引数の長さが"1"または"2"であることを判定
    if argv_length == 2 or argv_length == 3:
        # コマンドライン引数の先頭で指定した行列名を判定して問題なければその行列を返却
        if sys_args[1] == "pre_round" or sys_args[1] == "preY" or sys_args[1] == "diff":
            if argv_length == 2:
                argv_args = [sys_args[1], default_path]
            elif argv_length == 3:
                argv_args = [sys_args[1], sys_args[2]]

        # ifの条件に合致しなければエラーを出力し、プログラムを強制終了
        else:
            print('The first content of the command line argument is incorrect("pre_round" or "preY" or "diff").')
            sys.exit(1)

    # ifの条件に合致しなければエラーを出力し、プログラムを強制終了
    else:
        print('Wrong length of command line arguments(Command line length must be "1" or "2").')
        sys.exit(1)

    # function終端
    return argv_args


# list配列をndarray配列に変換
def num2cell(list_arys):
    # print('"num2cell関数"の処理に移行します。')

    # "list型の配列"であることを判定
    if isinstance(list_arys, list):
        num2cell_arys   = []
        list_chain_arys = list(itertools.chain.from_iterable(str(list_arys)))

        for ary in list_chain_arys:
            num2cell_arys.append(ary)
        num2cell_ndarys = np.array(num2cell_arys)
        num2cell_ndarys = np.resize(num2cell_ndarys, (11))

    # ifの条件に合致しなければエラーを出力し、プログラムを強制終了
    else:
        print("This arrays isn't in list format.")
        sys.exit(1)
    
    # function終端
    return num2cell_ndarys


# int値からlist配列を作成してndarray配列に変換
def cell2mat(bd):
    # print('"cell2mat関数"の処理に移行します。')

    # "int型の値"であることを判定
    if isinstance(bd, int):
        tmp_arys        = []
        tmp_arys.append(bd)
        tmp_arys.append(1)
        cell2mat_arys   = [tmp_arys]
        cell2mat_ndarys = np.array(cell2mat_arys)
    
    # ifの条件に合致しなければエラーを出力し、プログラムを強制終了
    else:    
        print("This args isn't in int format.")
        sys.exit(1)

    # function終端
    return cell2mat_ndarys


# 2次元座標成分(行列)を"result_csv"に出力
def output_csv(coordinate_x_y, train_folder, suffix, idx_arys):
    # print('"output_csv関数"の処理に移行します。')

    # "ndarray型"の配列であることを判定
    if isinstance(coordinate_x_y, numpy.ndarray):
        dt_now      = datetime.datetime.now()
        dt_now      = dt_now.strftime('%Y%m%d%H%M%S')
        i           = 0
        df          = pd.DataFrame()
        row_length  = coordinate_x_y.shape[0]

        # 保存先のcsvファイル名を設定
        result_csv  = f"{train_folder}/result_({suffix})({dt_now}).csv"
        
        # 座標成分行列の1列目と2列目の要素をPDデータフレームに代入
        while i < row_length:
            df.loc[i, 0] = idx_arys[i]          # frame画像の番号
            df.loc[i, 1] = coordinate_x_y[i, 1] # PDデータフレーム：1列目, 座標成分行列：y座標
            df.loc[i, 2] = coordinate_x_y[i, 0] # PDデータフレーム：2列目, 座標成分行列：x座標
            i += 1
        df.to_csv(result_csv, encoding="utf-8", mode="w", header=False, index=False)

    # ifの条件に合致しなければエラーを出力し、プログラムを強制終了
    else:
        print("This args isn't in numpy.ndarray format.")
        sys.exit(1)
    
    
# "main関数"の処理開始
def main(argv_args):
    # print('"main関数"の処理に移行します。')

    # タイマー開始
    time_start = time.time()

    # カレントディレクトリまでの絶対パスを取得
    current_dir       = os.path.dirname(__file__)

    # 保存先のフォルダパスを作成(存在している場合は作成しない)
    train_folder      = f"{current_dir}/{argv_args[1]}"
    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)
    
    # プログラム処理に必要なファイルとフォルダを複製(存在している場合は複製しない)
    fileutils = ["eye_point.csv", "mouse_point.csv", "dlib_point.csv"]
    for fs in fileutils:      
        if not os.path.isfile(f"{train_folder}/{fs}"):
            shutil.copy(f"{current_dir}/{fs}",   f"{train_folder}/{fs}")
    
    if not os.path.isdir(f"{train_folder}/frame"):
        shutil.copytree(f"{current_dir}/frame", f"{train_folder}/frame")

    # 前処理(使用する画像およびcsvデータが入ったフォルダを指定)--------------------------------------------------------------------
    test_folder         = train_folder
    
    # 特徴量選択
    feature_pattern     = [21]                 # 変換する特徴量の選択(getFeatureVal_hatanaka.m参照) # (21:HL/I/HOG/DLIB)
    pattern             = feature_pattern[0]   # 特徴量を"pattern"変数に代入
    featureN            = len(feature_pattern) # 実行する特徴量選択の長さ
    
    # 画像枚数
    maxN                = 1641                 # 実験用画像枚数(写真総数)
    trainN              = 300                  # 学習画像枚数()
    testN               = maxN - trainN        # テスト画像枚数

    # 全ての学習用画像を使用
    train_i             = 1                    # 学習データのframe番号[始まり]
    test_i              = trainN + 1           # テストデータのframe番号[始まり]
    
    # 目の両端＋αのサイズで切り取る
    eyeH_alpha          = 4               
    eyeW_alpha          = 4

    # PLS反復回数(初期化)
    m                   = 0
    
    # PLSにてvipを計算するなら1，しないなら0
    mode                = 0
    
    # モニターのサイズ[縦,横]                
    disp_size           = [1824, 2736]
    
    # figure番号(グラフ作成用)
    fnum                = 1

    # 目の配列のサイズ
    eye_max             = 6

    # 目の座標位置のインデックス(列)
    eyeR_y_columns      = [12, 13, 14, 15, 16, 17]
    eyeR_x_columns      = [0, 1, 2, 3, 4, 5]
    eyeL_y_columns      = [18, 19, 20, 21, 22, 23]
    eyeL_x_columns      = [6, 7, 8, 9, 10, 11]

    # 必要データファイルの読み込み-----------------------------------------------

    # 画像上における目の座標を読み込み
    train_eye           = pd.read_csv(f"{train_folder}/eye_point.csv",   encoding='shift_jis') # 2行目以降読み込み
    test_eye            = pd.read_csv(f"{test_folder}/eye_point.csv",    encoding='shift_jis') # 2行目以降読み込み
    
    # 注視点の真値を読み込み, ground trueth
    train_gt            = pd.read_csv(f"{train_folder}/mouse_point.csv", encoding='shift_jis') # 2行目以降読み込み
    test_gt             = pd.read_csv(f"{test_folder}/mouse_point.csv",  encoding='shift_jis') # 2行目以降読み込み
    
    # 画像上における顔特徴点の座標を読み込み（特徴量にDLIBを用いるときのみ必要！）
    train_dlib          = pd.read_csv(f"{train_folder}/dlib_point.csv",  encoding='shift_jis') # 2行目以降読み込み
  
    # diff_table---------------------------------------------------------------
    diff_table          = np.empty((featureN+2, 12), dtype='str') # 空行列からなるcell配列を返す
    diff_table[0, 0]    = test_folder
    diff_table[0, 2]    = "RMSE[pixel]"
    diff_table[0, 5]    = "最大誤差[pixel]"
    diff_table[0, 7]    = "最小誤差[pixel]"
    diff_table[0, 9]    = "精度"
    diff_table[1, 1:]   = ["入力次元", "垂直", "水平", "２次元", "垂直", "水平", "垂直", "水平", "垂直", "水平", "２次元"]
  
    feature_i           = 0
    exe_cnt             = 1
    while feature_i < featureN:
        # 学習用画像から特徴量を取得
        # kを求める----------------------------------------------------------------
        # 画像読み込み
        img_path        = f"{train_folder}/frame/frame{str(1)}.png"
        img_bgr         = cv2.imread(img_path)
        img_rgb         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #RGB画像に変換
      
        # 顔特徴点座標データ読み込み（DLIB特徴量使用時のみ）
        dlib            = train_dlib.iloc[0, :]
      
        # 目の中心座標
        eyeR_y_sum      = 0
        eyeR_x_sum      = 0
        eyeL_y_sum      = 0
        eyeL_x_sum      = 0
        eye_num         = 0
        while eye_num < eye_max:
            eyeR_y_sum  = np.add(eyeR_y_sum, train_eye.iloc[0, eyeR_y_columns[eye_num]])
            eyeR_x_sum  = np.add(eyeR_x_sum, train_eye.iloc[0, eyeR_x_columns[eye_num]])
            eyeL_y_sum  = np.add(eyeL_y_sum, train_eye.iloc[0, eyeL_y_columns[eye_num]])
            eyeL_x_sum  = np.add(eyeL_x_sum, train_eye.iloc[0, eyeL_x_columns[eye_num]])
            eye_num    += 1

        eyeR_y          = Decimal(eyeR_y_sum / 6)
        eyeR_x          = Decimal(eyeR_x_sum / 6)
        eyeR_center     = [eyeR_y.quantize(Decimal('0'), rounding = ROUND_DOWN), eyeR_x.quantize(Decimal('0'), rounding = ROUND_DOWN)] # 右目の中心座標(y,x)
        eyeL_y          = Decimal(eyeL_y_sum / 6)
        eyeL_x          = Decimal(eyeL_x_sum / 6)
        eyeL_center     = [eyeL_y.quantize(Decimal('0'), rounding = ROUND_DOWN), eyeL_x.quantize(Decimal('0'), rounding = ROUND_DOWN)] # 左目の中心座標(y,x)
        eyeRL_center    = [eyeR_center, eyeL_center]
        eye_center      = list(itertools.chain.from_iterable(eyeRL_center))
        eyeW_arys       = [np.subtract(train_eye.iloc[0, 3], train_eye.iloc[0, 0]), np.subtract(train_eye.iloc[0, 9], train_eye.iloc[0, 6])] 
        eyeW            = np.add(np.amax(eyeW_arys), (eyeW_alpha * 2))
        eyeH_arys       = [np.subtract(train_eye.iloc[0, 17], train_eye.iloc[0, 14]), np.subtract(train_eye.iloc[0, 22], train_eye.iloc[0, 19])]
        eyeH            = np.add(np.amax(eyeH_arys), (eyeH_alpha * 2))
        eye_size        = [eyeH, eyeW]
        eye_numel       = eyeH * eyeW

        if pattern == 21:
            [X, bd]     = getFeatureVal_hatanaka.getFeatureVal_hatanaka(exe_cnt, img_rgb, eye_center, eye_size, dlib, pattern)
        else:
            [X, bd_tmp] = getFeatureVal_hatanaka.getFeatureVal_hatanaka(exe_cnt, img_rgb, eye_center, eye_size, dlib, pattern)
    
        k               = X.shape[0]
        X               = np.zeros((trainN, k))
        Y               = np.zeros((trainN, 2))
        exe_cnt        += 1

        # 特徴量----------------------------------------------------------------
        i               = 0
        index           = train_i
        dlib_i          = index - 1
        eye_i           = index - 1
        train_gt_i      = index - 1
    
        while i < trainN:
            # 画像読み込み
            img_path              = f"{train_folder}/frame/frame{str(index)}.png"
            img_bgr               = cv2.imread(img_path)
            img_rgb               = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #RGB画像に変換
            
            # 顔特徴点座標データ読み込み（DLIB特徴量使用時のみ）
            dlib                  = train_dlib.iloc[dlib_i, :]

            # 目の中心座標
            eyeR_y_sum            = 0
            eyeR_x_sum            = 0
            eyeL_y_sum            = 0
            eyeL_x_sum            = 0
            eye_num               = 0
            while eye_num < eye_max:
                eyeR_y_sum        = np.add(eyeR_y_sum, train_eye.iloc[eye_i, eyeR_y_columns[eye_num]])
                eyeR_x_sum        = np.add(eyeR_x_sum, train_eye.iloc[eye_i, eyeR_x_columns[eye_num]])
                eyeL_y_sum        = np.add(eyeL_y_sum, train_eye.iloc[eye_i, eyeL_y_columns[eye_num]])
                eyeL_x_sum        = np.add(eyeL_x_sum, train_eye.iloc[eye_i, eyeL_x_columns[eye_num]])
                eye_num += 1

            eyeR_y                = Decimal(eyeR_y_sum / 6)
            eyeR_x                = Decimal(eyeR_x_sum / 6)
            eyeR_center           = [eyeR_y.quantize(Decimal('0'), rounding = ROUND_DOWN), eyeR_x.quantize(Decimal('0'), rounding = ROUND_DOWN)] # 右目の中心座標(y,x)
            eyeL_y                = Decimal(eyeL_y_sum / 6)
            eyeL_x                = Decimal(eyeL_x_sum / 6)
            eyeL_center           = [eyeL_y.quantize(Decimal('0'), rounding = ROUND_DOWN), eyeL_x.quantize(Decimal('0'), rounding = ROUND_DOWN)] # 左目の中心座標(y,x)

            # 特徴量
            eyeRL_center          = [eyeR_center, eyeL_center]
            eye_center            = list(itertools.chain.from_iterable(eyeRL_center))
            eye_size              = [eyeH, eyeW]
            [X[i, :], bd_tmp]     = getFeatureVal_hatanaka.getFeatureVal_hatanaka(exe_cnt, img_rgb, eye_center, eye_size, dlib, pattern)
           
            # 教師データ
            Y[i, 0]               = train_gt.iloc[train_gt_i, 1]
            Y[i, 1]               = train_gt.iloc[train_gt_i, 0]

            i                    += 1
            index                += 1
            dlib_i               += 1
            eye_i                += 1
            train_gt_i           += 1
            exe_cnt              += 1

        # Xを変数ごとに標準化
        trainX_mean               = np.mean(X, axis=0)
        trainX_std                = np.std(X,  axis=0)

        # "trainX_std"の標準偏差が"０"の場合は１にする
        trainX_idxes              = np.nonzero(trainX_std == 0)
        trainX_std[trainX_idxes]  = 1

        trainX_mean_repmat        = np.matlib.repmat(trainX_mean, trainN, 1)
        trainX_std_repmat         = np.matlib.repmat(trainX_std,  trainN, 1)
        X_sub                     = np.subtract(X, trainX_mean_repmat)
        X2                        = np.divide(X_sub, trainX_std_repmat)
        
        # Yを変数ごとに標準化
        trainY_mean               = np.mean(Y, axis=0)
        trainY_std                = np.std(Y,  axis=0)

        # "trainY_std"の標準偏差が"０"の場合は１にする
        trainY_idxes              = np.nonzero(trainY_std == 0)
        trainY_std[trainY_idxes]  = 1
        
        trainY_mean_repmat        = np.matlib.repmat(trainY_mean, trainN, 1)
        trainY_std_repmat         = np.matlib.repmat(trainY_std,  trainN, 1)
        Y_sub                     = np.subtract(Y, trainY_mean_repmat)
        Y2                        = np.divide(Y_sub, trainY_std_repmat)
       
        # PLS2実行-------------------------------------------------------------
        if pattern == 18:
            m    = 2
        else:
            m    = 30 # この変数を変えることでRMSEを良くも悪くもさせる(default:m=30)
  
        if pattern == 21:
            mode = 1
            [W, a_PLS, Y_av, X_av, T, P, vip]     = myPLS2_vip.myPLS2_vip(X2, Y2, m, mode)
        else:
            mode = 0
            [W, a_PLS, Y_av, X_av, T, P, vip_tmp] = myPLS2_vip.myPLS2_vip(X2, Y2, m, mode)
        """
        print("回帰結果")
        print(f"W:{W.shape}")
        print(W)
        print(f"a_PLS:{a_PLS.shape}")
        print(a_PLS)
        print(f"Y_av:{Y_av.shape}")
        print(Y_av)
        print(f"X_av:{X_av.shape}")
        print(X_av)
        print(f"T:{T.shape}")
        print(T)
        print(f"P:{P.shape}")
        print(P)
        print(f"vip:{vip.shape}")
        print(vip)
        """

        con_P              = np.conjugate(P.T)
        con_P_W            = np.dot(con_P, W)
        inv_P_W            = LA.inv(con_P_W)
        W_inv_P_W          = np.dot(W, inv_P_W)
 
        # 回帰結果
        b_PLS              = np.dot(W_inv_P_W, a_PLS)

        # 予測------------------------------------------------------------------
        pre                = np.zeros((testN, 2))
        dec_pre            = np.zeros((testN, 2))
        preX               = np.zeros((testN, k))
        preY               = np.zeros((testN, 2))
        i                  = 0
        index              = test_i
        dlib_i             = index - 1
        eye_i              = index - 1
        test_gt_i          = index - 1
        idx_arys           = []
      
        while i < testN:
            idx_arys.append(index)
            # 画像読み込み
            img_path       = f"{test_folder}/frame/frame{str(index)}.png"
            img_bgr        = cv2.imread(img_path)
            img_rgb        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #RGB画像に変換

            # 顔特徴点座標データ読み込み（DLIB特徴量使用時のみ
            dlib           = train_dlib.iloc[dlib_i, :]
            
            # 目の中心座標
            eyeR_y_sum     = 0
            eyeR_x_sum     = 0
            eyeL_y_sum     = 0
            eyeL_x_sum     = 0
            eye_num        = 0
            while eye_num < eye_max:
                eyeR_y_sum             = np.add(eyeR_y_sum, test_eye.iloc[eye_i, eyeR_y_columns[eye_num]])
                eyeR_x_sum             = np.add(eyeR_x_sum, test_eye.iloc[eye_i, eyeR_x_columns[eye_num]])
                eyeL_y_sum             = np.add(eyeL_y_sum, test_eye.iloc[eye_i, eyeL_y_columns[eye_num]])
                eyeL_x_sum             = np.add(eyeL_x_sum, test_eye.iloc[eye_i, eyeL_x_columns[eye_num]])
                eye_num               += 1

            eyeR_y                     = Decimal(eyeR_y_sum / 6)
            eyeR_x                     = Decimal(eyeR_x_sum / 6)
            eyeR_center                = [eyeR_y.quantize(Decimal('0'), rounding = ROUND_DOWN), eyeR_x.quantize(Decimal('0'), rounding = ROUND_DOWN)] # 右目の中心座標(y,x)
            eyeL_y                     = Decimal(eyeL_y_sum / 6)
            eyeL_x                     = Decimal(eyeL_x_sum / 6)
            eyeL_center                = [eyeL_y.quantize(Decimal('0'), rounding = ROUND_DOWN), eyeL_x.quantize(Decimal('0'), rounding = ROUND_DOWN)] # 左目の中心座標(y,x)
            
            # 特徴量
            eyeRL_center               = [eyeR_center, eyeL_center]
            eye_center                 = list(itertools.chain.from_iterable(eyeRL_center))
            eye_size                   = [eyeH, eyeW]
            [preX[i, :], bd_tmp]       = getFeatureVal_hatanaka.getFeatureVal_hatanaka(exe_cnt, img_rgb, eye_center, eye_size, dlib, pattern)
          
            # 標準化
            preX_trainX_mean           = np.subtract(preX[i, :], trainX_mean)
            preX[i, :]                 = np.divide(preX_trainX_mean, trainX_std)
            preX_X_av                  = np.subtract(preX[i, :], X_av)
            b_PLS_Y_av                 = np.add(b_PLS, Y_av)
            tmp                        = np.dot(preX_X_av, b_PLS_Y_av) # PLS
            tmp_trainY_std             = np.multiply(tmp, trainY_std)
            pre[i, :]                  = np.add(tmp_trainY_std, trainY_mean)
        
            # 四捨五入
            pre_col1                   = Decimal(pre[i, 1])
            pre_col2                   = Decimal(pre[i, 0])
            dec_pre[i, :]              = [pre_col1.quantize(Decimal('0'), rounding=ROUND_HALF_UP), pre_col2.quantize(Decimal('0'), rounding=ROUND_HALF_UP)] # pre行列を"1"の方向に丸める(四捨五入)

            # 真値
            preY[i, :]                 = [test_gt.iloc[test_gt_i, 1], test_gt.iloc[test_gt_i, 0]]
            i                         += 1
            index                     += 1
            dlib_i                    += 1
            eye_i                     += 1
            test_gt_i                 += 1
            exe_cnt                   += 1
    
        # 誤差の平均値と標準誤差を求める-----------------------------------------
        pre_round                   = dec_pre
        
        # pre_roundの値が"0"より小さい場合
        pr_idxes1                   = np.nonzero(pre_round < 0)
        pre_round[pr_idxes1, :]     = 0 # 教師用データの範囲外に出たものは丸める

        # pre_roundの値が"disp_size[0](1824)"より大きい場合
        pr_idxes2                   = np.nonzero(pre_round[:, 0] > disp_size[0])
        pre_round[pr_idxes2, 0]     = disp_size[0]

        # pre_roundの値が"disp_size[1](2736)"より大きい場合
        pr_idxes3                   = np.nonzero(pre_round[:, 1] > disp_size[1])
        pre_round[pr_idxes3, 1]     = disp_size[1]

        # 予測値の2次元座標
        pre_round_2_add             = np.add(np.power(pre_round[:, 0], 2), np.power(pre_round[:, 1], 2))
        pre_round_2                 = np.sqrt(pre_round_2_add)
        
        # 真値の2次元座標
        preY_2_add                  = np.add(np.power(preY[:, 0], 2), np.power(preY[:, 1], 2))
        preY_2                      = np.sqrt(preY_2_add)

        diff                        = np.subtract(pre_round, preY) # diff:誤差,pre_round:予測値（丸め済）,preY:真値

        disp_diag_add               = np.add(np.power(disp_size[0], 2), np.power(disp_size[1], 2))
        disp_diag                   = np.sqrt(disp_diag_add)

        diff_rms                    = np.sqrt(np.square(diff))
        diff_abs                    = np.abs(diff)
        diff_max                    = np.amax(diff_abs)
        diff_min                    = np.amin(diff_abs)

        # 2次元上の予測誤差を求める
        diff_2_add                  = np.add(np.power(diff[:, 0], 2), np.power(diff[:, 1], 2))
        diff_2                      = np.sqrt(diff_2_add)
        
        diff_rms_add                = diff_2_add
        diff_rms2_tmp               = np.sum(diff_rms_add)
        diff_rms2                   = np.sqrt(diff_rms2_tmp / diff.shape[0])

        num2cell_arg1               = diff_rms[0] / (disp_size[0] * 100)
        num2cell_arg2               = diff_rms[1] / (disp_size[1] * 100)
        num2cell_arg3               = diff_rms2   / (disp_diag    * 100)
        num2cell_args               = [k, diff_rms, diff_rms2, diff_max, diff_min, num2cell_arg1, num2cell_arg2, num2cell_arg3]

        diff_table[feature_i+2, 1:] = num2cell(num2cell_args)
        feature_i      += 1
        exe_cnt         = 0

    if pattern == 21:
        vip_mean        = np.zeros((4, 2))
        vs              = 1
        bd_index        = 0
        vip_mean_index  = 0
        i               = 0
        
        while i < 5:
            ve1                           = cell2mat(bd[bd_index][1])
            ve1                           = ve1[0, 0] * ve1[0, 1]
            ve2                           = cell2mat(bd[bd_index+1][1])
            ve2                           = ve2[0, 0] * ve2[0, 1]
            ve                            = ve1 + ve2
            np_mean_vip                   = np.mean(vip[vs:vs+ve-1, :], axis=0)
            vip_mean[vip_mean_index, :]   = np_mean_vip[0]
            vs                            = vs + ve
            i                            += 2
            bd_index                     += 2
            vip_mean_index               += 1

        # 0の行を消す
        del_num         = 0
        vip_mean_idxes  = np.nonzero(vip_mean[:, 0] == 0)
        for vip_mean_idx in vip_mean_idxes:
            vip_mean    = np.delete(vip_mean, vip_mean_idx, 0)
            del_num    += 1
            i += 1

        ve1             = cell2mat(bd[6][1])
        ve              = ve1[0, 0] * ve1[0, 1]
        np_mean_vip     = np.mean(vip[vs:vs+ve-1, :], axis=0)
        [v_r, v_c]      = vip_mean.shape
        cover_index     = v_r + del_num
        vip_mean        = np.resize(vip_mean, (cover_index, 2))
        vip_mean[3, :]  = np_mean_vip[0]

        # コマンドライン引数の先頭に指定した行列名の"行列(接尾辞)"を"output_csv"関数に渡す
        if argv_args[0] == "pre_round":
            com_args = pre_round
            suffix = "予測値"
        elif argv_args[0] == "preY":
            com_args = preY
            suffix = "真値"
        elif argv_args[0] == "diff":
            com_args = diff
            suffix = "誤差"
        output_csv(com_args, train_folder, suffix, idx_arys)

    # 実行結果表示
    """
    print(pre_round[0:testN, 0])  # 予測値のy座標成分
    print("\n")
    print(pre_round[0:testN, 1])  # 予測値のx座標成分
    print("\n")
    print(pre_round_2)            # 予測値の2次元座標成分
    print("\n")
    print(preY[0:testN, 0])       # 真値のy座標成分
    print("\n")
    print(preY[0:testN, 1])       # 真値のx座標成分
    print("\n")
    print(preY_2)                 # 真値の2次元座標成分
    print("\n")
    print(diff[0:testN, 0])       # 誤差(予測値-真値)のy座標成分
    print("\n")
    print(diff[0:testN, 1])       # 誤差(予測値-真値)のx座標成分
    print("\n")
    print(diff_2)                 # 誤差(予測値-真値)の2次元座標成分
    print("\n")
    print(diff_table)             # RMSE
    """

    # タイマー終了
    time_end    = time.time()
    result_time = time_end - time_start
    print('"注視点推定プログラム(gazePrediction_hatanaka.py)"の処理を終了します。')
    print(f'経過時間は"{result_time}"秒です。')


# "注視点推定プログラム(gazePrediction_hatanaka.py)"の処理開始
if __name__ == '__main__':
    print('"注視点推定プログラム(gazePrediction_hatanaka.py)"の処理を開始します。')
    argv_args = cmdline_check()
    main(argv_args)