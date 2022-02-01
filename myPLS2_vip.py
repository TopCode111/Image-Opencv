# -*- coding: utf-8 -*-
# プログラム実行に必要なライブラリを取り込む
import os
import numpy as np
from numpy import linalg as LA
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# "PLS回帰プログラム(myPLS2_vip.py)"の処理開始
def myPLS2_vip(X, Y, m, mode):
    # print('"PLS回帰プログラム(myPLS2_vip.py)"の処理を開始します。')
    # PLS1で重み行列W，パラメータベクトルaを学習
    #
    # X : データ行列 n x k
    # Y : 出力ベクトル（教師データ） n×d
    # m : 打ち切り次数 ( mの最大値 =  d )
    #
    # a : 回帰パラメータ m x 1
    # W : 重み行列 k x m
    # Y_av : 出力平均値
    # X_av : データ平均ベクトル（Xの各列の平均）
    # mode : mode=1ならvipを計算する
    #
    # T : スコア行列 n x m
    # P : ローディング行列 k x m

    # 初期化
    [n, k] = X.shape
    [q, d] = Y.shape

    E      = 0.01
    n2     = 1
   
    # 平均除去
    X_av   = np.mean(X, axis=0)
    Y_av   = np.mean(Y, axis=0)

    i = 0
    while i < k: 
        X[:, i] = X[:, i] - X_av[i]
        i += 1

    i = 0
    while i < d:
        Y[:, i] = Y[:, i] - Y_av[i]
        i += 1

    W  = np.zeros((k, m))
    T  = np.zeros((n, m))
    P  = np.zeros((k, m))
    a  = np.zeros((m, d))
    ss = np.zeros((n, d, m))
    Yj = Y
    Xj = X 

    # ------------- ループ -------------
    j = 0
    while j < m:
        print(j)
        # ------------ Step1 -------------s
        # ウェイトベクトルwの初期化
        wj              = np.zeros((k, 1))
        wj[99, :]       = 1   # |w|=1となるように % 特徴量の要素が0である部分と乗算しない要素に1を代入すること！ % デフォルト:wj(100) = 1
     
        while(1):
            # ------------ Step2 -------------
            # スコア行列の計算
            tj          = np.dot(Xj, wj)
            
            # ------------ Step3 -------------
            # 単回帰でパラメータ計算
            con_tj      = np.conjugate(tj.T)
            aj_mul1     = np.dot(con_tj, Yj)
            aj_mul2     = np.dot(con_tj, tj)
            aj          = np.divide(aj_mul1, aj_mul2)

            # ------------ Step4 -------------
            # ローディングベクトル計算
            con_Xj      = np.conjugate(Xj.T)
            pj_mul1     = np.dot(con_Xj, tj)
            pj_mul2     = np.dot(con_tj, tj)
            pj          = np.divide(pj_mul1, pj_mul2)

            # ------------ Step5 -------------
            con_aj      = np.conjugate(aj.T)
            wj_tmp_mul1 = np.dot(con_Xj, Yj)
            wj_tmp_mul2 = np.dot(wj_tmp_mul1, con_aj)
            wj_tmp_mul3 = LA.norm(wj_tmp_mul2)
            wj_tmp      = np.divide(wj_tmp_mul2, wj_tmp_mul3)
        
            # 収束判定
            cnv_arg     = np.subtract(wj, wj_tmp) 
            err         = LA.norm(cnv_arg)
            if err < E:
                W[:, j] = wj[:, 0]
                T[:, j] = tj[:, 0]
                a[j, :] = aj[0, :]
                P[:, j] = pj[:, 0]
                break
            wj          = wj_tmp
        
        # ------------ Step6 -------------
        # データ更新
        con_pj          = np.conjugate(pj.T)
        tj_pj           = np.dot(tj, con_pj)
        Xj              = np.subtract(Xj, tj_pj)
        
        tj_a            = tj * a[j, :]
        Yj_tj_a_sub     = np.subtract(Yj, tj_a)
        ss[:, :, j]     = Yj_tj_a_sub
        Yj              = Yj_tj_a_sub

        j              += 1

# --------- ループ終端 ----------
    if mode == 1:
        # VIP計算---------------------------
        vip = np.zeros((k, d))
        n1 = 0
        while n1 < k:
            n2 = 0
            while n2 < d: 
                f = 0.0
                g = 0.0
                j = 0
                while j < m:
                    # f = f + np.power(W[n1, j], 2) * np.var(ss[:, n2, j])
                    # g = g + np.var([ss[:, n2, j]])
                    f  = f + np.power(W[n1, j], 2) * np.var(T[:, j] * a[j, n2])
                    g  = g + np.var(T[:, j] * a[j, n2])
                    j += 1
                vip[n1, n2]  = np.sqrt((k * f) / g)
                n2          += 1
            n1 += 1
    else:
        vip = []
    
    # function終端
    # print('"PLS回帰プログラム(myPLS2_vip.py)"の処理を終了します。')
    return [W, a, Y_av, X_av, T, P, vip]