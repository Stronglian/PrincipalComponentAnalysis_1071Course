# -*- coding: utf-8 -*-
"""
@author: StrongPria
機器/統計學習:主成分分析(Principal Component Analysis, PCA):
    https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8-%E7%B5%B1%E8%A8%88%E5%AD%B8%E7%BF%92-%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90-principle-component-analysis-pca-58229cd26e71
"""
import os
import numpy as np
import cv2
#%% 
class PCA():
    def __init__(self, data, compressionRate, boolReCal = False, dims = 0):
        self.data = data.copy().astype('float32')
        self.compressionRate = compressionRate
        self.boolReCal = boolReCal
        
        self.dims = dims
        return
    
    def EvaluateMean(self, y):
#        return  y.sum(axis = 0)/len(y)
        return  y.mean(axis = 0)
    
    def TranslateData(self, data, mean):
        return data - mean
    
    def EvaluateCovarianceMatrix(self, data):
        return np.dot(data.T, data) / (data.shape[0] -1) #部分取樣，所以 -1
#        return np.cov(data.T)
    
    def Cal_Eigen(self, covMatrix, boolReCal = False):
        npyName_val = "Cal_Eigen_val"+ str(self.compressionRate) + "_" + str(self.dims)+".npy"
        npyName_vec = "Cal_Eigen_vec"+ str(self.compressionRate) + "_" + str(self.dims)+".npy"
        if os.path.exists(npyName_val) and os.path.exists(npyName_vec) and (not boolReCal):
            eigenvalues  =  np.load(npyName_val)
            eigenvectors =  np.load(npyName_vec)
        else:
            eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
            eigenvalues  = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            np.save(npyName_val, eigenvalues)
            np.save(npyName_vec, eigenvectors)
        return eigenvalues, eigenvectors
    
    def FLOW_1dim(self):
        # step 1
        self.y_mean = self.EvaluateMean(self.data)
        # step 2
        self.data = self.TranslateData(self.data, self.y_mean)
        # step 3: Evaluate the covariance matrix M
        self.covMatrix = self.EvaluateCovarianceMatrix(self.data)
        # step 4: Find eigenvalues and corresponding eigenvectors of M
        eigenvalues, eigenvectors = self.Cal_Eigen(self.covMatrix, boolReCal = self.boolReCal) #c, v
        ## sort
        argEigenvals = eigenvalues.argsort()[::-1]
#        print(len(argEigenvals), argEigenvals)
        eigenvalues  = eigenvalues[argEigenvals]
        eigenvectors = eigenvectors[:, argEigenvals]
        ## normalize
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors[:, 0])
        # step 5: Find p
        ## sum Of corresponding eigenvectors
        sumOfEigenvals = eigenvalues.sum()
#        print(sumOfEigenvals)
        ## find p
#        print(argEigenvals)
        sumOfP = 0.0
        for i, tmpVal in enumerate(eigenvalues):
            sumOfP += tmpVal
#            print(sumOfP)
            if sumOfP >= self.compressionRate * sumOfEigenvals:
                break
        self.p = i +1 # 取的個數，因為在 :p 要取 p 個，所以要 +1
        # step 6: Project each of the vectors in {x1, x2, …, xk} on the space spanned by the first p eigenvectors
        self.D = np.dot(eigenvectors[:, :self.p].T, self.data.T)
        
        #轉存
        self.eigenvalues, self.eigenvectors = eigenvalues, eigenvectors
        return self.D
    
    def Recover_1dim(self, D):
        # step 7: Recover approximate face image for each yi:
        return np.dot(D.T, self.eigenvectors[:, :self.p].T) + self.y_mean
#%% 
if __name__ == '__main__' :
    import time
    startTime = time.time()
    #%% 共同參數
    pokerFolder = "./poker_card_colorful/"
    compressionRate = 1
    outputFolder = "./outputFolder_"+str(compressionRate)+"(colorful)/"
#    pokerFolder = "./poker13/"
#    outputFolder = "./outputFolder_poker13/"
    
    #%% 讀檔設置
    ###### 檔名 - 再處理非讀取資料的問題
    imgNameList = np.array(os.listdir(pokerFolder))
    imgNameList = imgNameList[np.argsort([int(na.split(".", 1)[0]) for na in imgNameList ], axis = 0)]
    imgFilenameExtension = '.' + imgNameList[0].rsplit(".", 1)[-1]
    ###### 取用首張圖片格式
    tmp = cv2.imread(pokerFolder + imgNameList[0], 1)
    rows, cols = tmp.shape[:2]
    if len(tmp.shape) >= 3:
        dims = tmp.shape[2]
    else:
        dims = 1
    ###### 蒐集成 Array
    imgArr = np.zeros((len(imgNameList), rows, cols, dims))
    for i, name in enumerate(imgNameList):
        imgArr[i, :, :] = cv2.imread(pokerFolder + name, 1).copy()
    ###### 預處理
    imgArr = imgArr.reshape(len(imgNameList), rows * cols, dims) / 255
    
    #%% 計算
    output = np.zeros_like(imgArr)
    for i in range(dims):
        pca = PCA(imgArr[:, :, i], compressionRate, boolReCal = False, dims = i)
        D = pca.FLOW_1dim()
    #    eigenvalues, eigenvectors = pca.eigenvalues, pca.eigenvectors
        p = pca.p
        print("p:", p)
        output[:, :, i] = pca.Recover_1dim(D).copy()
        print("PCA", i, "計算完成,", round(time.time() - startTime, 4), "sec")
    
    #%% 算 MSE
    ###### 後處理
    output = np.clip(output, 0, 1)
    imgArr = imgArr * 255
    output = output * 255
    mse = ( np.square(imgArr - output)).mean() #(axis = 1)
    print("MSE:",mse)
    
    print("MSE 計算完成,", round(time.time() - startTime, 4), "sec")
    
    #%% 輸出
    ###### 後處理 - shape 復原
    output = output.reshape(len(imgNameList), rows, cols, dims) #.astype("uint8")
    if os.path.exists(outputFolder):
        for idx, img in enumerate(output):
#            cv2.imwrite(outputFolder + str(idx+1) + imgFilenameExtension, img)
            cv2.imwrite(outputFolder + imgNameList[idx], img)
    print("影像輸出完成,", round(time.time() - startTime, 4), "sec")
    
    #%% 算 MSE - shape 復原後
    UimgArr = imgArr.reshape(len(imgNameList), rows * cols * dims).astype("uint8")
    Uoutput = output.reshape(len(imgNameList), rows * cols * dims).astype("uint8")
    Umse = ( np.square(UimgArr - Uoutput)).mean() #(axis = 1)
    print("MSE:",Umse)