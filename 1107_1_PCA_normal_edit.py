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
    def __init__(self, data, compressionRate):
        self.data = data.copy().astype('float32')
        self.compressionRate = compressionRate
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
        npyName_val = "Cal_Eigen_val.npy"
        npyName_vec = "Cal_Eigen_vec.npy"
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
    
    def FLOW(self):
        # step 1
        self.y_mean = self.EvaluateMean(self.data)
        # step 2
        self.data = self.TranslateData(self.data, self.y_mean)
        # step 3: Evaluate the covariance matrix M
        self.covMatrix = self.EvaluateCovarianceMatrix(self.data)
        # step 4: Find eigenvalues and corresponding eigenvectors of M
        eigenvalues, eigenvectors = self.Cal_Eigen(self.covMatrix) #c, v
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
        self.p = i
        # step 6: Project each of the vectors in {x1, x2, …, xk} on the space spanned by the first p eigenvectors
        self.D = np.dot(eigenvectors[:, :self.p].T, self.data.T)
        
        #轉存
        self.eigenvalues, self.eigenvectors = eigenvalues, eigenvectors
        return self.D
    
    def Recover(self, D):
        # step 7: Recover approximate face image for each yi:
        return np.dot(D.T, self.eigenvectors[:, :self.p].T) + self.y_mean
#%% 
if __name__ == '__main__' :
    import time
    startTime = time.time()
    #%% 共同參數
    pokerFolder = "./pca_poker_data/"
    outputFolder = "./outputFolder/"
    
    #%% 讀檔設置
    imgNameList = np.array(os.listdir(pokerFolder))
    imgNameList = imgNameList[np.argsort([int(na.split(".")[0]) for na in imgNameList ], axis = 0)]
    tmp = cv2.imread(pokerFolder + imgNameList[0], 0)
    rows, cols = tmp.shape[:2]
    
    imgArr = np.zeros((len(imgNameList), rows, cols))
    for i, name in enumerate(imgNameList):
        imgArr[i, :, :] = cv2.imread(pokerFolder + name, 0).copy()
    imgArr = imgArr.reshape(len(imgNameList), rows * cols) / 255
    
    #%% 計算
    pca = PCA(imgArr, 0.9)
    D = pca.FLOW()
#    eigenvalues, eigenvectors = pca.eigenvalues, pca.eigenvectors
#    p = pca.p
    output = pca.Recover(D)
    print("PCA 計算完成,", round(time.time() - startTime, 4), "sec")
    
    #%% 算 MSE
    output = np.clip(output, 0, 1)
    imgArr = imgArr * 255
    output = output * 255
    mse = ( np.square(imgArr - output)).mean() #(axis = 1)
    print("MSE:",mse)
    
    print("MSE 計算完成,", round(time.time() - startTime, 4), "sec")
    
    #%% 輸出
    output = output.reshape(len(imgNameList), rows, cols) #.astype("int")
    if os.path.exists(outputFolder):
        for idx, img in enumerate(output):
            cv2.imwrite(outputFolder + str(idx+1) + ".bmp", img)
    print("影像輸出完成,", round(time.time() - startTime, 4), "sec")
    
    #%% 算 MSE
    UimgArr = imgArr.reshape(len(imgNameList), rows * cols).astype("uint8")
    Uoutput = output.reshape(len(imgNameList), rows * cols).astype("uint8")
    Umse = ( np.square(UimgArr - Uoutput)).mean() #(axis = 1)
    print("MSE:",Umse)