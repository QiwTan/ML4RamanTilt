# all the EDFA machine learning based libraries
 ###############################################
# all the external libs for EDFA postprocessing 
###############################################

# data
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import scipy.stats as stats
import json,copy
from collections import defaultdict
import statistics

# MISC
import math,os,shutil,fnmatch
import datetime
import matplotlib.pyplot as plt

# ML
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import math as TFmath


#####################################################################
####################  ML help function
#####################################################################
### helper labels - fixed
Numchannels = 90
num_inputFeatures = Numchannels * 2 + 3
labels = {"inSpectra":'Fibre_input_spectra_',
            "WSS":'Activated_channel_index_',
            "result":'Raman_tilt',
            "linear_loss":"linear_loss",
            "inPower":"Total_fiber_in_power",
            "outPower":"Total_fiber_out_power"}
inSpectra_labels = [labels['inSpectra']+str(i).zfill(2) for i in range(Numchannels)]
inSpectra_labels.extend([labels['linear_loss'],labels['inPower'],labels['outPower']])
onehot_labels = [labels['WSS']+str(i).zfill(2) for i in range(Numchannels)]
# drop_labels = inSpectra_labels
result_labels = [labels['outPower']]#[labels['result']+str(i).zfill(2) for i in range(Numchannels)]

### helper function for preTrain
def logsumexp10(x):
    return 10*np.log10(np.sum(np.power(10, np.array(x)/10)))

def dB_to_linear(data):
  return np.power(10,data/10)

def linear_TO_Db(data):
  result = 10*np.log10(data).to_numpy()
  return result[result != -np.inf]

def linear_TO_Db_full(data):
  result = 10*np.log10(data).to_numpy()
  result[result == -np.inf] = 0
  return result

def divideZero(numerator,denominator):
  with np.errstate(divide='ignore'):
    result = numerator / denominator
    result[denominator == 0] = 0
  return result

def getCSVData(fileName):
  
  # read csv file
  trainData = pd.read_csv(fileName,index_col=0)
  
  # convert dB to linear 
#   trainData[preProcess_labels] = trainData[preProcess_labels].applymap(dB_to_linear)
  
  X_train = trainData.copy()
  y_train = pd.concat([X_train.pop(x) for x in result_labels], axis=1)
#   [X_train.pop(x) for x in result_ignore]

  # change unloaded to 0
  # onehot_train = X_train[onehot_labels]
  # y_train = pd.DataFrame(y_train.values*onehot_train.values, columns=y_train.columns, index=y_train.index)

  return X_train,y_train

### train and loss function
def custom_loss(y_actual,y_pred):
  # calculate the loaded channel numbers for each batch
  # batch default is [batch size=32, outputchannel number]
  loaded_size = tf.dtypes.cast(TFmath.count_nonzero(y_actual), tf.float32)
  # turn unloaded y_pred prediction to zero
  y_pred_cast_unloaded_to_zero = TFmath.divide_no_nan(TFmath.multiply(y_pred,y_actual),y_actual)
  # error [unloaded,unloaded,loaded,loaded]: y_pred = [13->0,15->0,18.5,18.3], y_actual = [0,0,18.2,18.2]
  error = TFmath.abs(TFmath.subtract(y_pred_cast_unloaded_to_zero,y_actual))
  # custom_loss = (0.3+0.2) / 2
  custom_loss = TFmath.divide(TFmath.reduce_sum(error),loaded_size)
  return custom_loss

def custom_loss_L2(y_actual,y_pred):
  loaded_size = tf.dtypes.cast(TFmath.count_nonzero(y_actual), tf.float32)
  y_pred_cast_unloaded_to_zero = TFmath.divide_no_nan(TFmath.multiply(y_pred,y_actual),y_actual)
  error = TFmath.square(TFmath.subtract(y_pred_cast_unloaded_to_zero,y_actual))
  custom_loss = TFmath.sqrt(TFmath.divide(TFmath.reduce_sum(error),loaded_size))
  return custom_loss
    


##############################################################
# use the DNN model
def input_spectra_with_PDPower_to_DNN(inputSpectra,inputChannelLoading,inputPower,outputPower,targetPower=18):
    # convert input spectra into dataset into DNN model
    returnFeature = pd.DataFrame()
    metaResult={}
    metaResult['target_gain'] = targetPower
    metaResult['EDFA_input_power_total'] = dB_to_linear(inputPower)
    metaResult['EDFA_output_power_total'] = dB_to_linear(outputPower)
    # change [1,2,3,4...] to [1,1,1,1,0,0,1,...]
    channelloadings = [1 if id in inputChannelLoading else 0 for id in range(Numchannels)]
    inputSpectra = [inputSpectra[id] if channelloadings[id]==1 else -50 for id in range(Numchannels)]
    for indx in range(Numchannels):
        post_indx = str(indx)
        metaResult['EDFA_input_spectra_'+post_indx] = dB_to_linear(inputSpectra[indx])
        metaResult['DUT_WSS_activated_channel_index_'+post_indx] = channelloadings[indx]
    # return returnFeature.append([metaResult],ignore_index=True)
    return pd.concat([returnFeature.reset_index(drop=True), pd.DataFrame.from_dict([metaResult]).reset_index(drop=True)])

def DNN_predict_single_spectra_with_PDPower(inputSpectra,inputChannelLoading,inputPower,outputPower,TrainModelPath):

    ExtractedFeature = input_spectra_with_PDPower_to_DNN(inputSpectra,inputChannelLoading,inputPower,outputPower)
    # load model, only once since all the base model are same....
    base_model = designed_DNN_model(Numchannels)
    base_model.compile(loss=custom_loss_L2, optimizer=tf.keras.optimizers.Adam(
        0.01))  # avoid warnings
    base_model.load_weights(TrainModelPath)  # load the model
    # convert linear to dB
    y_pred = base_model.predict(ExtractedFeature)
    # back to dB
    y_pred_result = 10*np.log10(y_pred[0])

    return y_pred_result + inputSpectra
    
##########################################################################
def input_spectra_to_DNN(inputSpectra,inputChannelLoading,targetPower=18):
    # convert input spectra into dataset into DNN model
    returnFeature = pd.DataFrame()
    metaResult={}
    metaResult['target_gain'] = targetPower
    # change [1,2,3,4...] to [1,1,1,1,0,0,1,...]
    channelloadings = [1 if id in inputChannelLoading else 0 for id in range(Numchannels)]
    inputSpectra = [inputSpectra[id] if channelloadings[id]==1 else -50 for id in range(Numchannels)]
    for indx in range(Numchannels):
        post_indx = str(indx)
        metaResult['EDFA_input_spectra_'+post_indx] = dB_to_linear(inputSpectra[indx])
        metaResult['DUT_WSS_activated_channel_index_'+post_indx] = channelloadings[indx]
    # return returnFeature.append([metaResult],ignore_index=True)
    return pd.concat([returnFeature.reset_index(drop=True), pd.DataFrame.from_dict([metaResult]).reset_index(drop=True)])

def DNN_predict_single_spectra(inputSpectra,inputChannelLoading,TrainModelPath):

    ExtractedFeature = input_spectra_to_DNN(inputSpectra,inputChannelLoading)
    # load model, only once since all the base model are same....
    base_model = designed_DNN_model(Numchannels)
    base_model.compile(loss=custom_loss_L2, optimizer=tf.keras.optimizers.Adam(
        0.01))  # avoid warnings
    base_model.load_weights(TrainModelPath)  # load the model
    # convert linear to dB
    y_pred = base_model.predict(ExtractedFeature)
    # back to dB
    y_pred_result = 10*np.log10(y_pred[0])

    return y_pred_result + inputSpectra

#################################################################
# CM model and usage

class CM_Model():
    def __init__(self):
        self.model_paras = {}

    def train(self, csvFile):
        with open(csvFile, "r") as read_file:
            self.model_paras = json.load(read_file)

    def predict(self, channelLoaidng, csvFile):
        # indx from 0
        # input list, return gain
        self.train(csvFile)
        num = len(channelLoaidng)
        # print(channelIndxs) # 
        diff_avg = sum([self.model_paras["single"][indx] -
                    self.model_paras["wdm"][indx] for indx in channelLoaidng]) / num
        # x = [self.model_paras["single"][indx] -
        #             self.model_paras["wdm"][indx] for indx in channelLoaidng]
        # x_0 = np.mean(np.power(10, np.array(x)/10))
        # diff_avg = 10*np.log10(x_0)
        gain = [self.model_paras["wdm"][indx] + diff_avg for indx in channelLoaidng]
        
        return gain

def CM_predict(dataIn,channelLoading,cmModelPath):
    cmModel = CM_Model()
    gains =  cmModel.predict(dataIn,channelLoading, cmModelPath)
    dataOut = list(map(float, copy.copy(dataIn)))
    for indx in range(len(channelLoading)):
      dataOut[channelLoading[indx]] = dataIn[channelLoading[indx]] + gains[indx]
    return np.array(dataOut)

#################################################################
# Temp functions


### debug function after train -> go to csv
def plot_loss(indx,history,ingnoreIndex):
  plt.figure(indx)
  plt.plot(history.history['loss'][ingnoreIndex:], label='loss')
  plt.plot(history.history['val_loss'][ingnoreIndex:], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [gain]')
  plt.legend()
  plt.grid(True)

def figure_comp(figure_prepath,figIndx,y_test_result,y_pred_result,filename,setFrontSize):
    plt.figure(figIndx)
    plt.axes(aspect='equal')
    plt.scatter(y_test_result, y_pred_result)
    plt.xlabel('Measured EDFA Gain (dB)', fontsize=setFrontSize)
    plt.ylabel('predicted EDFA Gain (dB)', fontsize=setFrontSize)
    minAxis = math.floor(min(y_test_result.min(),y_pred_result.min()) - 0.5)
    maxAxis = math.ceil (max(y_test_result.max(),y_pred_result.max()) + 0.5)
    # print(min(y_test_result.min(),y_pred_result.min()),max(y_test_result.max(),y_pred_result.max()))
    # print(minAxis,maxAxis)
    limss = [*np.arange(minAxis,maxAxis+1,1)]
    lims = [limss[0],limss[-1]]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims, 'k--')
    plt.xticks(ticks=limss,labels=limss,fontsize=setFrontSize)
    plt.yticks(fontsize=setFrontSize)
    plt.savefig(figure_prepath+filename, dpi=900)

def figure_hist(figure_prepath,figIndx,error,filename,setFrontSize):
    plt.figure(figIndx)
    bins_list = [*np.arange(-0.6,0.7,0.1)]
    labelList = ['-0.6','','-0.4','','-0.2','','0.0','','0.2','','0.4','','0.6']
    plt.hist(error, bins=bins_list)
    for i in [-0.2,-0.1,0.1,0.2]: # helper vertical line
        plt.axvline(x=i,color='black',ls='--')
    plt.xlabel('Prediction Gain Error (dB)', fontsize=setFrontSize)
    plt.ylabel('Histogram', fontsize=setFrontSize)
    plt.xticks(ticks=bins_list, labels=labelList, fontsize=setFrontSize)
    plt.yticks(fontsize=setFrontSize)
    plt.savefig(figure_prepath+filename, dpi=900)

""" ===========================================
TODO: finalize the plot based on discussion
    ===========================================
""" 
def plot_per_channel_error(figure_prepath,figIndx,y_pred,y_test,filename,setFrontSize):
    y_pred_result = linear_TO_Db_full(y_pred)
    y_test_result = linear_TO_Db_full(y_test)
    error = y_test_result - y_pred_result
    error_min_0_1s,error_means,error_min_0_2s,within95ranges,mses,ames,max_ames = [],[],[],[],[],[],[]
    for j in range(len(error[0])):
        error_channel = error[:][j]
        error_channel = error_channel[error_channel!=0]
        # calculate the distribution
        error_reasonable = [i for i in error_channel if abs(i)<=0.2]
        error_measureError = [i for i in error_channel if abs(i)<=0.1]
        error_min_0_1 = len(error_measureError)/len(error_channel)
        error_min_0_2 = len(error_reasonable)/len(error_channel)
        error_sorted = np.sort(abs(error_channel))
        within95range = error_sorted[int(0.95*len(error_channel))]
        ame = (np.abs(error_channel)).mean(axis=None)
        max_ame = (np.abs(error_channel)).max(axis=None)
        mse = (np.square(error_channel)).mean(axis=None)
        error_mean = error_channel.mean(axis=None)
        error_means.append(error_mean)
        error_min_0_1s.append(error_min_0_1)
        error_min_0_2s.append(error_min_0_2)
        within95ranges.append(within95range)
        mses.append(mse)
        ames.append(ame)
        max_ames.append(max_ame)
    # plt.figure(100)
    # plt.plot(error_min_0_1s)
    # plt.figure(101)
    # plt.plot(error_min_0_2s)
    # plt.figure(102)
    # plt.plot(within95ranges)
    # plt.figure(104)
    # plt.plot(error_means)
    plt.figure(figIndx)
    plt.plot(ames)
    plt.plot(max_ames)
    plt.legend(["ame","max error"])
    plt.xlabel('Channel indices', fontsize=setFrontSize)
    plt.ylabel('ASE (dB)', fontsize=setFrontSize)
    plt.title("Per channel ASE", fontsize=setFrontSize)
    plt.xticks(fontsize=setFrontSize)
    plt.yticks(fontsize=setFrontSize)
    plt.savefig(figure_prepath+filename, dpi=900)

def plot_cdf(figure_prepath,figIndx,error,filename,setFrontSize):
    plt.figure(figIndx)
    plt.hist(np.abs(error), density=True, cumulative=True, histtype='step')
    plt.xlabel('Channel indices', fontsize=setFrontSize)
    plt.ylabel('ASE (dB)', fontsize=setFrontSize)
    plt.title("Per channel ASE", fontsize=setFrontSize)
    plt.xticks(fontsize=setFrontSize)
    plt.yticks(fontsize=setFrontSize)
    plt.savefig(figure_prepath+filename, dpi=900)


def getErrorInfo(error):
  error_reasonable = [i for i in error if abs(i)<=0.2]
  error_measureError = [i for i in error if abs(i)<=0.1]
  # error_95 = [i for i in error if abs(i)<=0.25]
  error_min_0_1 = len(error_measureError)/len(error)
  error_min_0_2 = len(error_reasonable)/len(error)
  # error_min_0_25= len(error_95)/len(error)
  # error_max_0_2 = 1-len(error_reasonable)/len(error)
  error_sorted = np.sort(np.abs(error))
  within95range = error_sorted[int(0.95*len(error))]
  ame = (np.abs(error)).mean(axis=None)
  mse = (np.square(error)).mean(axis=None)
  return_ame = "{:.2f}".format(ame)
  return_error_min_0_1 = "{:.2f}".format(error_min_0_1)
  return_error_min_0_2 = "{:.2f}".format(error_min_0_2)
  return_within95range = "{:.2f}".format(within95range)
  return_mse = "{:.2f}".format(mse)
  return return_error_min_0_1,return_error_min_0_2,return_ame,return_within95range,return_mse
