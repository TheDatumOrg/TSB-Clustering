import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from .framework import DTCR
from .config import config_dtcr
from .utils import read_dataset, construct_classification_dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'




NORMALIZED = False 
#os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def dtcr_clustering(x, y, n_clusters, params, best=True):
    y = y + 10
    y = -1 * y
    
    label_dict = None
    if label_dict is None:
        label_dict = {}
        label_list = np.unique(y)
        for idx in range(len(label_list)):
            label_dict[str(label_list[idx])] = idx

    o_label = list(label_dict.keys())
    for l in o_label:
        y[y == float(l)] = label_dict[l]
        
    y = y.astype(int)

    cls_x, cls_y = construct_classification_dataset(x)
    
    config_dtcr['input_length'] = x.shape[1]
    if cls_x.shape[0] // 200 == 0:
        config_dtcr['training_samples_num'] = x.shape[0]
    else:
        config_dtcr['training_samples_num'] = 100 #x.shape[0]
    config_dtcr['cluster_num'] = len(np.unique(y))
    config_dtcr['encoder_hidden_units'] = params[0]
    config_dtcr['lambda'] = params[1]
    config_dtcr['dilations'] = params[2]

    DTCR_model = DTCR(config_dtcr)
    best_inertia, best_predictions, inertia, predictions = DTCR_model.train(cls_x, cls_y, x, y)

    if best:
        o_label = list(label_dict.values())
        for i, l in enumerate(o_label):
            best_predictions[best_predictions == float(l)] = list(label_dict.keys())[i]
        
        best_predictions = (best_predictions.astype(int) * -1) + 10
        best_predictions = best_predictions.astype(int)
        return best_inertia, best_predictions
    else:
        o_label = list(label_dict.values())
        for i, l in enumerate(o_label):
            predictions[predictions == float(l)] = list(label_dict.keys())[i]
        
        predictions = (predictions.astype(int) * -1) + 10
        predictions = predictions.astype(int)
        return inertia, predictions

    

'''
log_folder = 'train_log'
if os.path.exists(log_folder) == False: os.makedirs(log_folder)
img_folder = 'train_img'
if os.path.exists(img_folder) == False: os.makedirs(img_folder)

config_dtcr['indicator'] = 'RI'


if __name__ == '__main__':
    
    dataset_name = 'ChlorineConcentration' # any sub-dataset in UCRArchive_2018
    
    # dataset setting 
    config_dtcr['train_file'] = './../UCRArchive_2018/{0}/{0}_TRAIN.tsv'.format(dataset_name) # re-config the path
    config_dtcr['test_file'] = './../UCRArchive_2018/{0}/{0}_TEST.tsv'.format(dataset_name) # re-config the path
    
    config_dtcr['img_path'] = os.path.join(img_folder, dataset_name)
    if os.path.exists(config_dtcr['img_path']) == False: os.makedirs(config_dtcr['img_path'])
    
    # load data  
    real_train_data, real_train_label, label_dict = read_dataset(config_dtcr, 'train', if_n=NORMALIZED)
    real_test_data, real_test_label, _ = read_dataset(config_dtcr, 'test', label_dict, if_n=NORMALIZED)
    
    # construct classification dataset
    cls_train_data, cls_train_label = construct_classification_dataset(real_train_data)
    
    # dataset setting
    config_dtcr['input_length'] = real_train_data.shape[1]
    config_dtcr['training_samples_num'] = real_train_data.shape[0]
    config_dtcr['cluster_num'] = len(np.unique(real_train_label))

    # Network config searching
    for encoder_config in [[100,50,50], [50,30,30]]:#[100,50,50], [50,30,30]
        config_dtcr['encoder_hidden_units'] = encoder_config

        for lambda_1 in [1, 1e-1, 1e-2, 1e-3]:#1, 1e-1, 1e-2, 1e-3
                
            config_dtcr['lambda'] = lambda_1
            
            # init the model
            print('Init the DTCR model')
            DTCR_model = DTCR(config_dtcr)
            # train the model
            print('Start training...')
            best, best_epoch, train_list, test_list = DTCR_model.train(cls_train_data, cls_train_label, real_train_data, real_train_label, real_test_data, real_test_label)
            
            #log
            log_file = os.path.join(log_folder, '{}_log.txt'.format(dataset_name))
            if os.path.exists(log_file) == False:
                f = open(log_file, 'w')
                f.close()
            f = open(log_file, 'a')
            print('dataset: {}\trun_index: {}'.format(dataset_name, INDEX), file=f)
            print('network config:\nencoder_hidden_units = {}, lambda = {}, indicator = {}, normalized = {}'.
                format(config_dtcr['encoder_hidden_units'], config_dtcr['lambda'], config_dtcr['indicator'], NORMALIZED), file=f)
            print('best\t{} = {}, epoch = {}\n\n'.format(config_dtcr['indicator'], best, best_epoch), file=f)
            f.close()
'''
    
    
