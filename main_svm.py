# %% initialization
from matplotlib.pyplot import axis
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import median
from numpy.lib.npyio import load
from Data.Dataset.fnirs import fNIRS_mb_label_balance_leave_subject
import pandas as pd
from Experiments.Config.issue03 import *
from Tools.logger import *
import os
import json
import scipy.stats as stats
from skfeature.function.similarity_based import fisher_score as Fs
from Tools.utils import ensure
from tqdm import trange
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from Tools.metric import Performance_Test_ensemble_multi



args = EXP01('debug', 'standard.log')
args.logpath = os.path.join(args.log_root, args.name, args.logfile)
setup_global_logger(args.mode, logging.INFO, logpath=args.logpath)
save_root = './Files/svmfisherZscoreVPL/'
pca_save_root = './Files/svmPCA/'
ensure(save_root)

# %% features inside time window 
def mean(data):
    """
    @parames:
        - data: ndarray with shape [N,C,Channels]; contining oxy or deoxy only. 
                N is the time. calculate mean over time. 
    @outputs:
        - out: shape [C,Channels]
    """
    m = np.mean(data, axis=0, keepdims=False)
    return m

def var(data):
    v = np.var(data, axis=0, keepdims=False)
    return v

def skew(data):
    sk = stats.skew(data, axis=0)
    return sk

def kurtosis(data):
    k = stats.kurtosis(data, axis=0)
    return k

def slope(data):
    dy = data[-1,:,:] - data[0,:,:]
    dx = len(data)
    slop = dy/dx
    return slop

def peak(data):
    p = np.amax(data, axis=0)
    return p 

def feature_selection(dataset, featFunc, topk=6):
    feat_pool = []
    label_pool = []
    selected_feat = []
    for i in trange(len(dataset)):
        data, label, oxy_file = dataset[i]
        feat = featFunc(data)
        feat_pool.append(feat)
        label_pool.append(label)
    feat_pool = np.stack(feat_pool, axis=1) # [C, N, Channels], where C is the oxy and deoxy stuff
    # for i in trange(2):
    #     feat_ = feat_pool[i,:,:]
    #     fisher_score = Fs.fisher_score(feat_, label_pool)
    #     rank = Fs.feature_ranking(fisher_score)
    #     selected_feat.append(rank[:topk])
    return selected_feat, feat_pool, label_pool

def kfoldFeatSave(args, k, featFunc, mode='preprocess'):
    Basic_Name = args.name
    feat_order = ['Oxy', 'Deoxy']
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping = json.load(jsf)
    for fold_id in range(k):
        args.data_config['train_ids'] = fold_id_mapping[str(fold_id)]['train_ids']
        args.data_config['eval_ids'] = fold_id_mapping[str(fold_id)]['eval_ids']
        args.name = "{}_{:02}".format(Basic_Name, fold_id)
        print(f"Runing {args.name} | eval on {args.data_config['eval_ids']}")

        train_dataset = fNIRS_mb_label_balance_leave_subject(\
                    list_root = args.list_path,
                    steps = args.steps_sizes,  
                    mode='train',
                    data_config=args.data_config,
                    runtime=True,
                    fold_id=fold_id)
        test_dataset = fNIRS_mb_label_balance_leave_subject(\
                list_root = args.list_path,
                steps = args.steps_sizes,
                mode='eval',
                data_config=args.data_config,
                runtime=True,
                fold_id=fold_id)

        print(f"Train : {len(train_dataset)}")
        print(f"Test : {len(test_dataset)}")
        saveKroot = os.path.join(save_root, '{:02}'.format(fold_id))
        ensure(saveKroot)
        # fswriter = open(os.path.join(saveKroot, f'featSelection_{featFunc.__name__}.txt'), 'w')
        # fs, feat_pool, label_pool = feature_selection(train_dataset, featFunc)
        fs, feat_pool, label_pool = feature_selection(test_dataset, featFunc)

        np.save(os.path.join(saveKroot, f'test_featPool_{featFunc.__name__}.npy'), feat_pool)
        np.save(os.path.join(saveKroot, f'test_labelPool_{featFunc.__name__}.npy'), label_pool)
        # for i in range(2):
        #     fswriter.write(f'Selected {feat_order[i]}=='+','.join([str(s) for s in fs[i].tolist()])+'\n')
        # fswriter.close()

def load_data(root, fold_id, featFunc):
    feat_path = os.path.join(root, '{:02}'.format(fold_id), f'featPool_{featFunc.__name__}.npy')
    label_path = os.path.join(root, '{:02}'.format(fold_id), f'labelPool_{featFunc.__name__}.npy')
    fs_path = os.path.join(root, '{:02}'.format(fold_id), f'featSelection_{featFunc.__name__}.txt')
    featPool = np.load(feat_path)
    labelPool = np.load(label_path)
    with open(fs_path, 'r') as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]
        oxy_index = lines[0].split('==')[-1].split(',')
        deoxy_index = lines[1].split('==')[-1].split(',')
        oxy_index = [int(v) for v in oxy_index]
        deoxy_index = [int(v) for v in deoxy_index]
    

    return featPool, labelPool, oxy_index, deoxy_index

def load_allfeatures(root, fold_id, topk=1):
    all_features = []
    for func in ['kurtosis', 'mean', 'skew', 'slope', 'var']:
        feat_path = os.path.join(root, '{:02}'.format(fold_id), f'featPool_{func}.npy')
        label_path = os.path.join(root, '{:02}'.format(fold_id), f'labelPool_{func}.npy')
        fs_path = os.path.join(root, '{:02}'.format(fold_id), f'featSelection_{func}.txt')
        featPool = np.load(feat_path)
        labelPool = np.load(label_path)
        with open(fs_path, 'r') as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
            oxy_index = lines[0].split('==')[-1].split(',')
            deoxy_index = lines[1].split('==')[-1].split(',')
            oxy_index = [int(v) for v in oxy_index][:topk]
            deoxy_index = [int(v) for v in deoxy_index][:topk]
            temp_feat = np.stack([featPool[0][:, oxy_index], featPool[1][:, deoxy_index]], 0)
        all_features.append(temp_feat)

    all_features = np.concatenate(all_features, axis=2)  # (2, 5, N, 52), 5 for all five types of features. 
    return all_features, labelPool, oxy_index, deoxy_index 
        

class svmEngine():
    def __init__(self, k, funcList, args) -> None:
        self.k = k
        self.funcList = funcList
        self.args = args

    def extractFeat(self):
        for func in self.funcList:
            kfoldFeatSave(self.args, k=self.k, featFunc=func)

    def runTrain(self):
        args = self.args
        BasicName = args.name
        with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
            fold_id_mapping = json.load(jsf)
        for func in [mean, var, skew, kurtosis, slope]:
            for fold_id in range(self.k):
                ensure(os.path.join(save_root, '{:02}'.format(fold_id)))
                print("="*10)
                args.data_config['train_ids'] = fold_id_mapping[str(fold_id)]['train_ids']
                args.data_config['eval_ids'] = fold_id_mapping[str(fold_id)]['eval_ids']
                args.name = "{}_{:02}".format(BasicName, fold_id)
                print(f"Runing {args.name} | eval on {args.data_config['eval_ids']}")
                featPool, labelPool, oxy_index, deoxy_index = load_data(save_root, fold_id, func)
                # featPool = stats.zscore(featPool, axis=1) # normalize the features. 
                clf = trainSVM(featPool, labelPool, oxy_index, deoxy_index)
                testFeat, testLabel = extractTestFeature(args, fold_id, oxy_index, deoxy_index, func)
                preds = clf.predict(testFeat)
                with open(os.path.join(save_root, '{:02}'.format(fold_id), f'Results_{func.__name__}.txt'), 'w') as resWriter:
                    for i in range(len(preds)):
                        msg = f"{preds[i]},{testLabel[i]}\n"
                        resWriter.write(msg)                    
                # df = pd.DataFrame(clf.cv_results_)
                # df.to_csv(os.path.join(save_root, '{:02}'.format(fold_id), f'cv_results_{func.__name__}.csv'))
                # return
                print("Results saved.")

    def runAllFeatures(self, topk=1):
        args = self.args
        BasicName = args.name
        with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
            fold_id_mapping = json.load(jsf)
        for fold_id in range(self.k):
            ensure(os.path.join(save_root, '{:02}'.format(fold_id)))
            print("="*10)
            args.data_config['train_ids'] = fold_id_mapping[str(fold_id)]['train_ids']
            args.data_config['eval_ids'] = fold_id_mapping[str(fold_id)]['eval_ids']
            args.name = "{}_{:02}".format(BasicName, fold_id)
            print(f"Runing {args.name} | eval on {args.data_config['eval_ids']}")
            featPool, labelPool, _, _ = load_allfeatures(save_root, fold_id, topk)
            clf = trainSVM_allfeatures(featPool, labelPool)
            testFeat, testLabel = extractTestAllFeature(args, save_root, fold_id, topk)
            preds = clf.predict(testFeat)
            with open(os.path.join(save_root, '{:02}'.format(fold_id), f'Results_AllFeatures_topk{topk}.txt'), 'w') as resWriter:
                for i in range(len(preds)):
                    msg = f"{preds[i]},{testLabel[i]}\n"
                    resWriter.write(msg)    
            print("Resutls saved.")


def trainSVM_GS(X, y, oxy_index, deoxy_index):
    """
        svm using grid search. 
    """
    oxy_feat = X[0][:,oxy_index]
    deoxy_feat = X[1][:,deoxy_index]
    selected_feature = np.concatenate((oxy_feat, deoxy_feat), axis=1)
    para_dict = {
        'C':[1, 10, 100, 1000]
    }
    svm = SVC(kernel='linear')
    clf = GridSearchCV(svm, param_grid=para_dict)
    clf.fit(selected_feature, y)
    return clf

def trainSVM(X, y, oxy_index, deoxy_index):
    oxy_feat = X[0][:,oxy_index]
    deoxy_feat = X[1][:,deoxy_index]
    selected_feature = np.concatenate((oxy_feat, deoxy_feat), axis=1)
    # para_dict = {
    #     'C':[1, 10, 100, 1000]
    # }
    clf = SVC(kernel='linear')
    # clf = GridSearchCV(svm, param_grid=para_dict)
    clf.fit(selected_feature, y)
    return clf
    
def trainSVM_allfeatures(X, y):
    oxy_feat = X[0]
    deoxy_feat = X[1]
    selected_feature = np.concatenate((oxy_feat, deoxy_feat), axis=1)
    clf = SVC(kernel='linear')
    clf.fit(selected_feature, y)
    return clf

def extractTestFeature(args, fold_id, oxy_index, deoxy_index, featFunc):
    test_dataset = fNIRS_mb_label_balance_leave_subject(\
                list_root = args.list_path,
                steps = args.steps_sizes,
                mode='eval',
                data_config=args.data_config,
                runtime=True,
                fold_id=fold_id)
    featPool = []
    labelPool = []
    for i in range(len(test_dataset)):
        data, label, oxy_files =test_dataset[i]
        feat = featFunc(data)
        featPool.append(feat)
        labelPool.append(label)
    featPool = np.stack(featPool, axis=1)

    # pick feat
    oxy_feat = featPool[0][:,oxy_index]
    deoxy_feat = featPool[1][:,deoxy_index]
    feat = np.concatenate((oxy_feat, deoxy_feat), axis=1)
    return feat, labelPool

def extractTestAllFeature(args, root, fold_id, topk=1):
    test_dataset = fNIRS_mb_label_balance_leave_subject(\
                list_root = args.list_path,
                steps = args.steps_sizes,
                mode='eval',
                data_config=args.data_config,
                runtime=True,
                fold_id=fold_id)
    all_features = []

    for func in ['kurtosis', 'mean', 'skew', 'slope', 'var']:
        featFunc = eval(func)
        featPool = []
        labelPool = []
        for i in range(len(test_dataset)):
            data, label, oxy_files =test_dataset[i]
            feat = featFunc(data)
            featPool.append(feat)
            labelPool.append(label)
        featPool = np.stack(featPool, axis=1)
        fs_path = os.path.join(root, '{:02}'.format(fold_id), f'featSelection_{func}.txt')
        with open(fs_path, 'r') as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
            oxy_index = lines[0].split('==')[-1].split(',')
            deoxy_index = lines[1].split('==')[-1].split(',')
            oxy_index = [int(v) for v in oxy_index][:topk]
            deoxy_index = [int(v) for v in deoxy_index][:topk]
            temp_feat = np.stack([featPool[0][:, oxy_index], featPool[1][:, deoxy_index]], 0)
        all_features.append(temp_feat)
    all_features = np.concatenate(all_features, axis=2)
    all_features = np.concatenate((all_features[0], all_features[1]), axis=1)
    return all_features, labelPool


def fisherSelectedSVMMetrics(k, funcList):
    info("fisher score selected SVM")
    for func in funcList:
        esemble_metric = Performance_Test_ensemble_multi(joint=True, self_supervise=False)
        for fold_id in range(k):
            results_root = os.path.join(save_root, '{:02}'.format(fold_id))
            result_path = os.path.join(results_root, f'Results_{func.__name__}.txt')
            with open(result_path, 'r') as f:
                lines = f.readlines()
                lines = [l.rstrip() for l in lines]
                for l in lines:
                    p, t = l.split(',')
                    esemble_metric(int(p), int(t), svm=True)
        print("="*10)
        print(f"Selected Feature: {func.__name__} | esemble results")
        performance = esemble_metric.value(singular=True)
        # for i in ['wml', 'vpl']:
        #     conf_mat, accu, precision, recall, f1 = performance[i]
        #     info("ACCURACY_{} = {} ".format(i, accu))
        #     info("Precision_{} = {} ".format(i, precision))
        #     info("RECALL_{} = {} ".format(i, recall))
        #     info("F1_{} = {} ".format(i, f1))
        conf_mat, accu, precision, recall, f1 = performance
        i = save_root.split('/')[-2]
        info("ACCURACY_{} = {} ".format(i, accu))
        info("Precision_{} = {} ".format(i, precision))
        info("RECALL_{} = {} ".format(i, recall))
        info("F1_{} = {} ".format(i, f1))

def fisherSelectedSVMMetrics_byFolds(k, funcList):
    info("fisher score selected SVM")
    search_records = open('./Files/cnn/SVM_mean_10FoldsCV_Metric_VPL_new.txt', 'w')
    for func in funcList:
        for fold_id in range(k):
            results_root = os.path.join(save_root, '{:02}'.format(fold_id))
            result_path = os.path.join(results_root, f'Results_{func.__name__}.txt')
            esemble_metric = Performance_Test_ensemble_multi(joint=True, self_supervise=False)
            with open(result_path, 'r') as f:
                lines = f.readlines()
                lines = [l.rstrip() for l in lines]
                for l in lines:
                    p, t = l.split(',')
                    esemble_metric(int(p), int(t), svm=True)
            # print("="*10)
            print(f"Selected Feature: {func.__name__} | esemble results")
            performance = esemble_metric.value(singular=True)
        # for i in ['wml', 'vpl']:
        #     conf_mat, accu, precision, recall, f1 = performance[i]
        #     info("ACCURACY_{} = {} ".format(i, accu))
        #     info("Precision_{} = {} ".format(i, precision))
        #     info("RECALL_{} = {} ".format(i, recall))
        #     info("F1_{} = {} ".format(i, f1))
            conf_mat, accu, precision, recall, f1 = performance
            fL = save_root.split('/')[-2]
            search_records.write("+"*20+f"Fold id {fold_id}"+'\n')
            search_records.write("ACCURACY_{} = {}\n".format(fL, accu))
            search_records.write("Precision{} = {}\n".format(fL, precision))
            search_records.write("Recall_{} = {}\n".format(fL, recall))
            search_records.write("F1_{} = {}\n".format(fL, f1))

def fisherSelectedSVMMetrics_byFolds_AllFeatures(k, save_root, record, topk=1): 
    info(f'fisher score selected SVM all Features.')
    search_records = open(record, 'w')
    ACCU = 0
    PREC = 0
    RECA = 0
    F1 = 0
    for fold_id in range(k):
        results_root = os.path.join(save_root, '{:02}'.format(fold_id))
        result_path = os.path.join(results_root, f'Results_AllFeatures_topk{topk}.txt')
        esemble_metric = Performance_Test_ensemble_multi(joint=True, self_supervise=False)
        with open(result_path, 'r') as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
            for l in lines:
                p, t = l.split(',')
                esemble_metric(int(p), int(t), svm=True)
        # print("Selected All Features | esemble results" )
        performance = esemble_metric.value(singular=True)

        conf_mat, accu, precision, recall , f1 = performance
        fL = save_root.split('/')[-2]
        search_records.write("+"*20+f"Fold id {fold_id}"+'\n')
        search_records.write("ACCURACY_{} = {}\n".format(fL, accu))
        search_records.write("Precision{} = {}\n".format(fL, precision))
        search_records.write("Recall_{} = {}\n".format(fL, recall))
        search_records.write("F1_{} = {}\n".format(fL, f1))

        ACCU += accu
        PREC += precision
        RECA += recall
        F1 += f1
    print(f" >>>> {record} >> MEAN OVER 10 FOLDS")
    print(f"ACCURACY -- {ACCU/k}")
    print(f"PRECISION -- {sum(PREC/k)/3}")
    print(f"RECA -- {sum(RECA/k)/3}")
    print(f"F1 --- {sum(F1/k)/3}")

    search_records.close()    

if __name__ == "__main__":
    # fold_id = 0
    # for func in [mean, var, skew, kurtosis, slope]:
    # #     kfoldFeatSave(args, k=10, featFunc=func)
    #     featPool, labelPool, oxy_index, deoxy_index = load_data(save_root, '00', slope)
    #     # clf = trainSVM(featPool, labelPool, oxy_index, deoxy_index)
    #     feat, labelPool = extractTestFeature(args, fold_id, oxy_index, deoxy_index, slope)
    #     # preds = clf.predict(feat)
    #     print("-")
# +++++++++++++++++++++++++++++++++++++++++++++++
#  Debug 
# +++++++++++++++++++++++++++++++++++++++++++++++

    # data_mean = load_data('./Files/svmfisher/', 0, mean)
    # data_var = load_data('./Files/svmfisher/', 0, var)
    # data = load_allfeatures('./Files/svmfisher/', 0, 1)

    # print(np.sum(data_mean[1] == data_var[1]))
    
# +++++++++++++++++++++++++++++++++++++++++++++
#        The Engine Entry
# +++++++++++++++++++++++++++++++++++++++++++++
    
    # engine = svmEngine(k=10, funcList=[mean, var, skew, kurtosis, slope], args=args)
    # # engine = svmEngine(k=10, funcList=[mean, var, skew, kurtosis, slope, peak], args=args)
    # engine.extractFeat()
    # engine.runTrain()
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++
#       Test SVM based on the predictions records
# +++++++++++++++++++++++++++++++++++++++++++++++++++
    # fisherSelectedSVMMetrics_byFolds(k=10, funcList=[mean])
# +++++++++++++++++++++++++++++++++++++++++++++
#        Run Using All Features
# +++++++++++++++++++++++++++++++++++++++++++++
    # print("RUNNING ALL FEATURES TEST")
    # save_root = './Files/svmfisherZscoreVPL/'
    # engine = svmEngine(k=10, funcList=[mean, var, skew, kurtosis, slope], args=args)
    # engine.runAllFeatures(topk=1)
    # engine.runAllFeatures(topk=2)
    # save_root = './Files/svmfisherZscoreWML/'
    # engine = svmEngine(k=10, funcList=[mean, var, skew, kurtosis, slope], args=args)
    # engine.runAllFeatures(topk=1)
    # engine.runAllFeatures(topk=2)

    # TODO: THE METRICS
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++
#       Test SVM based on the predictions records, all features.
# +++++++++++++++++++++++++++++++++++++++++++++++++++
    topk=1
    target = 'VPL'
    fisherSelectedSVMMetrics_byFolds_AllFeatures(k=10, \
                                                save_root=f'./Files/svmfisherZscore{target}/',\
                                                record=f'./Files/cnn/SVM_mean_10Folds_metric_{target}_all_features_topk{topk}.txt',\
                                                topk=topk)
    # topk=2
    # target = 'VPL'
    # fisherSelectedSVMMetrics_byFolds_AllFeatures(k=10, \
    #                                             save_root=f'./Files/svmfisherZscore{target}/',\
    #                                             record=f'./Files/cnn/SVM_mean_10Folds_metric_{target}_all_features_topk{topk}.txt',\
    #                                             topk=topk)

    topk=1
    target = 'WML'
    fisherSelectedSVMMetrics_byFolds_AllFeatures(k=10, \
                                                save_root=f'./Files/svmfisherZscore{target}/',\
                                                record=f'./Files/cnn/SVM_mean_10Folds_metric_{target}_all_features_topk{topk}.txt',\
                                                topk=topk)
    # topk=2
    # target = 'WML'
    # fisherSelectedSVMMetrics_byFolds_AllFeatures(k=10, \
    #                                             save_root=f'./Files/svmfisherZscore{target}/',\
    #                                             record=f'./Files/cnn/SVM_mean_10Folds_metric_{target}_all_features_topk{topk}.txt',\
    #                                             topk=topk)