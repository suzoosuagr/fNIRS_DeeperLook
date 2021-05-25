import torch
import numpy as np
import torch.nn.functional as F

class Metric():
    """Base class for all metrics. 
    """
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class Performance_Test_ensemble_multi(Metric):
    def __init__(self, joint=False, self_supervise=False):
        """
        For single label\\
        `size`: int, the size of confusion matrix, the total number of class
        """
        super(Performance_Test_ensemble_multi, self).__init__()
        size = 3
        self.confusion_matrix = {
            'wml': np.zeros((size, size)),
            'vpl': np.zeros((size, size)),
            # 'apl': np.zeros((size, size))
        }
        self.size = size
        self.total = 0.0 + 1e-8
        self.correct = 0.0
        self.joint = joint
        self.locator = lambda x: 0 if x > 0.5 else 1
        self.joint_label = {
            0:[0, 1],  # rt gng
            1:[2, 0],  # anb
            2:[1, 2]  # ewm
        }
        self.self_supervise = self_supervise

    def __call__supervised(self, pred, truth):
        
        pred = pred.cpu()
        truth = truth.cpu()
        pred = F.softmax(pred, dim=1)


        if self.joint:
            for p, t in zip(pred, truth):
                p_ = np.argmax(p)

                self.confusion_matrix['wml'][self.joint_label[int(t)][0]][self.joint_label[int(p_)][0]] += 1.0
                self.confusion_matrix['vpl'][self.joint_label[int(t)][1]][self.joint_label[int(p_)][1]] += 1.0

            pred = torch.argmax(pred, dim=1).long()
            self.correct += torch.sum((pred == truth))
            self.total += len(truth)        

        else:
            pred = torch.argmax(pred, dim=1).long()
            correct_ = torch.sum((pred == truth), dim=1)
            self.correct += torch.sum(correct_ == 2)
            self.total += len(truth)

            for p, t in zip(pred, truth):
                self.confusion_matrix['wml'][t[0]][p[0]] += 1
                self.confusion_matrix['vpl'][t[1]][p[1]] += 1
                # self.confusion_matrix['apl'][t[2]][p[2]] += 1

    def __call__self_supervised(self, pred, truth):
        # task_label = truth[0]
        B = pred[0].size(0)
        pred_0 = F.softmax(pred[0].cpu(), dim=1)
        pred_1 = F.softmax(pred[1].cpu(), dim=1)
        pred_ = pred_0 + pred_1
        pred_ = torch.argmax(pred_, dim=1)

        for i in range(B):
            p = pred_[i]
            # wml_p0 = pred_0[i][0] + pred_1[i][1]
            # vpl_p0 = pred_1[i][2] + pred_1[i][3]

            self.confusion_matrix['wml'][self.joint_label[int(truth[0][i]//2)][0]][self.joint_label[int(p//2)][0]] += 1.0
            self.confusion_matrix['vpl'][self.joint_label[int(truth[0][i]//2)][1]][self.joint_label[int(p//2)][1]] += 1.0

    def __call__self_supervised_mb(self, pred, truth):
        pred_0, pred_1 = pred
        pred_0 = F.softmax(pred_0.cpu(), dim=1)
        pred_1 = F.softmax(pred_1.cpu(), dim=1)
        pred_ = pred_0 + pred_1
        B = pred_0.size(0)
        pred_ = torch.argmax(pred_, dim=1)

        for i in range(B):
            p = pred_[i].squeeze()
            
            p_wml = p[0]
            p_vpl = p[1]
            self.confusion_matrix['wml'][int(truth[0][i][0]//2)][int(p_wml//2)] += 1.0
            self.confusion_matrix['vpl'][int(truth[0][i][1]//2)][int(p_vpl//2)] += 1.0

    def __call__(self, pred, truth):
        if self.self_supervise:
            self.__call__self_supervised_mb(pred, truth)
        else:
            self.__call__supervised(pred, truth)
        


    def reset(self):
        self.confusion_matrix = {
            'wml': np.zeros((self.size, self.size)),
            'vpl': np.zeros((self.size, self.size)),
            # 'apl': np.zeros((self.size, self.size))
        }
        self.total = 0
        self.correct = 0

    def _get_value(self, index):
        confusion_matrix = self.confusion_matrix[index]
        TP = confusion_matrix.diagonal()
        precision = TP / confusion_matrix.sum(0)
        recall = TP / confusion_matrix.sum(1)

        f1 = (2*precision*recall) / (precision + recall)

        accu = TP.sum()
        accu = TP.sum() / confusion_matrix.sum()

        return confusion_matrix, accu, precision, recall, f1

    def value(self):
        """
        calculate the precision and recall. 
        https://en.wikipedia.org/wiki/Confusion_matrix
        
        """
        results = {}
        for i in ['wml', 'vpl']:
            results[i] = self._get_value(i)
        try:
            results['comb'] = self.correct.data.item() / self.total
        except AttributeError:
            results['comb'] = np.nan

        return results