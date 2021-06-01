from Tools.base_engine import BaseEngine
import torch
from Tools.logger import *
from Tools.utils import visual_conf_mat
import skimage.io as io
import shap
import numpy as np

class fNIRS_Engine(BaseEngine):
    def __init__(self, train_loader, eval_loader, test_loader, args, writer, device) -> None:
        super(fNIRS_Engine, self).__init__(train_loader, eval_loader, test_loader, args, writer, device) 
        self.shap_map = {
            'wml':0,
            'vpl':1
        }

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(self.train_loader):
            in_data, label, tr_label, oxy_file = batch_data
            in_data = in_data.to(self.device)
            label = torch.stack(label, dim=1).to(self.device).squeeze()

            out_wml, out_vpl = self.model(in_data)
            loss_wml = self.criterion(out_wml, label[:, 0])
            loss_vpl = self.criterion(out_vpl, label[:, 1])
            loss = loss_wml + loss_vpl
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def eval_epoch(self, epoch):
        self.model.eval()
        epoch_loss = 0
        for step, batch_data in enumerate(self.eval_loader):
            in_data, label, tr_label, oxy_file = batch_data
            in_data = in_data.to(self.device)
            label = torch.stack(label, dim=1).to(self.device).squeeze()

            with torch.no_grad():
                out_wml, out_vpl = self.model(in_data)
                loss_wml = self.criterion(out_wml, label[:, 0])
                loss_vpl = self.criterion(out_vpl, label[:, 1])
                loss = loss_wml + loss_vpl
                epoch_loss += loss.item()
        return epoch_loss / len(self.eval_loader)

    def test_epoch(self):
        self.model.eval()
        epoch_loss = 0
        for step, batch_data in enumerate(self.test_loader):
            augdata_0, augdata_1 = batch_data

            data_0 = augdata_0[0].to(self.device)
            data_1 = augdata_1[0].to(self.device)

            label_0 = torch.stack(augdata_0[1], dim=1).to(self.device).squeeze()
            label_1 = torch.stack(augdata_1[1], dim=1).to(self.device).squeeze()

            with torch.no_grad():
                out_wml_0, out_vpl_0 = self.model(data_0)
                out_wml_1, out_vpl_1 = self.model(data_1)
                self.metric([torch.stack([out_wml_0.data, out_vpl_0.data], dim=2), torch.stack([out_wml_1.data, out_vpl_1.data], dim=2)], [label_0.data, label_1.data])

    def test(self, model, optimizer, metric, last=False):
        self.model, self.optimizer, start_epoch, self.min_loss = self.load_ckpt(model, optimizer)
        self.metric = metric
        # self.metric.reset()
        info("Resume from {} epoch".format(start_epoch))
        self.test_epoch()
        
        if last:
            performance = self.metric.value()
            for i in ['wml', 'vpl']:
                conf_mat, accu, precision, recall, f1 = performance[i]
                info("ACCURACY_{} = {} ".format(i, accu))
                info("Precision_{} = {} ".format(i, precision))
                info("RECALL_{} = {} ".format(i, recall))
                info("F1_{} = {} ".format(i, f1))
                conf_mat_img = visual_conf_mat(conf_mat, ['off', 'low', 'high'])
                confmat_root = "./Visual/Conf_Mat/{}/".format(self.args.name)
                ensure(confmat_root)
                io.imsave(os.path.join(confmat_root, "ENSEMBLE_{}.png".format(i)), conf_mat_img)
        return self.metric

    def shap(self, model, optimizer, proc='wml', index=0):
        self.model, self.optimizer, start_epoch, self.min_loss = self.load_ckpt(model, optimizer)
        info(f"Shap resume from {start_epoch} epoch")
        shap_device = torch.device('cpu')
        self.model = self.model.to(shap_device)

        batch_data = next(iter(self.test_loader))
        background = self.background_shapper(batch_data)
        batch_data_2 = next(iter(self.test_loader))
        test_data_0, test_data_1, label_0, label_1 = self.test_finder(batch_data_2, proc)

        e=shap.DeepExplainer(self.model, background)
        shap_values_0 = e.shap_values(test_data_0)
        shap_values_1 = e.shap_values(test_data_1)

        shap_meta_dict = {
            'cr-task':shap_values_0,
            'label_cr_task_wml':label_0,
            'id':index,
            'task-cr':shap_values_1,
            "label_task_cr_wml":label_1
        }

        shap_save_root = os.path.join("./Visual/SHAP_VALUES", self.args.name)
        ensure(shap_save_root)
        save_path = os.path.join(shap_save_root, '{:2}_{}_b0_55_60.npy'.format(index, proc))
        np.save(save_path, shap_meta_dict)

    def background_shapper(self, batch_data):
        augdata_0, augdata_1 = batch_data
        background = torch.cat([augdata_0[0][:50], augdata_1[0][:50]], dim=0)
        return background

    def test_finder(self, batch_data, proc):
        augdata_0, augdata_1 = batch_data
        label_list = augdata_0[1][self.shap_map[proc]]
        label_list = label_list // 2
        target_idx = []
        for i in range(3):
            cand = (label_list == i).nonzero(as_tuple=True)
            target_idx += list(cand[0])[:2]
        target_idx = [idx.item() for idx in target_idx]

        test_data_0 = augdata_0[0][target_idx]
        test_data_1 = augdata_1[0][target_idx]
        label_0 = augdata_0[1][self.shap_map[proc]][target_idx]
        label_1 = augdata_1[1][self.shap_map[proc]][target_idx]
        return test_data_0, test_data_1, label_0, label_1

