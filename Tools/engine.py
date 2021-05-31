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

    def shap(self, model, optimizer):
        self.model, self.optimizer, start_epoch, self.min_loss = self.load_ckpt(model, optimizer)
        info(f"Shap resume from {start_epoch} epoch")
        shap_device = torch.device('cpu')
        self.model = self.model.to(shap_device)

        batch_data = next(iter(self.test_loader))
        background, test_data_0, test_data_1, label_0, label_1 = self.shap_epoch(batch_data)
        # background = background.to(shap_device)
        # test_data_0 = test_data_0.to(shap_device)
        # test_data_1 = test_data_1.to(shap_device)
        
        e=shap.DeepExplainer(self.model, background)
        shap_values_0 = e.shap_values(test_data_0)
        shap_values_1 = e.shap_values(test_data_1)

        shap_meta_dict = {
            'cr-task':shap_values_0,
            'label_cr_task_wml':label_0[0][55:60],
            'id':self.args.data_config['eval_id'],
            'task-cr':shap_values_1,
            "label_task_cr_wml":label_1[0][55:60],
        }

        shap_save_root = os.path.join("./Visual/SHAP_VALUES", self.args.name)
        ensure(shap_save_root)
        save_path = os.path.join(shap_save_root, 'wml_{}_b0_55_60.npy'.format(id))
        np.save(save_path, shap_meta_dict)

    def shap_epoch(self, batch_data):
        augdata_0, augdata_1 = batch_data
        background = torch.cat([augdata_0[0][:50], augdata_1[0][:50]], dim=0)
        test_data_0 = augdata_0[0][55:60]
        test_data_1 = augdata_1[0][55:60]
        label_0 = augdata_0[1]
        label_1 = augdata_1[1]
        return background, test_data_0, test_data_1, label_0, label_1



