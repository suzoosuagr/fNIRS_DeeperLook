import numpy as np
from numpy.core.fromnumeric import shape
import shap
import matplotlib.pyplot as plt
from torch._C import dtype
import cv2
import Tools.colors as colors
from Tools.utils import ensure, get_files
import os
from tqdm import trange

def draw_single_frame(sv, y, save_root, frame, parts='task'):
    FRAME = frame
    sv = np.array(sv)
    nrows = sv.shape[1]
    ncols= sv.shape[3]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15), facecolor='white')
    for row in range(nrows):
        gt_label = int(y[row])
        target_sv = sv[gt_label]
        # print(target_sv.shape)  # [N, 100, 2, 5, 21]
        # break
        abs_vals = np.abs(target_sv[row]).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)

        im = axes[row, 0].imshow(target_sv[row, FRAME, 0, :, :], cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
        if row == 0:
            axes[row, 0].set_title('Oxy')
            axes[row, 1].set_title('DeOxy')
        im = axes[row, 1].imshow(target_sv[row, FRAME, 1, :, :], cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
        gt_label = gt_label // 2
        axes[row, 0].set_xlabel('[{}/100]-[{}]-[{}]'.format(FRAME, parts, label_map[gt_label]), loc='left', fontsize='xx-large')
    fig.subplots_adjust(hspace=0.5)
    fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal")
    save_path = os.path.join(save_root, '{:04}.png'.format(FRAME))
    plt.savefig(save_path, dpi=80, transparent=False, bbox_inches='tight')
    plt.close('all')

def release_shap_values(file_names, rel_video_name):
    """
        release the frames
    """
    img_size = cv2.imread(file_names[0]).shape
    rel = cv2.VideoWriter(rel_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (img_size[1], img_size[0]))
    for i in range(len(file_names)):
        img = cv2.imread(file_names[i])
        rel.write(img)
    rel.release()

def visual_shap_videos(args, video_root):
    ensure(video_root)
    Basic_Name = args.name.copy()
    IDS = args.data_config["ids"].copy()
    for i, id in enumerate(IDS):
        args.data_config['train_ids'] = args.data_config['ids'].copy()
        args.data_config['train_ids'].remove(id)
        args.data_config['eval_ids'] = [id]
        args.name = "{}_{:02}".format(Basic_Name, i)
        
        

if __name__ == '__main__':
    shap_video_save = os.path.join('../outputs/fnirs_shap_video/ISSUE01_EXP01/')
    ensure(shap_video_save)
    shap_path = './Visual/SHAP_VALUES/ISSUE01_EXP01_00/wml_<built-in function id>_b0_55_60.npy'
    shap_meta_dict = np.load(shap_path, allow_pickle=True)
    shap_meta_dict = shap_meta_dict[()]

    sv_cr_task = shap_meta_dict['cr-task']
    y_cr_task = shap_meta_dict['label_cr_task_wml']
    sv_task_cr = shap_meta_dict['task-cr']
    y_task_cr = shap_meta_dict['label_task_cr_wml']

    label_map = {
        0:'off',
        1:'low',
        2:'high'
    }
    save_root = "../outputs/fnirs_temp/"
    ensure(save_root)

#   Drawing cr task
    for frame in trange(100):
        if frame < 50:
            parts = 'cr'
        else:
            parts = 'task'
        draw_single_frame(sv_cr_task, y_cr_task, save_root, frame, parts=parts)
    file_names = get_files(save_root, extension_filter='.png')
    release_shap_values(file_names, os.path.join(shap_video_save, 'wml_2001_b0_55_60.mp4'))  # b0_55_60 : batch 0, batch[55:60] been used as to_test for shap.


