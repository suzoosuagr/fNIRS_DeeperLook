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
from Experiments.Config.issue01 import *

label_map = {
        0:'off',
        1:'low',
        2:'high'
    }

def draw_ensemble_single_frame(sv, y, save_path, parts='task'):
    # sv = np.array(sv)
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

        # mean along the sequence axis (time, I mean. )
        target_sv = np.mean(target_sv, axis=1, keepdims=False)
        im = axes[row, 0].imshow(target_sv[row, 0, :, :], cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
        if row == 0:
            axes[row, 0].set_title('Oxy')
            axes[row, 1].set_title('DeOxy')
        im = axes[row, 1].imshow(target_sv[row, 1, :, :], cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
        gt_label = gt_label // 2
        axes[row, 0].set_xlabel('[mean]-[{}]-[{}]'.format(parts, label_map[gt_label]), loc='left', fontsize='xx-large')
    fig.subplots_adjust(hspace=0.5)
    fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal")
    # save_path = os.path.join(save_root, '{:04}.png'.format(FRAME))
    plt.savefig(save_path, dpi=80, transparent=False, bbox_inches='tight')
    plt.close('all')


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

def visual_shap_videos(args, video_root, proc):
    ensure(video_root)
    Basic_Name = args.name
    IDS = args.data_config["ids"].copy()
    for i, id in enumerate(IDS):
        args.data_config['train_ids'] = args.data_config['ids'].copy()
        args.data_config['train_ids'].remove(id)
        args.data_config['eval_ids'] = [id]
        args.name = "{}_{:02}".format(Basic_Name, i)
        shap_path = os.path.join('./Visual/SHAP_VALUES/', args.name, f'{i}_{proc}_b0_55_60.npy')
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

        video_path = os.path.join(video_root, f'{i}_{proc}_b0_55_60.mp4')

        # drawing cr task
        for frame in trange(100):
            if frame < 50:
                parts = 'cr'
            else:
                parts = 'task'
            draw_single_frame(sv_cr_task, y_cr_task, "../outputs/fnirs_temp/", frame, parts=parts)
        file_names = get_files("../outputs/fnirs_temp/", extension_filter='.png')
        release_shap_values(file_names, video_path)


def ensemble_visual(args, save_root, proc):
    """
        save the esemble along the time and subjects. 
    """
    ensure(save_root)
    Basic_Name = args.name
    IDS = args.data_config["ids"].copy()
    # synthetic_sv = np.zeros((6, 6, 1, 2, 6, 22))
    for i, id in enumerate(IDS):
        args.data_config['train_ids'] = args.data_config['ids'].copy()
        args.data_config['train_ids'].remove(id)
        args.data_config['eval_ids'] = [id]
        args.name = "{}_{:02}".format(Basic_Name, i)
        shap_path = os.path.join('./Visual/SHAP_VALUES/', args.name, f'{i}_{proc}_b0_55_60.npy')
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
        synthetic_sv = (np.array(sv_cr_task)[:, :, 50:, :, :, :] \
                        + np.array(sv_task_cr)[:, :, :50, :, :, :])\
                        /2 
        synthetic_sv = topK_shap(synthetic_sv, k=1)
        output_path = os.path.join(save_root, f'jojo__{i}_{proc}_allframes.png')
        draw_ensemble_single_frame(synthetic_sv, y_cr_task, output_path)


def topK_shap(sv, k=3):
    sv_array = np.array(sv)
    sv_array_time = np.sum(sv_array, axis=(4,5))
    topk_time_indx = np.argsort(sv_array_time, axis=2)
    topk_time_indx = topk_time_indx[:,:,:k,:]
    topk_indx_sv_array = np.expand_dims(topk_time_indx, axis=(4,5))

    return np.take_along_axis(sv_array, topk_indx_sv_array, axis=2)


if __name__ == '__main__':
    """
        - need to get the SHAPE VALUE first, at the main_entry.py function. 
    """

    shap_saveRoot = './Visual/shap_map/'
    ensure(shap_saveRoot)
    args = EXP01('debug', 'visualize.log')
    # visual_shap_videos(args, shap_video_save, 'vpl')
    ensemble_visual(args, shap_saveRoot, 'wml')
    


