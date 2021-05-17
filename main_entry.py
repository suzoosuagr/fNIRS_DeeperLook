from Tools import env_init
from Tools.logger import *
import argparse
from Experiments.Config.issue01 import *
from Data.Dataset.fnirs import fNIRS_mb_label_balance_leave_subject_sla
from Experiments.Config.issue01 import *
import torch.utils.data as data 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest="mode", default="debug", type=str)
    parser.add_argument("-l", "--log", dest="logfile", default='debug.log', type=str)
    
    return parser.parse_args()

# initialization
parser = parse_args()
args = EXP01(parser.mode, parser.logfile)
warning("STARTING >>>>>> {} ".format(args.name))
args.logpath = os.path.join(args.log_root, args.name, args.logfile)
ngpu, device, writer = env_init(args, logging.INFO)
args.ngpu = ngpu

if args.mode == 'debug':
    args.data_config['train_ids'] = args.data_config['ids'].copy()
    args.data_config['train_ids'].remove('2004')
    args.data_config['eval_ids'] = ['2004']
    args.summary = False
    args.resume = False

def update_loader(fold_id, ins_list):
    train_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
            list_root = args.list_path,
            steps = args.steps_sizes,
            mode='train',
            data_config=args.data_config,
            runtime=True,
            fold_id=fold_id)
    eval_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
            list_root = args.list_path,
            steps = args.steps_sizes,
            mode='eval',
            data_config=args.data_config,
            runtime=True,
            fold_id=fold_id)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last)
    eval_loader = data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=args.drop_last)

    return train_loader, eval_loader

def update_model():
    model = 

def generate_instructors(args):
    args_ = args
    IDS = args.data_config["ids"].copy()
    for i, id in enumerate(IDS):
        print("generating instructors ... ")
        print(f"[{i}/{len(IDS)}]managing {id}")
        args_.data_config['train_ids'] = args_.data_config['ids'].copy()
        args_.data_config['train_ids'].remove(id)
        args_.data_config['eval_ids'] = [id]

        # Dataset
        train_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
                        list_root=args.list_path,
                        steps=args.steps_sizes,
                        mode='train',
                        data_config=args.data_config,
                        runtime=False,
                        fold_id=i)
        eval_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
                        list_root=args.list_path,
                        steps=args.steps_sizes,
                        mode='eval',
                        data_config=args.data_config,
                        runtime=False,
                        fold_id=i)


if __name__ == "__main__":
    # generate_instructors(args)


