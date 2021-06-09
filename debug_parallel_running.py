
from Tools import env_init
from Tools.logger import *
import argparse
from Experiments.Config.issue01 import *
from Experiments.Config.issue02 import *
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest="mode", default="debug", type=str)
    parser.add_argument("-l", "--log", dest="logfile", default='debug.log', type=str)
    parser.add_argument("-e", "--exp", dest="exp", default="EXP01", type=str)
    
    return parser.parse_args()

# initialization
parser = parse_args()
if parser.exp is None:
    raise ValueError
args = eval(parser.exp)(parser.mode, parser.logfile)
warning("STARTING >>>>>> {} ".format(args.name))
args.logpath = os.path.join(args.log_root, "parallel_tests", args.logfile)
ngpu, device, writer = env_init(args, logging.INFO)
args.ngpu = ngpu

for i in range(10):
    time.sleep(5)
    info("wrote by {}  == seed: {}".format(parser.exp, args.seed))