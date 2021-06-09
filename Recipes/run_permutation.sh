#!/bin/bash
# python main_static_test.py -m train -l condor_debug.log
python debug_parallel_running.py -m train -l debug.log -e EXP01 &
python debug_parallel_running.py -m train -l debug.log -e EXP02 &
python debug_parallel_running.py -m train -l debug.log -e EXP03 &
python debug_parallel_running.py -m train -l debug.log -e EXP04 &
python debug_parallel_running.py -m train -l debug.log -e EXP05 &
python debug_parallel_running.py -m train -l debug.log -e EXP06 &
python debug_parallel_running.py -m train -l debug.log -e EXP07 &
python debug_parallel_running.py -m train -l debug.log -e EXP08 &
python debug_parallel_running.py -m train -l debug.log -e EXP09 &
python debug_parallel_running.py -m train -l debug.log -e EXP10 &
