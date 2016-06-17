#!/usr/bin/env python
# encoding: utf-8

import time
import subprocess
import os


for item in ['coae2014', 'coae2015', 'nlpcc_emotion', 'nlpcc_sentence']:
    data_dir = 'data' + os.sep + item
    subprocess.call('python cv_main.py ' + data_dir + ' &', shell=True)
    time.sleep(1)
