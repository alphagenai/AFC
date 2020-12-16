# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:06:51 2020

@author: mark
"""

import os
import platform

if platform.system()=='Linux':
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'../AFCproj-keyfile.json'
else:
    key_path = r"C:\Users\mat4m_000\Documents\Wellow data\SFC\AFCproj-keyfile.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

