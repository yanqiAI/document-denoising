# coding:utf-8
'''
files rename
'''
import os
import glob

# file list
file_path = 'E:/datasets/denoising/processing_data/blur_test_2'
file_list = os.listdir(file_path)

for (i, file) in enumerate(file_list):
    oldname = file
    newname = str('%05d'%i) + '.jpg'
    os.rename(file_path + '/' + oldname, file_path + '/' + newname)
    print(oldname, '==========>', newname)