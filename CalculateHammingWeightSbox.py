import numpy as np
import os

traceset = np.load('convert0-10000.npz')
nlist = traceset['n']
valuelist = traceset['value']
keylist = traceset['key']
messagelist = traceset['message']
cipherlist = traceset['cipher']
hamminglist = []

FileName = "CalHammingWeight.exe"
for i in range(len(keylist)):
    cmd = FileName + " " + messagelist[i] + " " + keylist[i]
    os.system(cmd)
    result = os.popen(cmd)
    info = result.readlines()  # 读取命令行的输出到一个list
    HW = info[0]
    print(HW)
    hamminglist.append(HW)
traceset = dict(
    {'n': nlist, 'value': valuelist, 'key': keylist, 'message': messagelist, 'cipher': cipherlist,
     'HW': hamminglist})
np.savez('convert0-10000.npz', n=nlist, value=valuelist, key=keylist, message=messagelist, cipher=cipherlist,
         HW=hamminglist)
