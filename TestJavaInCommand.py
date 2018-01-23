import os

Key = "414A434135383034"
FileName = "GetSM4RoundKey"
FileExt = ".java"
JavaCCmd = "javac " + FileName + FileExt
JavaECmd = "java " + FileName + ' ' + Key
os.system(JavaCCmd)
result = os.popen(JavaECmd)
info = result.readlines()  # 读取命令行的输出到一个list
for line in info:  # 按行遍历
    line = line.strip('\r\n')
    print(line)
