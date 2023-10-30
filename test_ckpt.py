import os
cnt=0
cmd=open('test_cmd.sh','r').readlines()

for i in range(len(cmd)):
        os.system(cmd[i])
