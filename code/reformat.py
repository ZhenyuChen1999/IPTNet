import os
from shutil import copyfile
from shutil import move

# This script will change the files in ROOT *PERMANENTLY*.

# Dir of raw dataset generated by `render_dataset.py`.
ROOT="./dataset/"

# The number of generated images per combination of parameters.
LIGHT_NUM=10

if __name__=="__main__":
    with open(ROOT+"params.txt") as f:
        params=f.readlines()
        for line in params:
            num=line.split(" ")[0]
            if not os.path.exists(ROOT+num):
                os.mkdir(ROOT+num)
            with open(ROOT+num+"/"+"params.txt","w") as f2:
                f2.write(line)
            for i in range(LIGHT_NUM):
                if os.path.exists(ROOT+num+"_"+str(i)+".exr"):
                    # copyfile(ROOT+num+"_"+str(i)+".exr",ROOT+num+"/"+num+"_"+str(i)+".exr")
                    move(ROOT+num+"_"+str(i)+".exr",ROOT+num+"/"+num+"_"+str(i)+".exr")
                