import os


SRC = "/home/utz/Master/Master_Project_2023/aFolder/file"
DST = "/home/utz/Master/Master_Project_2023/file"

if not os.path.exists(SRC):
    os.mknod(SRC)

if not os.path.islink(DST):
    os.symlink(SRC, DST)
