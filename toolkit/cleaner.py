import os
import shutil


def list_files(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        dirpath = dirpath.replace("\\", "/")
        dir_list = dirpath.split("/")[1:]
        hidden = False
        for i in dir_list:
            if i[0] == ".":
                hidden = True
                break
        if hidden == True:
            continue

        for i in dirnames:
            if i == "__pycache__":
                target_dir = f"{dirpath}/{i}"
                shutil.rmtree(target_dir)
                print(target_dir)

    print("clean up")


list_files(".")
