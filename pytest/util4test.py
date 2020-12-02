import os


def delete_dir_if_exist(dir_):
    if os.path.isdir(dir_):
        command = 'rm -rf %s' % dir_
        print(command)
        os.system(command)
