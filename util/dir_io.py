import os


def delete_dir_if_exist(dire):
    if os.path.isdir(dire):
        # command = 'sudo rm -rf %s' % dire
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def save_file(dire):
    os.system('sudo touch %s' % dire)
    os.system('sudo chmod 766 %s' % dire)
