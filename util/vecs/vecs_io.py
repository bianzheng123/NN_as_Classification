import numpy as np
import struct


# to get the .vecs
# np.set_printoptions(threshold=np.inf)  # display all the content when print the numpy array


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy(), d


def fvecs_read(fname):
    data, d = ivecs_read(fname)
    return data.view('float32'), d


def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy(), d


# put the part of file into cache, prevent the slow load that file is too big
def fvecs_read_mmap(fname):
    x = np.memmap(fname, dtype='int32', mode='r', order='C')
    # x = np.memmap(fname, dtype='int32')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:], d


def bvecs_read_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r', order='C')
    # x = np.memmap(fname, dtype='uint8')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:], d


def ivecs_read_mmap(fname):
    x = np.memmap(fname, dtype='int32', mode='r', order='C')
    # x = np.memmap(fname, dtype='int32')
    d = x[0]
    return x.reshape(-1, d + 1)[:, 1:], d


# store in format of vecs
def fvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))  # *dimension就是int, dimension就是list
        f.write(struct.pack('f' * len(x), *x))

    f.close()


def ivecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('i' * len(x), *x))

    f.close()


def bvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('B' * len(x), *x))

    f.close()


def read_all(global_path, base_file_name, gnd_file_name, learn_file_name, query_file_name):
    # read from the file
    base_data, vector_dim = fvecs_read(global_path + base_file_name)
    groundtruth_data, gnd_dim = ivecs_read(global_path + gnd_file_name)
    learn_data = fvecs_read(global_path + learn_file_name)[0]
    query_data = fvecs_read(global_path + query_file_name)[0]
    return base_data, groundtruth_data, learn_data, query_data, vector_dim, gnd_dim


def read_all_bvecs(global_path, base_file_name, gnd_file_name, learn_file_name, query_file_name):
    # read from the file
    base_data, vector_dim = bvecs_read_mmap(global_path + base_file_name)
    groundtruth_data, gnd_dim = ivecs_read(global_path + gnd_file_name)
    learn_data = None
    # learn_data = bvecs_read_mmap(global_path + learn_file_name)[0]
    query_data = bvecs_read_mmap(global_path + query_file_name)[0]

    base_data = base_data.astype(np.float32)
    # learn_data = learn_data.astype(np.float32)
    query_data = query_data.astype(np.float32)
    print(base_data.flags['C_CONTIGUOUS'])

    return base_data, groundtruth_data, learn_data, query_data, vector_dim, gnd_dim


# global_path_b = '/home/bz/SIFT/bigann/'
# base_data = bvecs_read_mmap(global_path_b + 'bigann_base.bvecs')
# bvecs_write(global_path_b + 'test.bvecs', base_data)
# test_data = bvecs_read(global_path_b + 'test.bvecs')
# print(test_data)
