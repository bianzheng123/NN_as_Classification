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
    return data.view('float32').astype(np.float32), d


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


def read_txt(dire):
    with open(dire, "r") as f:
        txt = f.read().split('\n')[:-1]
    return np.array(txt).astype(np.float32)


def read_data_l2(config):
    # read from the file
    data_dir = config['data_dir']
    base_dir = "%s/base.fvecs" % data_dir
    base = fvecs_read_mmap(base_dir)[0].astype(np.float32)
    query_dir = '%s/query.fvecs' % data_dir
    query = fvecs_read_mmap(query_dir)[0].astype(np.float32)
    gnd_dir = '%s/gnd-50.ivecs' % data_dir
    gnd = ivecs_read_mmap(gnd_dir)[0].astype(np.int)
    base_base_gnd_dir = '%s/base_base_gnd-150.ivecs' % data_dir
    base_base_gnd = ivecs_read_mmap(base_base_gnd_dir)[0].astype(np.int)
    return base, query, gnd, base_base_gnd


def read_data_string(config):
    # read from the file
    data_dir = config['data_dir']
    base_dir = "%s/base.npy" % data_dir
    base = np.load(base_dir)
    query_dir = '%s/query.npy' % data_dir
    query = np.load(query_dir)
    gnd_dir = '%s/gnd.npy' % data_dir
    gnd = np.load(gnd_dir)
    base_base_gnd_dir = '%s/base_base_gnd.npy' % data_dir
    base_base_gnd = np.load(base_base_gnd_dir)
    return base, query, gnd, base_base_gnd

# data, d = ivecs_read("data/dataset/deep/gnd-50.ivecs")
# print(data.shape)
# print(d)
