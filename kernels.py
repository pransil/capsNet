""" kernels.py """
import numpy as np
import chainer.links as L

def rotate3(a):
    """ rotate 3x3 array clockwise by one
    0   1   2
    3   4   5
    6   7   8
    """
    save = a[0]
    a[0] = a[1]
    a[1] = a[2]
    a[2] = a[5]
    a[5] = a[8]
    a[8] = a[7]
    a[7] = a[6]
    a[6] = a[3]
    a[3] = save
    return a

def rotation_check(a, b):
    """ Is matrix a a rotation of matrix b?
    Both must be 3x3 numpy arrays and are assumed binary, 0/1 (but should work if not)
    Returns:
        -1 if not
        0 if equal (same after 0 rotations)
        1..8 for how many rotations needed to make match

    """
    assert a.shape == (9,)
    assert b.shape == (9,)

    if sum(a) != sum(b):
        return -1
    if a[4] != b[4]:        # Center bits don't rotate, must be =
        return -1

    a = a.reshape(9)
    b = b.reshape(9)
    for i in range(8):
        if np.array_equal(a, b):
            return i
        else:
            b = rotate3(b)

    return -2        # same # of bits but not a match


def make_similiarity_matrix(kernels, k_props):
    """ How similar are two kernels?
        kernels: array of all 512 (3x3) kernels
        k_props: array of (index, rotation) for all kernels
        Returns:
        k_sim:  512 x 512 array of |a-b|

    """
    length = len(kernels)           # Should be 512
    kS = np.zeros((length, length), dtype=np.float16)
    for i in range(length):
        a = np.asarray(kernels[0:512][i][0:1][0])
        a = a.reshape(9)
        for j in range(length):
            b = np.asarray(kernels[0:512][j][0:1][0])
            b = b.reshape(9)
            if i == j:
                diff = 0
            else:
                diff = np.sum(np.abs(a - b))
            kS[i][j] = (9.0 - diff)/9.0

    return kS


def update_csm(csm, z):
    """Update avg evidence given z
        csm['p'] - Cross support matrix, probability, 512 x 512
        csm['n'] - Cross support matrix, n or count, 512 x 512
        z   - layer output, m x 512 (512=# of kernels/channels)
    Returns:
        updated csm (which is not really necessary because its just a reference)
    """
    batch_size, kernels, x_dim, y_dim = z.shape
    csm_p = csm['p']
    csm_n = csm['n']
    winner = np.argmax(z.data, axis=1)
    winner_value = np.amax(z.data, axis=1)

    for b in range(batch_size):
        for k in range(kernels):
            for x in range(x_dim):
                for y in range(y_dim):
                    n = batch_size * (kernels - 1)
                    e = np.sum(z.data, axis=1) - winner_value
                    em = np.sum(e, axis=0)
                    p = em / (batch_size * (kernels - 1))
                    #winner = z[b][k][x][y]
                    csm_n[winner] += 1

        csm_p += (y[b] - csm_p)/csm_n



def make_cross_support_matrix(size):
    """ What is the likelihood of k[i] given evidence for all k?
        Build the cross_support_matrix (csm) where each time k[i] wins
        update avg evidence for all other k.
        size    - size the array will be, also the # of kernels
        Returns:
        csm  512 x 512 array of |a-b|
    """
    csm = {'p': 0, 'r': 0}
    csm['p'] = np.zeros((size, size), dtype=np.float16)
    csm['n'] = np.zeros((size, size), dtype=np.int32)
    return csm



def similarity_check(kernels, k_props):
    """ How similar are two kernels?
        kernels: array of all 512 (3x3) kernels
        k_props: array of (index, rotation) for all kernels
        Returns:
        |argmin(sum(a-b))| across all rotations

    """
    kS = np.zeros((72,72), dtype=int)
    k_uniques = get_uniques(kernels, k_props)
    length = len(k_uniques)
    for i in range(length):
        a = np.asarray(k_uniques[i][1][0:1][0])
        a = a.reshape(9)
        for j in range(length):
            b = np.asarray(k_uniques[j][1][0:1][0])
            b = b.reshape(9)
            if i == j:
                min = 0
            else:
                min = 81                    # Max possible diff
                for r in range(8):
                    diff = np.sum(np.abs(a - b))
                    min = min if min < diff else diff
                    b = rotate3(b)
            kS[i][j] = min

    return kS



def make_array(i):
    s = str(bin(i))
    s = s[2:]
    a = np.zeros((9), dtype=np.float32)
    for x in range(9):
        if len(s) > x:
            x += 1
            if s[-x] == '1':
                index = 9 - x
                a[index] = 1

    return a


def get_uniques(k, k_props):
    c = 0
    k_uniques = []
    for i in range(len(k)):
        if k_props[i][0] == -1:
            k_uniques.append([i, k[i]])
            c += 1
    return k_uniques


def display_uniques(k, k_props):
    c = 0
    for i in range(len(k)):
        if k_props[i][0] == -1:
            display(k[i])
            c += 1
    #print(c, " uniques")


def display(a):
    b = a.reshape((3,3))
    print(b)


def make_kernels():
    """
    Returns:
        kernels: array of all 512 (3x3) kernels
        k_props: array of (index, rotation) for all kernels

    :return:
    """
    k = []
    k_props = np.zeros((512,2), dtype=np.float32)
    k_props -= 1
    for i in range(512):
        a = make_array(i)
        k.append(a)
    for i in range(512):
        for j in range(i, 512):
            if i == j:
                continue
            if np.sum(k[j]) > np.sum(k[i]):
                continue
            if k_props[j][0] != -1:
                continue                   # Has already matched
            b = np.copy(k[j])
            r = rotation_check(k[i], b)
            if r >= 0:
                k_props[j] = [i, r]
                continue

    kernels = np.reshape(k, (512,1,3,3))
    return kernels, k_props

if __name__ == '__main__':
    make_kernels()
