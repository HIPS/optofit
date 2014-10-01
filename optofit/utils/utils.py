from __future__ import division
import numpy as np
import scipy.linalg
import copy, itertools, collections

### these are significantly faster than the wrappers provided in scipy.linalg
potrs, potrf, trtrs = scipy.linalg.lapack.get_lapack_funcs(('potrs','potrf','trtrs'),arrays=False) # arrays=false means type=d


def extract_names_from_dtype(dtype):
    """
    Recursively extract the names from a numpy structured array dtype
    """
    def _extract_helper(dt, prefix=''):
        # dt is either a list of variables or a type
        sofar = []
        for (name, statevars) in dt:
            if isinstance(statevars, type):
                sofar.append(prefix + '_' + name)
            elif isinstance(statevars, list):
                # Recurse on the variables in the list
                if prefix == '':
                    p = name
                else:
                    p = prefix + '_' + name
                sofar.extend(_extract_helper(statevars,
                                             prefix=p))
            else:
                raise Exception("Expected statevars to be list or type, not %s" % statevars.__class__)
        return sofar

    return _extract_helper(dtype)

def solve_psd(A,b,overwrite_b=False,overwrite_A=False,chol=None):
    assert A.dtype == b.dtype == np.float64
    if chol is None:
        chol = potrf(A,lower=0,overwrite_a=overwrite_A,clean=0)[0]
    return potrs(chol,b,lower=0,overwrite_b=overwrite_b)[0]

def cholesky(A,overwrite_A=False):
    assert A.dtype == np.float64
    return potrf(A,lower=0,overwrite_a=overwrite_A,clean=0)[0]

def solve_triangular(L,b,overwrite_b=False):
    assert L.dtype == b.dtype == np.float64
    return trtrs(L,b,lower=0,trans=1,overwrite_b=overwrite_b)[0]

def solve_chofactor_system(A,b,overwrite_b=False,overwrite_A=False):
    assert A.dtype == b.dtype == np.float64
    L = cholesky(A,overwrite_A=overwrite_A)
    return solve_triangular(L,b,overwrite_b=overwrite_b), L


def interleave(*iterables):
    return list(itertools.chain.from_iterable(zip(*iterables)))

def joindicts(dicts):
    # stuff on right clobbers stuff on left
    return reduce(lambda x,y: dict(x,**y), dicts, {})

def one_vs_all(stuff):
    stuffset = set(stuff)
    for thing in stuff:
        yield thing, stuffset - set([thing])

def rle(stateseq):
    pos, = np.where(np.diff(stateseq) != 0)
    pos = np.concatenate(([0],pos+1,[len(stateseq)]))
    return stateseq[pos[:-1]], np.diff(pos)

def irle(vals,lens):
    out = np.empty(np.sum(lens))
    for v,l,start in zip(vals,lens,np.concatenate(((0,),np.cumsum(lens)[:-1]))):
        out[start:start+l] = v
    return out

def ibincount(counts):
    'returns an array a such that counts = np.bincount(a)'
    return np.repeat(np.arange(counts.shape[0]),counts)

def deepcopy(obj):
    return copy.deepcopy(obj)

def nice_indices(arr):
    '''
    takes an array like [1,1,5,5,5,999,1,1]
    and maps to something like [0,0,1,1,1,2,0,0]
    modifies original in place as well as returns a ref
    '''
    # surprisingly, this is slower for very small (and very large) inputs:
    # u,f,i = np.unique(arr,return_index=True,return_inverse=True)
    # arr[:] = np.arange(u.shape[0])[np.argsort(f)][i]
    ids = collections.defaultdict(itertools.count().next)
    for idx,x in enumerate(arr):
        arr[idx] = ids[x]
    return arr

def ndargmax(arr):
    return np.unravel_index(np.argmax(np.ravel(arr)),arr.shape)

def match_by_overlap(a,b):
    assert a.ndim == b.ndim == 1 and a.shape[0] == b.shape[0]
    ais, bjs = list(set(a)), list(set(b))
    scores = np.zeros((len(ais),len(bjs)))
    for i,ai in enumerate(ais):
        for j,bj in enumerate(bjs):
            scores[i,j] = np.dot(np.array(a==ai,dtype=np.float),b==bj)

    flip = len(bjs) > len(ais)

    if flip:
        ais, bjs = bjs, ais
        scores = scores.T

    matching = []
    while scores.size > 0:
        i,j = ndargmax(scores)
        matching.append((ais[i],bjs[j]))
        scores = np.delete(np.delete(scores,i,0),j,1)
        ais = np.delete(ais,i)
        bjs = np.delete(bjs,j)

    return matching if not flip else [(x,y) for y,x in matching]

def hamming_error(a,b):
    return (a!=b).sum()

def scoreatpercentile(data,per,axis=0):
    'like the function in scipy.stats but with an axis argument and works on arrays'
    a = np.sort(data,axis=axis)
    idx = per/100. * (data.shape[axis]-1)

    if (idx % 1 == 0):
        return a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]]
    else:
        lowerweight = 1-(idx % 1)
        upperweight = (idx % 1)
        idx = int(np.floor(idx))
        return lowerweight * a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]] \
                + upperweight * a[[slice(None) if ii != axis else idx+1 for ii in range(a.ndim)]]

def stateseq_hamming_error(sampledstates,truestates):
    sampledstates = np.array(sampledstates,ndmin=2).copy()

    errors = np.zeros(sampledstates.shape[0])
    for idx,s in enumerate(sampledstates):
        # match labels by maximum overlap
        matching = match_by_overlap(s,truestates)
        s2 = s.copy()
        for i,j in matching:
            s2[s==i] = j
        errors[idx] = hamming_error(s2,truestates)

    return errors if errors.shape[0] > 1 else errors[0]

def _sieve(stream):
    # just for fun; doesn't work over a few hundred
    val = stream.next()
    yield val
    for x in itertools.ifilter(lambda x: x%val != 0, _sieve(stream)):
        yield x

def primes():
    return _sieve(itertools.count(2))

def get_dict_indices(d, prop, init = []):
    ans = []
    for key, value in d.iteritems():
        loc = init + [key]
        if prop(value):
            ans.append(loc)
        elif isinstance(value, dict):
            ans.extend(get_dict_indices(value, prop, init=loc))
    return ans

def value_from_dict_index(d, index):
    if index == []:
        return d
    else:
        return value_from_dict_index(d[index[0]], index[1:])

def get_indices(
        data,
        get_keys, stop_condition, parent_condition,
        init=[], get_value = lambda x, k: x[k]):

    ans = []
    for key in get_keys(data):
        loc   = init + [key]
        value = get_value(data, key)
        if stop_condition(value):
            ans.append(loc)
        elif parent_condition(value):
            ans.extend(
                get_indices(
                    value,
                    get_keys, stop_condition, parent_condition,
                    init = loc, get_value = get_value
                )
            )
    return ans

def index_to_dtype(index):
    if index == []:
        return np.float64
    else:
        def inner_iter(idx):
            if idx == []:
                return np.float64
            else:
                return [(idx[0], inner_iter(idx[1:]))]
        return np.dtype(inner_iter(index))

def set_index(d, index, value):
    if len(index) == 1:
        d[index[0]] = value
    else:
        set_index(d[index[0]], index[1:], value)

def array_to_dtype(array, dtype, axis=1):
    indices = get_indices(dtype,
                          lambda dt: dt.names,
                          lambda v: v.names == None,
                          lambda v: v.names != None
                          )
    if len(array.shape) == 2:
        ans = np.zeros(array.shape[axis], dtype = dtype)
        for i, index in enumerate(indices):
            if axis == 1:
                set_index(ans, index, array[i])
            else:
                set_index(ans, index, array[:, i])
        return ans
    else:
        return array.astype(dtype)

def indices_to_dtype(indices):
    tree = {}
    for index in indices:
        cur_level = tree
        for key in index:
            if key not in cur_level:
                cur_level[key] = {}
            cur_level = cur_level[key]

    def tree_to_dtype(t):
        ans = []
        for key, value in t.iteritems():
            #print key, value
            if len(value) == 0:
                return [(key, np.float64)]
            else:
                ans.append((key, tree_to_dtype(value)))
        return ans

    return tree_to_dtype(tree)
    
def get_item_at_path(arr, path):
    """
    Get an item at the specified path in a hierarchical structured array.
    :param arr: hierarchical structured array
    :param path: list of keys, e.g. ['neuron', 'body']
    :return: e.g. arr['neuron']['body']
    """
    return reduce(lambda x,p: x.__getitem__(p), path, arr)

def as_matrix(sarray, D0=None):
    """
    Convert a hierarchical struct array to a matrix

    :param sarray:
    :param D0: size of the structured array
    :return:
    """
    if D0 is None:
        D0 = sz_dtype(sarray.dtype)

    D1 = sarray.size
    return sarray.view(np.float).reshape((D0, D1), order='F')

def as_sarray(matrx, dtype):
    """
    View the matrix as a hierarchical structured array
    :param matrx:
    :param dtype:
    :return:
    """
    return matrx.ravel('F').view(dtype)

def sz_dtype(dtype):
    return np.zeros(1, dtype=dtype).view(np.float).size