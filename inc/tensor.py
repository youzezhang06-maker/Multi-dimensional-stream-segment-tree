from inc.header import *

def mat_svd(X, top):
    """leading singular values and left singular vectors"""
    U, s, VH = tlin.svd(X, full_matrices = (top > min(X.size())))
    return s[: top], U[:, : top]

def _mat_svd_eigh(X, top):
    """leading singular values and left singular vectors of an Hermitian matrix"""
    if X.size(dim = 0) > X.size(dim = 1):
        s, V = _mat_svd_eigh(X.T, top = top)
        return s, X.mm(torch.where(s != 0, V / s, s))
    else:
        s2, U = tlin.eigh(X.mm(X.T))
        return s2[-top :].flip(-1).clamp_min(0.).sqrt(), U[:, -top :].flip(-1)

def mat_svd_eigh(X, top):
    s, U = _mat_svd_eigh(X, top)
    if U.size(dim = 1) < top:
        s = F.pad(s, (0, top - s.size(dim = 0)))
        U = F.pad(U, (0, top - U.size(dim = 1)))
    return s, U

def tensor_norm(X):
    """Frobenius norm of the tensor X"""
    return torch.sqrt(torch.sum(torch.square(X)))

def tensor_norm2(X):
    """squared Frobenius norm of the tensor X"""
    return torch.sum(torch.square(X))

def tensor_unfold(X, dim):
    """matricization"""
    return X.permute(*[(p if p > dim else ((p - 1) if p > 0 else dim)) for p in range(X.dim())]).flatten(start_dim = 1)

def tensor_fold(X, dim, sizes):
    """reshape a matrix into a tensor"""
    return X.reshape([(sizes[p] if p > dim else (sizes[p - 1] if p > 0 else sizes[dim])) for p in range(len(sizes))]).permute(*[(p if p > dim else ((p + 1) if p < dim else 0)) for p in range(len(sizes))])

def tensor_mat_mul(X, A, dim): # 
    """tensor--matrix product (X x_dim A)"""
    sizes = list(X.size())
    X = A.mm(tensor_unfold(X, dim = dim))
    sizes[dim] = X.size(dim = 0)
    return tensor_fold(X, dim = dim, sizes = sizes)

def tensor_mats_mul(X, A_dim_list): # assuming that dims are distinct
    """product of a tensor with a list of matrices"""
    A_dim_list = sorted(A_dim_list, key = lambda A_dim: (A_dim[0].size(dim = 0) / A_dim[0].size(dim = 1))) # to reduce computation
    for A, dim in A_dim_list:
        X = tensor_mat_mul(X = X, A = A, dim = dim)
    return X

def tensor_pad(X, sizes):
    """pad a tensor to a given shape"""
    n_dims = len(sizes)
    sizes = [sizes[p] - X.size(dim = p) for p in range(n_dims)]
    return F.pad(X, pad = [(sizes[n_dims - (p // 2) - 1] if p & 1 else 0) for p in range(n_dims * 2)], mode = 'constant', value = 0.)
