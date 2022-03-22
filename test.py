"""
DCRNN
Description: 
Author: LQZ
Time: 2022/3/22 14:05 
"""


import scipy.sparse
import numpy as np

row = np.array([0, 3, 1, 0])
col = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
res = scipy.sparse.coo_matrix((data, (row, col)), shape=(4, 4))

d = np.array(res.sum(1))
print(d)
d_inv_sqrt = np.power(d, -0.5).flatten()
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)

i_mat = scipy.sparse.eye(res.shape[0])
print(res.toarray())
print(d_mat_inv_sqrt.toarray())
print(res.dot(d_mat_inv_sqrt).transpose().toarray())
print(res.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray())
res_mat = i_mat - res.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
new = i_mat - d_mat_inv_sqrt.dot(res).dot(d_mat_inv_sqrt)

print(res_mat.toarray())
print(new.toarray())
print("complete")
