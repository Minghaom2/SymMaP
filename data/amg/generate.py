import numpy as np
import scipy.sparse as sp
import struct, json, subprocess, heapq
import numpy.polynomial.chebyshev as cb

subprocess.run(['make', 'e'])

N = 100  # Finite Difference Side Length
M = 1000 # Data set size

# Build coefficient matrix
def build_chebyshev(n, N, k):
    X, Y = np.meshgrid(np.arange(0, 1+1/(N+1), 1/(N+1)), np.arange(0, 1+1/(N+1), 1/(N+1)))
    K = []
    for i in range(N):
        for j in range(N):
            m = 0
            for i_0 in range(n):
                for j_0 in range(n):
                    x_cheby = cb.Chebyshev([int(t==i_0) for t in range(n)])
                    y_cheby = cb.Chebyshev([int(t==j_0) for t in range(n)])
                    m = m + k[i_0 + n*j_0] * x_cheby.__call__(X[i][j]) * y_cheby.__call__(Y[i][j])
            K.append(m)
    return np.array(K).reshape(N, N)

def build_darcy(coef):
    K = coef.shape[0]
    s = K - 2
    diag_list = []
    off_diag_list = []
    for j in range(1, K-1):
        diag_values = np.array([
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0])),
            0.5 * (coef[0:K-2, j] + coef[1:K-1, j]) + 0.5 * (coef[2:K, j] + coef[1:K-1, j]) + \
            0.5 * (coef[1:K-1, j-1] + coef[1:K-1, j]) + 0.5 * (coef[1:K-1, j+1] + coef[1:K-1, j]),
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0]))
        ])
        diag_list.append(diag_values)
        if j != K-2:
            off_diag = -0.5 * (coef[1:K-1, j] + coef[1:K-1, j+1])
            off_diag_list.append(off_diag)
    diag_output = np.concatenate(diag_list,axis=1)
    off_diag_output = np.concatenate(off_diag_list,axis=0)
    A = (sp.diags(diag_output,[-1,0,1],(s**2,s**2)) + sp.diags((off_diag_output,off_diag_output),[-(K-2),(K-2)],(s**2,s**2))) * (K-1)**2
    return A
    
# Write the created coefficient matrix to .bin
def writetobin(A):
    if A.format != 'csr':
        A = A.tocsr()
    with open('A.bin', 'wb') as f:
        f.write(struct.pack('ii', A.shape[0], A.shape[1]))
        nnz = len(A.data)
        f.write(struct.pack('i', nnz))
        for item in [A.indptr, A.indices, A.data]:
            f.write(struct.pack(f'{len(item)}i', *item.astype(np.int32))) if item.dtype != np.float64 else f.write(struct.pack(f'{len(item)}d', *item))
    return None

# Run the file e.c to solve the equation
def run(pre='sor', var=None, var_value=1.0):
    cmd = ['./e', '-ksp_max_it', '1000', '-pc_type']
    cmd.append(pre)
    if var != None:
        cmd.append(var)
        cmd.append(str(var_value))
    out = []
    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    out.append(float(result.stdout.split()[0]))
    out.append(int(result.stdout.split()[1]))
    out.append(float(result.stdout.split()[6]))
    return out

# Find the optimal parameters by dichotomy 
def best_dichotomy(A, pre='sor', var='-pc_sor_omega', var_value=[0.0, 2.0], accuracy=11, index=2, is_min=True):
    writetobin(A)
    tol = (var_value[1] - var_value[0]) / 4
    i_0, i, j, k, k_0 = var_value[0], var_value[0] + tol, var_value[0] + 2*tol, var_value[0] + 3*tol, var_value[1]
    result_i = run(pre=pre, var=var, var_value=i)
    result_j = run(pre=pre, var=var, var_value=j)
    result_k = run(pre=pre, var=var, var_value=k)
    for t in range(accuracy):
        if((result_i[index] <= result_j[index] and is_min) or (result_i[index] >= result_j[index] and not is_min)):
            k_0 = j
            k = (i+j)/2
            j = i
            i = (i_0+j)/2
            result_j = result_i
            result_i = run(pre=pre, var=var, var_value=i)
            result_k = run(pre=pre, var=var, var_value=k)
        elif((result_k[index] <= result_j[index] and is_min) or (result_k[index] >= result_j[index] and not is_min)):
            i_0 = j
            i = (j+k)/2
            j = k
            k = (j+k_0)/2
            result_j = result_k
            result_i = run(pre=pre, var=var, var_value=i)
            result_k = run(pre=pre, var=var, var_value=k)
        else:
            i_0 = i
            i = (i+j)/2
            k = (j+k)/2
            k_0 = k
            result_i = run(pre=pre, var=var, var_value=i)
            result_k = run(pre=pre, var=var, var_value=k)
    return j

# Find the optimal parameters by grid search
def best_enumeration(A, pre='sor', var='-pc_sor_omega', var_value=[0.0, 2.0], index=0, is_max=False):
    writetobin(A)
    results = []
    pars = np.arange(var_value[0]+0.02, var_value[1], 0.02)
    for par in pars:
        result = run(pre=pre, var=var, var_value=par)
        results.append(result[index])
    pars_precise = []
    results_precise = []
    if is_max:
        m = np.max(np.array(results))
        for i in range(len(results)):
            if results[i]/m > 0.95:
                t = var_value[0] + 0.02*(i+1)
                pars_precise = pars_precise + np.arange(t-0.01, t+0.01, 0.001).tolist()
    else:
        m = np.min(np.array(results))
        for i in range(len(results)):
            if results[i]/m < 1.05:
                t = var_value[0] + 0.02*(i+1)
                pars_precise = pars_precise + np.arange(t-0.01, t+0.01, 0.001).tolist()
    for par in pars_precise:
        result = run(pre=pre, var=var, var_value=par)
        results_precise.append(result[index])
    if is_max:
        return pars_precise[np.argmax(np.array(results_precise))]
    return pars_precise[np.argmin(np.array(results_precise))]

# Build data set
X = []
y = []
i = 0
n = 3
while(i<M):
    i = i+1
    par = np.random.rand(n**2)
    A = build_darcy(build_chebyshev(n, N, par))
    X.append(par)
    y.append(best_enumeration(A, pre='gamg', var='-pc_gamg_threshold', var_value=[0.0, 1.0]))
with open('X.json', 'w') as f:
    json.dump(X, f)
with open('y.json', 'w') as f:
    json.dump(y, f)