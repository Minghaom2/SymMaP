import numpy as np
import scipy.sparse as sp
import struct, json, subprocess, heapq

subprocess.run(['make', 'e'])

N = 100  # Finite Difference Side Length
M = 1000 # Data set size

# Build coefficient matrix
def build_elliptic(n):
    A = np.random.random()
    B = np.random.random() * 2 / N
    C = np.random.random()
    D = np.random.random() * 2 - 1
    E = np.random.random() * 2 - 1
    F = np.random.random() * 2 - 1
    x = [A, B, C, D, E, F]
    a_1 = A * n - D * 2
    a_2 = A * n + D * 2
    b = B * (4 * n)
    c_1 = C * n - E * 2
    c_2 = C * n + E * 2
    d = 2 * (A + C) * n + F
    ones = np.ones(n-1)
    zero_ones = np.ones(n)
    zero_ones[0] = 0
    diag_main_main = d * np.ones(n**2)
    diag_main_lower = c_1 * np.concatenate((ones, np.tile(zero_ones, n-1)))
    diag_main_upper = c_2 * np.concatenate((ones, np.tile(zero_ones, n-1)))
    diag_lower_main = a_1 * np.ones(n**2 - n)
    diag_lower_lower = b * np.concatenate((ones, np.tile(zero_ones, n-2)))
    diag_lower_upper = -b * np.concatenate((np.tile(zero_ones, n-1), np.zeros(1)))
    diag_upper_main = a_2 * np.ones(n**2 - n)
    diag_upper_lower = -b * np.concatenate((np.tile(zero_ones, n-1), np.zeros(1)))
    diag_upper_upper = b * np.concatenate((ones, np.tile(zero_ones, n-2)))
    diagss = [diag_lower_lower, diag_lower_main, diag_lower_upper, diag_main_lower, diag_main_main, diag_main_upper, diag_upper_lower, diag_upper_main, diag_upper_upper]
    P = sp.diags(diagss, offsets=[-n-1, -n, -n+1, -1, 0, 1, n-1, n, n+1]).tocoo()
    if(d > 1e-5):
        return x, P
    elif(d < -1e-5):
        return x, -P
    
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
def best_dichotomy(A, pre='sor', var='-pc_sor_omega', var_value=[0.0, 2.0], accuracy=11, index=0, is_min=True):
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
while(i<M):
    i = i+1
    par, A = build_elliptic(N)
    X.append(par)
    y.append(best_dichotomy(A, pre='eisenstat', var='-pc_eisenstat_omega', index=0))
with open('X.json', 'w') as f:
    json.dump(X, f)
with open('y.json', 'w') as f:
    json.dump(y, f)