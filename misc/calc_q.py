#coding: UTF-8
"""
Calculating Q values referred to in the book 
"強くなるロボティック・ゲームプレイヤーの作り方".
"""

import numpy as np

### Parameters.

n_iter = 100
discount = 0.9

# Pt(s(t=1)|s(t=0), L)
s0L = np.array([[0.9, 0.1, 0, 0],
                [0.9, 0.1, 0, 0],
                [0, 0.9, 0.1, 0],
                [0, 0, 0.9, 0.1]])

# Pt(s(t=1)|s(t=0), R)
s0R = np.array([[0.1, 0.9, 0, 0],
                [0, 0.1, 0.9, 0],
                [0, 0, 0.1, 0.9],
                [0, 0, 0.1, 0.9]])

### 1. calculating expected values of the rewards at the step t=0.

s0L_value = s0L[:,3]
s0R_value = s0R[:,3]

### 2. calculating expected values after t=0.

# state transition matrix.
A = np.array([[0.5, 0.5, 0, 0],
              [0.45, 0.1, 0.45, 0],
              [0, 0.45, 0.1, 0.45],
              [0, 0, 0.5, 0.5]])

# prepare to calculate powers of the transition matrix. 
la, v = np.linalg.eig(A)
D = np.diag(la)
inv_v = np.linalg.inv(v)

# sum up expected values of each step.
values = np.zeros(4)
for i in range(1, n_iter):
   expected = np.dot(np.dot(v, D ** i), inv_v)[:,3] * (discount ** i)
   values += expected

# calculate values at t=1
s1L_values = np.dot(s0L, values)
s1R_values = np.dot(s0R, values)

### 3. printing the results

print(s0L_value + s1L_values)
print(s0R_value + s1R_values)
