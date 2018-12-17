import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Problem data.
TOL = 1e-3
m1 = 3
m2 = 3
n = 2
np.random.seed(1)
length = 0.3
A = np.vstack([np.array([0, 0]) + length * (-0.5 + np.random.random((10, 2))),
               np.array([1, 1]) + length * (-0.5 + np.random.random((10, 2)))])
B = np.vstack([np.array([1, 0]) + length * (-0.5 + np.random.random((10, 2))),
               np.array([0, 1]) + length * (-0.5 + np.random.random((10, 2)))])
# A = np.vstack([A, np.array([1.2, -0.3])])
# B = np.vstack([B, np.array([-0.3, -0.3])])
xmin, xmax = -0.4, 1.4
ymin, ymax = -0.4, 1.25
# A = np.vstack([np.array([-1, -1]) + length * (-0.5 + np.random.random((10, 2))),
#                np.array([ 1,  1]) + length * (-0.5 + np.random.random((10, 2)))])
# B = np.vstack([np.array([ 1, -1]) + length * (-0.5 + np.random.random((10, 2))),
#                np.array([-1,  1]) + length * (-0.5 + np.random.random((10, 2)))])
# xmin, xmax = -1.5, 1.5
# ymin, ymax = -1.5, 1.5
m1 = A.shape[0]
m2 = B.shape[0]
H = np.hstack([A, np.ones((m1, 1))])
G = np.hstack([B, np.ones((m2, 1))])
D1_SR = cp.Parameter((m1, m1))
D2_SR = cp.Parameter((m2, m2))
c1 = cp.Parameter(nonneg=True)
c2 = cp.Parameter(nonneg=True)

c1.value = 0.001
c2.value = 0.001

# Construct the problem.
z1 = cp.Variable(n + 1)
q1 = cp.Variable(m2)
z2 = cp.Variable(n + 1)
q2 = cp.Variable(m1)
init_obj1 = cp.Minimize(cp.sum_squares(H * z1) / 2 + c1 * cp.sum(q1))
init_cons1 = [-G * z1 + q1 >= 1, q1 >= 0]
init_cons1.append(z1[0] == 1)
init_prob1 = cp.Problem(init_obj1, init_cons1)

init_obj2 = cp.Minimize(cp.sum_squares(G * z2) / 2 + c2 * cp.sum(q2))
init_cons2 = [H * z2 + q2 >= 1, q2 >= 0]
init_cons2.append(z2[0] == 1)
init_prob2 = cp.Problem(init_obj2, init_cons2)

objective1 = cp.Minimize(cp.sum_squares(D1_SR * H * z1) / 2 + c1 * cp.sum(q1))
constraints1 = [-G * z1 + q1 >= 1, q1 >= 0]
constraints1.append(z1[0] == 1)
prob1 = cp.Problem(objective1, constraints1)

objective2 = cp.Minimize(cp.sum_squares(D2_SR * G * z2) / 2 + c2 * cp.sum(q2))
constraints2 = [H * z2 + q2 >= 1, q2 >= 0]
constraints2.append(z2[0] == 1)
prob2 = cp.Problem(objective2, constraints2)

# Obtain the initial solution
init_prob1.solve()
z1_last = z1.value
print(z1.value)

init_prob2.solve()
z2_last = z2.value
print(z2.value)

# Plot points and planes
w1 = z1_last / np.linalg.norm(z1_last)
w2 = z2_last / np.linalg.norm(z2_last)
plane1 = np.array([[x, y] for x in np.arange(xmin, xmax, 0.005)
                          for y in np.arange(ymin, ymax, 0.005)
                          if abs(w1.dot(np.array([x, y, 1]))) < 0.001])
plane2 = np.array([[x, y] for x in np.arange(xmin, xmax, 0.005)
                          for y in np.arange(ymin, ymax, 0.005)
                          if abs(w2.dot(np.array([x, y, 1]))) < 0.001])
plt.subplot(121)
plt.scatter(A[:, 0], A[:, 1], color='r', label='Class 1')
plt.scatter(B[:, 0], B[:, 1], color='b', label='Class 2')
plt.plot(plane1[:, 0], plane1[:, 1], color='r', label='Plane 1')
plt.plot(plane2[:, 0], plane2[:, 1], color='b', label='Plane 2')
plt.title('TWSVM')
plt.legend()
# plt.show(); exit(0)

# Iterative procedure
while True:
    D1_SR.value = np.diag(1 / np.sqrt(np.abs(np.dot(H, z1_last))))
    prob1.solve()
    z1_cur = z1.value
    if np.linalg.norm(z1_cur - z1_last) < TOL:
        break
    else:
        z1_last = z1_cur.copy()

while True:
    D2_SR.value = np.diag(1 / np.sqrt(np.abs(np.dot(G, z2_last))))
    prob2.solve()
    z2_cur = z2.value
    if np.linalg.norm(z2_cur - z2_last) < TOL:
        break
    else:
        z2_last = z2_cur.copy()

print(z1_cur)
print(z2_cur)

z1_cur /= np.linalg.norm(z1_cur)
z2_cur /= np.linalg.norm(z2_cur)
plane1 = np.array([[x, y] for x in np.arange(xmin, xmax, 0.005)
                          for y in np.arange(ymin, ymax, 0.005)
                          if abs(z1_cur.dot(np.array([x, y, 1]))) < 0.002])
plane2 = np.array([[x, y] for x in np.arange(xmin, xmax, 0.005)
                          for y in np.arange(ymin, ymax, 0.005)
                          if abs(z2_cur.dot(np.array([x, y, 1]))) < 0.001])
# print(plane2.shape);exit(0)
plt.subplot(122)
plt.scatter(A[:, 0], A[:, 1], color='r', label='Class 1')
plt.scatter(B[:, 0], B[:, 1], color='b', label='Class 2')
plt.plot(plane1[:, 0], plane1[:, 1], color='r', label='Plane 1')
plt.plot(plane2[:, 0], plane2[:, 1], color='b', label='Plane 2')
plt.title('L1-TWSVM')
plt.legend()
plt.show()

# # The optimal Lagrange multiplier for a constraint is stored in
# # `constraint.dual_value`
# print(len(constraints))
# print(constraints[0].dual_value)
# print(constraints[1].dual_value)

