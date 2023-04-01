#Nathan Carney
#COT4500, Assignment 3
import numpy as np

# 1. Euler Method with the following details 
# a. Function: t – y2 
# b. Range: 0 < t < 2 
# c. Iterations: 10 
# d. Initial Point: f(0) = 1 
def euler_method(f, a, b, n, y0):
    # calculate step size
    h = (b-a)/n
    # initialize variables
    t, y = a, y0
    # use Euler's Method to approximate solution
    for i in range(1, n+1):
        y += h * f(t, y)
        t += h
    # return the final approximation
    return y


# Define the differential equation to approximate
def func(t, y):
    return t - y**2


# Set the interval and initial condition
a, b = 0, 2
n = 10
y0 = 1

# Approximate the solution using Euler's Method
approximation = euler_method(func, a, b, n, y0)

# Print the result
print(f"{approximation:.5f}")
print()




# 2. Runge-Kutta with the following details: 
# a. Function: t – y2 
# b. Range: 0 < t < 2 
# c. Iterations: 10 
# d. Initial Point: f(0) = 1 
def runge_kutta(f, a, b, n, y0):
    # Calculate step size
    h = (b-a)/n
    
    # Initialize time and function value
    t, y = a, y0
    
    # Run Runge-Kutta method
    for i in range(1, n+1):
        # Calculate four k values
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        
        # Calculate y value for next step
        y += (k1 + 2*k2 + 2*k3 + k4)/6
        
        # Update time for next step
        t += h

    return y

def func(t, y):
    # Define function to be solved
    return t - y**2

# Set initial conditions and solve using Runge-Kutta method
a, b = 0, 2
n = 10
y0 = 1
approximation = runge_kutta(func, a, b, n, y0)

# Print result
print(f"{approximation:.5f}")
print()


# 3. Use Gaussian elimination and backward substitution solve the following linear system of 
# equations written in augmented matrix format.  

# Define the augmented matrix
A = np.array([[2, -1, 1, 6],
              [1, 3, 1, 0],
              [-1, 5, 4, -3]]).astype(float)

# Apply Gaussian elimination
for i in range(len(A)):
    # Find the pivot row
    pivot_row = i
    for j in range(i+1, len(A)):
        if abs(A[j,i]) > abs(A[pivot_row,i]):
            pivot_row = j
    # Swap the pivot row with the current row
    A[[i,pivot_row]] = A[[pivot_row,i]]
    # Eliminate the variable from the remaining rows
    for j in range(i+1, len(A)):
        factor = A[j,i] / A[i,i]
        A[j] -= factor * A[i]

# Apply backward substitution
x = np.zeros(len(A))
for i in range(len(A)-1, -1, -1):
    x[i] = (A[i,-1] - np.dot(A[i,:-1], x)) / A[i,i]

# Print the solution
print(f"{x}")
print()



# 4. Implement LU Factorization for the following matrix and do the following: 
# 1 1 0 3
# 2 1 −1 1
# 3 −1 −1 2
# −1 2 3 −1
# a. Print out the matrix determinant.  
# b. Print out the L matrix. 
# c. Print out the U matrix.


def LU_decomposition(A):
    # Create empty matrices with same size as A
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    # Get the size of the square matrix
    N = np.size(A, 0)

    # Perform LU decomposition
    for k in range(N):
        # Set the diagonal entries of L to 1
        L[k, k] = 1
        # Compute the diagonal entries of U using the formula
        U[k, k] = (A[k, k] - np.dot(L[k, :k], U[:k, k])) / L[k, k]
        # Compute the entries of U above the diagonal using the formula
        for j in range(k+1, N):
            U[k, j] = (A[k, j] - np.dot(L[k, :k], U[:k, j])) / L[k, k]
        # Compute the entries of L below the diagonal using the formula
        for i in range(k+1, N):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    # Return the lower and upper triangular matrices
    return L, U

# Define matrix
A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]]).astype(float)

# Call the function and get L and U matrices
L, U = LU_decomposition(A)

# Print L and U matrices
print(L, "\n")
print(U, "\n")


# 5. Determine if the following matrix is diagonally dominate. 
# 9 0 5 2 1
# 3 9 1 2 1
# 0 1 7 2 3
# 4 2 3 12 2
# 3 2 4 0 8

# Define the matrix
A = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]])

# Check if the matrix is diagonally dominant
is_diagonally_dominant = True
for i in range(A.shape[0]):
    # Calculate the row sum, which is the sum of the 
    # absolute values of all the elements in the row except the diagonal element
    row_sum = np.sum(np.abs(A[i,:])) - np.abs(A[i,i])
    
    # Check if the diagonal element is less than the row sum,
    #  which means the matrix is not diagonally dominant
    if np.abs(A[i,i]) < row_sum:
        is_diagonally_dominant = False
        break

# Print the result
if is_diagonally_dominant:
    print("True")
else:
    print("False")
print()

 
# 6. Determine if the matrix is a positive definite. 
# 2 2 1
# 2 3 0
# 1 0 2


# Define the matrix
A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])

# Check the leading principal minors
is_positive_definite = True
for k in range(1, A.shape[0]+1):
    minor = A[:k, :k]
    if np.linalg.det(minor) <= 0:
        is_positive_definite = False
        break

# Print the result
if is_positive_definite:
    print("True")
else:
    print("False")

