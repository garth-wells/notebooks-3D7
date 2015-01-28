
# coding: utf-8

## Shape functions (basis functions)

# The notebook explores the computation of finite element shape functions. We start with the one-dimensional case.
# 
# We will use NumPy to compute the shape functions, and Matplotlib to visualise the shape functions, so we need to import both:

# In[2]:

import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


### Lagrange polynomials in 1D

# We start with the exmaple of a cubic finite element basis, and then develop a function for plotting shape functions of any order. In all cases we consider the interval $(-1, 1)$.

#### Cubic basis

# Cubic shape functions will have the form:
# 
# $$
# N_{i}(x) = c_{0} + c_{1} x + c_{2} x^{2} + c_{2}x^{3}
# $$
# 
# Recall the shape function $N_{i}$ should be equal to one at its own node ($N_{i}(x_{i}) = 1$) and zero at all other nodes ($N_{i}(x_{i}) = 0$ when  $i \ne j$).
# 
# The cubic function has four coefficients, so we need four nodes. First step is to create the nodes on the interval $(-1, 1)$. We will consider equally spaced nodes, in which case we can use the `linspace` function:

# In[3]:

x_n = np.linspace(-1.0, 1.0, 4)
print(x_n)


# Next, we construct the Vandermonde matrix for a third-order polynomial and the points `x_n`: 

# In[4]:

A = np.vander(x_n, 4)
print A


# We can now compute the four shape functions by solving $\boldsymbol{A} \boldsymbol{c}_{i} = \boldsymbol{f}_{i}$ to get the polynomial coefficients $\boldsymbol{c}_{i}$ for the shape function $N_{i}$. For node $i$, $f_{j=1} = 1$ if $i=j$ and $f_{j} = 0$ if $i \ne j$. We use a loop to compute the four shape functions at once:

# In[5]:

shape_functions = []
for i in range(4):
    f = np.zeros(4)
    f[i] = 1.0
    c = np.linalg.solve(A, f)
    print c
    shape_functions.append(np.poly1d(c))
    print("-Shape function for node {}: \n{}".format(i, shape_functions[-1]))


# We can now plot each the shape functions (we compute each shape function at 200 points to plot the function).

# In[6]:

# Evaluate the polynomial at the points
x = np.linspace(-1.0, 1.0, 200) 
plt.plot(x_n, np.zeros(4), '-o', color='k');
for shape_function in shape_functions:
    N = shape_function(x)
    plt.plot(x, N);


# We can use NumPy to compute the derivatives of the shape function, and then plot these.

# In[7]:

x = np.linspace(-1.0, 1.0, 200) 
plt.plot(x_n, np.zeros(len(x_n)), '-o', color='k');
for shape_function in shape_functions:
    dshape_function = np.polyder(shape_function)
    dN = dshape_function(x)
    plt.plot(x, dN);


#### Arbitary degree Lagrange polynomials

# We now write a function that performs the above tasks so we can compute and plot shape functions on any degree. The argument to the function, `n`, is the polynomial degree of the shape functions that we wish to compute. 

# In[8]:

def plot_lagrange(n):
    n = n + 1 # number of nodes
    x_n = np.linspace(-1.0, 1.0, n)
    A = np.vander(x_n, len(x_n))
    f = np.zeros(n)

    shape_functions = []
    x = np.linspace(-1.0, 1.0, 200) 
    plt.plot(x_n, np.zeros(len(x_n)), '-o', color='k');
    for i in range(n):
        f = np.zeros(n)
        f[i] = 1.0
        c = np.linalg.solve(A, f)
        plt.plot((x_n, x_n), (0.0, 1.0), '--', color='k');

        p = np.poly1d(c)
        N = p(x)
        plt.plot(x, N);


# For a $5$th order polynomial:

# In[9]:

plot_lagrange(5) 


# For a $10$th order polynomial:

# In[10]:

plot_lagrange(10) 


# For the $10$th order polynomial, not the oscillations near the ends of the element. This is known and Runge's phenomena when interpolating points with a polynomial. This element, with equally spaces nodes, wouldn't be recommended in a simulation.

### Hermitian shape functions

# In[11]:

def construct_hermitian_matrix(n, x):
    matrix = np.zeros(n)
    k = 0;
    for i in x:
        row = []
        for j in range(n,0,-1):
            row.extend([i**(j-1)])
        matrix = np.vstack((matrix, row))
        row = []
        for j in range(n,0,-1):
            if j==1: 
                row.extend([0])
            else:
                row.extend([(j-1)*i**(j-2)])
        matrix = np.vstack((matrix, row))
        
    matrix = matrix[1:, 0:]
    return matrix


# In[12]:

def plot_hermitian(n):
    x_n = np.linspace(-1.0, 1.0, n)
    A = construct_hermitian_matrix(2*n, x_n)
    print A
    for i in range(2*n):
        f = np.zeros(2*n)
        f[i] = 1
        c = np.linalg.solve(A,f)
        x = np.linspace(-1.0, 1.0, 200) 
        p = np.poly1d(c)
        plt.plot((x_n, x_n), (0.0, 1.0), '--', color='k');
        N = p(x)
        plt.plot(x, N)


# In[13]:

plot_hermitian(3)


# Hermitian using numpy polynomial differentiation. 

# In[14]:

def plot_hermitian_1(n):
    x_n = np.linspace(-1.0, 1.0, n)
    
    # Define polynomials
    p = np.poly1d(np.ones(2*n))
    px = np.polyder(p)
    
    # Construct matrix A
    A = np.zeros(2*n) # Initilise the first row of A with 0s. This line will be deleted later on
    for i in range(n):
        rows_p = [] # Initialise rows storing terms of p
        rows_px = [] # Initialise rows storing terms of px
        for k in range(2*n, 0, -1):
            rows_p.append(p[k-1]*pow(x_n[i],(k-1)))
            if (x_n[i]==0) and (k<2): # This condition here is needed as Python does not allow 0 to power of negative number
                rows_px.append(0)
            else:
                rows_px.append(px[k-2]*pow(x_n[i],(k-2)))
        A = np.vstack((A, rows_p))
        A = np.vstack((A, rows_px))
    A = A[1:, 0:]
    
    for i in range(2*n):
        f = np.zeros(2*n)
        f[i] = 1
        c = np.linalg.solve(A,f)
        x = np.linspace(-1.0, 1.0, 200) 
        p = np.poly1d(c)
        plt.plot((x_n, x_n), (0.0, 1.0), '--', color='k');
        N = p(x)
        plt.plot(x, N)
        
plot_hermitian_1(3)


### Shape functions in two dimensions

# Testing

# In[ ]:



