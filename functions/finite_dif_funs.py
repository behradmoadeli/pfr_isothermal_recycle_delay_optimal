def A_ode_fun(par, N_zeta):
    """
    Constructs a sparse matrix A_ode for a given set of parameters.

    Parameters:
    - par (dict): Dictionary containing parameters k, v, D, tau, and R.
    - N_zeta (int): Number of spatial discretization points.

    Returns:
    - A_ode (csr_matrix): Constructed sparse matrix.
    """
    
    from scipy.sparse import csr_matrix
    import numpy as np
    
    zeta = np.linspace(0, 1, N_zeta)
    dz = zeta[1]

    k, v, D, tau, R = par['k'], par['v'], par['D'], par['tau'], par['R']

    # Lists to hold data for the sparse matrix
    rows_data = [
        k - (2*D)/(dz**2) - (v)/(dz) - (v**2)/(2*D),
        (2*D)/(dz**2),
        R * ((v)/(dz) + (v**2)/(2*D))
    ]
    rows_row_indices = [0, 0, 0]
    rows_col_indices = [0, 1, N_zeta]

    # Fill the middle rows (from index 1 to N_zeta-2)
    for i in range(1, N_zeta-1):
        rows_data.extend([
            (D)/(dz**2) + (v)/(2*dz),
            k - (2*D)/(dz**2),
            (D)/(dz**2) - (v)/(2*dz)
        ])
        rows_row_indices.extend([i, i, i])
        rows_col_indices.extend([i-1, i, i+1])

    # Define the last row
    rows_data.extend([
        (2*D)/(dz**2),
        k - (2*D)/(dz**2)
    ])
    rows_row_indices.extend([N_zeta-1, N_zeta-1])
    rows_col_indices.extend([N_zeta-2, N_zeta-1])

    # Additional rows for the second block of the matrix
    for i in range(N_zeta, 2*N_zeta-2):
        rows_data.extend([
            -1/(tau*dz),
            1/(tau*dz)
        ])
        rows_row_indices.extend([i, i])
        rows_col_indices.extend([i, i+1])

    rows_data.extend([
        1/(tau*dz),
        -1/(tau*dz)
    ])
    rows_row_indices.extend([2*N_zeta-2, 2*N_zeta-2])
    rows_col_indices.extend([N_zeta-1, 2*N_zeta-2])

    rows_data.extend([
        (2*D)/(dz**2),
        k - (2*D)/(dz**2)
    ])
    rows_row_indices.extend([2*N_zeta-1, 2*N_zeta-1])
    rows_col_indices.extend([N_zeta-2, N_zeta-1])

    # Construct the sparse matrix
    A_ode = csr_matrix((rows_data, (rows_row_indices, rows_col_indices)), shape=(2*N_zeta, 2*N_zeta))

    return A_ode


def A_cl_fun(A_ode, par, K_controller):
    """
    Constructs the closed-loop system matrix A_cl.

    Parameters:
    - A_ode (csr_matrix): Sparse matrix representing the open-loop system dynamics.
    - par (dict): Dictionary containing parameters `k`, `v`, `D`, `tau`, and `R`.
    - K_controller (numpy.ndarray): Controller gain matrix.

    Returns:
    - A_cl (csr_matrix): Sparse matrix representing the closed-loop system dynamics.
    """
    
    from scipy.sparse import csr_matrix
    import numpy as np
    
    # Extract parameters from the dictionary
    k, v, D, tau, R = par['k'], par['v'], par['D'], par['tau'], par['R']

    # Infer N_zeta from the shape of A_ode
    N_zeta = A_ode.shape[0] // 2  # The matrix is 2*N_zeta x 2*N_zeta
    zeta = np.linspace(0, 1, N_zeta)
    dz = zeta[1]
    
    # Initialize the B matrix (input matrix)
    B = np.zeros((2 * N_zeta, 1))
    B[0, 0] = ((v)/(dz) + (v**2)/(2*D)) * (1-R)  # First element based on system dynamics

    # Initialize the C matrix (output matrix)
    C = np.zeros((1, 2 * N_zeta))
    C[0, int(N_zeta / 2 - 1)] = 1  # Place a 1 at the middle index in the first row

    # Construct the closed-loop system matrix
    # Note: B@K_controller performs matrix multiplication
    A_cl_matrix = A_ode + B @ K_controller

    # Convert the resulting matrix to a sparse format
    A_cl = csr_matrix(A_cl_matrix)

    return A_cl


def A_aug_fun(A_ode, par, K_controller, L_observer):
    """
    Constructs the augmented matrix A_aug for a control system with state feedback and an observer.

    Parameters:
    - A_ode (csr_matrix or array-like): Open-loop system dynamics matrix.
    - par (dict): Dictionary containing parameters `k`, `v`, `D`, `tau`, and `R`.
    - K_controller (csr_matrix or array-like): State feedback gain matrix.
    - L_observer (csr_matrix or array-like): Observer gain matrix.

    Returns:
    - A_aug (csr_matrix): Augmented sparse matrix for the closed-loop system.
    """

    import numpy as np
    from scipy.sparse import csr_matrix, hstack, vstack
    
    # Extract parameters from the dictionary
    k, v, D, tau, R = par['k'], par['v'], par['D'], par['tau'], par['R']

    N_zeta = A_ode.shape[0] // 2  # The matrix is 2*N_zeta x 2*N_zeta
    zeta = np.linspace(0, 1, N_zeta)
    dz = zeta[1]
    
    # Initialize the B matrix (input matrix)
    B = np.zeros((2 * N_zeta, 1))
    B[0, 0] = ((v)/(dz) + (v**2)/(2*D)) * (1-R)  # First element based on system dynamics
    
    # Initialize the C matrix (output matrix)
    C = np.zeros((1, 2 * N_zeta))
    C[0, int(N_zeta / 2 - 1)] = 1  # Place a 1 at the middle index in the first row
    
    B = csr_matrix(B)
    C = csr_matrix(C)

    if not isinstance(A_ode, csr_matrix):
        A_ode = csr_matrix(A_ode)
    if not isinstance(K_controller, csr_matrix):
        K_controller = csr_matrix(K_controller)
    if not isinstance(L_observer, csr_matrix):
        L_observer = csr_matrix(L_observer)

    # Compute submatrices for the augmented matrix
    A11 = csr_matrix(A_ode)
    A12 = csr_matrix(B @ K_controller)  # Interaction between B and K
    A21 = csr_matrix(L_observer @ C)   # Interaction between L and C
    A_est = csr_matrix(A_ode + B @ K_controller - L_observer @ C)  # Estimated dynamics

    # Assemble the augmented matrix
    A_top = hstack([A11, A12])  # Top half of the matrix
    A_bottom = hstack([A21, A_est])  # Bottom half of the matrix
    A_aug = vstack([A_top, A_bottom])  # Combine top and bottom halves

    # Ensure the final matrix is in CSR format
    if not isinstance(A_aug, csr_matrix):
        A_aug = csr_matrix(A_aug)

    return A_aug
