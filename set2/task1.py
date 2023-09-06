import numpy as np
import matplotlib.pyplot as plt


def pca(x: np.ndarray,
        one_d: bool,
        two_d: bool,
        three_d: bool):
    # 1) standardize
    x = (x - x.mean())/(x.std())
    # 2) covariance-matrix
    q = np.cov(np.transpose(x))
    # eigenvalues and vectors
    """
    Eigenvectors/values of cov-matrix indicate the spread of different variables(dims) in the data.
    (longer vec = larger spread)
    """
    e_val, e_vec = np.linalg.eig(q)
    print(f"Eigenvalues: {e_val}")
    if one_d:
        # plots for 1d-projections (projected onto different eigenvectors)
        for idx, e in enumerate(e_vec):
            temp_x = x @ e
            plt.scatter(temp_x, [0]*temp_x.shape[0])
            plt.title(e_val[idx])
            plt.show()
    if two_d:
        var = []
        # plots for 2d-projections (projected onto different eigenvectors-combinations)
        for a, b in [(0,1), (0,2), (1,2)]:
            temp_x = x @ np.transpose(np.stack([e_vec[a], e_vec[b]]))
            plt.scatter(*np.transpose(temp_x))
            plt.title(f"{e_val[a]} and {e_val[b]}")
            plt.show()
            var.append(temp_x.var())
        norm_var = np.array(var) / np.trace(q)
        # norm_var = var
        norm_eig_val = e_val / e_val.sum()
        # norm_var = np.linalg.norm(var)
        # norm_eig_val = np.linalg.norm(e_val)
        print(f"Normalized Variance: {norm_var}")
        print(f"Normalized Eigenvalues: {norm_eig_val}")
    if three_d:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*np.transpose(x))
        plt.show()



if __name__ == "__main__":
    data = np.loadtxt("data/fisher_iris_shuffled.txt")
    pca(data, True, True, three_d=False)

