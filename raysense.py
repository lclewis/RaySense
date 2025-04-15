import numpy as np
from sklearn.neighbors import NearestNeighbors
import faiss


def sph2cart(r, theta, phi):
    """Convert spherical coordinate to cartesian coordinate"""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])


def randomray(m, k, d):
    """Generate random rays of length roughly 1 by using normalized Gaussian
        Midpoints are in the [-1/2, 1/2]^d cube and points on rays are equi-spaced
        m = number of rays
        k = number of points on a ray
        d = embedding dimension
        return: pts_ray (m, k, d)
    """
    pi = np.pi
    pts_sph = np.random.randn(m,d) # random Gaussian
    
    if d > 100: #  Gaussian Annulus for high dim
        pts_sph /= np.sqrt(d) # normalized by radius of Gaussian s.t. ||pts_sph|| ~= 1
        pts_sph /= 2  # length 1
    else:
        pts_sph /= np.linalg.norm(pts_sph, axis = 1)[:,np.newaxis]
        pts_sph /= 2  # length 1

    center = np.random.rand(m, d)-0.5 # random uniform centers

    end1 = center - pts_sph
    end2 = center + pts_sph

    pts_ray = np.zeros((m, k, d))

    for indx in range(d):
        pts_ray[:, :, indx] = np.array([np.linspace(i, j, num=k) for i, j in zip(end1[:, indx], end2[:, indx])])

    return pts_ray


def generate_signature(m, k, d, pointset, num_neighbor=1, add_cp=True, add_cpv=False, count_freq=True):
    '''
    Generate RaySense Signature for an arbitrary dimension point cloud
    :param m: number of rays
    :param k: number of samples along a ray
    :param d: dimension of a point cloud
    :param pointset: (n, d) input point cloud
    :param num_neighbor: number of nearest neighbor
    :param add_cp: T/F if add closet point coordinate
    :param add_cpv: T/F if add vector to closest point
    :param count_freq: T/F if compute frequency of sampling
    :return: smatrix: (c, k, m) signature matrix, c - feature dimension
    '''
    pts_ray = randomray(m, k, d)
    distances, indices = kNN_search_faiss(pointset, pts_ray.reshape((m * k, d)), k=num_neighbor)

    if count_freq:
        unique, counts = np.unique(indices, return_counts=True) # return the unique indices and its count
        salient_pts = pointset[unique, :]
    D_in = 0  # dimension of the feature channel in the signature matrix
    D_in += d*num_neighbor if add_cp else 0
    D_in += d*num_neighbor if add_cpv else 0

    smatrix = np.zeros((m, k, D_in), dtype=float)
    marker = 0
    # adding the closest point
    for indx in range(np.size(indices, 1)): # add loop for potential multiple neighbors
        if add_cp:
            smatrix[:, :, marker:marker + d] = pointset[indices[:, indx], :].reshape(m, k, d)
            marker += d
            
    # adding the vector to the closest point
    for indx in range(np.size(indices, 1)): #add loop for potential multiple neighbors
        if add_cpv:
            direction = pointset[indices[:, indx], :].reshape(m, k, d) - pts_ray
            smatrix[:, :, marker:marker + d] = direction
            marker += d
            
    smatrix = smatrix.transpose(2, 1, 0)  # the first dimension is c, second is k, third is m
    return (smatrix, salient_pts, counts) if count_freq else smatrix


def kNN_search_faiss(data_target, data_source, k):
    """
    Function that uses FAISS to find the k nearest neighbor
    :param data_target (N-by-d): the target point cloud data to search for
    :param data_source (N-by-d): the source point cloud to search from (find kNN for each of its memeber)
    :param k: number of neighbors to look for
    :return [Dist, Indx]:
        Indx: N-k array contains indices of the k nearest neighbor in increasing distance
        np.sqrT(Dist): N-k array contains distances of the k nearest neighbor in increasing distance
                    The distance reported by FAISS is squared!!
    """
    data = data_target.astype('float32').copy(order='C') # convert to C-order
    index = faiss.IndexFlatL2(data.shape[1]) # arg: dimension of the data point
    index.add(data)  # add vectors to the index
    
    ### kNN search under FAISS
    # The result of this operation can be conveniently
    # stored in an integer matrix of size nq-by-k,
    # where row i contains the IDs of the neighbors of
    # query vector i, sorted by increasing distance.
    # In addition to this matrix, the search operation
    # returns a nq-by-k floating-point matrix with the
    # corresponding squared distances.
    
    Dist, Indx = index.search(data_source.astype(np.float32), k) # k number of neighbors
    return np.sqrt(Dist), Indx



