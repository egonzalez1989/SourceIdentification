import cv2, numpy as np
import glob, os, copy, time
from DenoiseTechniques import *

def zero_mean(data, ztype='both'):
    '''
    Subtract row mean value to center data around 0
    :param data: image data
    :param ztype:
    :return:
    '''
    if ztype == 'no':
        return data
    zm = data - np.mean(data, axis=0)
    zm = (zm.T - np.mean(zm, axis=1)).T
    return zm

def cosine_enhancer(noise, alpha=.055):
    '''
    Enhancement proposed by Caldelli et al based on cosine function
    '''
    cpnoise = np.copy(noise)
    # Greater than alpha
    gtalpha = cpnoise > alpha
    ltalpha = np.logical_not(gtalpha)
    # Greater than zero
    gtzero = cpnoise > 0
    ltzero = np.logical_not(gtzero)
    #
    gtmalpha = cpnoise > -alpha
    ltmalpha = np.logical_not(gtmalpha)

    cpnoise[gtalpha] = 0
    cpnoise[np.logical_and(gtzero, ltalpha)] = np.cos((np.pi / (2 * alpha)) * noise[np.logical_and(gtzero, ltalpha)])
    cpnoise[np.logical_and(ltzero, gtmalpha)] = -np.cos((np.pi / (2 * alpha)) * noise[np.logical_and(ltzero, gtmalpha)])
    cpnoise[ltmalpha] = 0
    return cpnoise

def noise_extraction(denoise_func, inpath, outpath = None):
    '''
    First step: Noise extraction applies the selected denoise function and
    generates noise files.

    :param denoise_func:
    :param inpath: Noisy image directory. Ensure that only image files can be found here
    :param outpath: If None, a temporal dir is created to store the noise files
    otherwise, the value is used to create a dir and store the files
    '''
    if outpath is None:
        outpath = 'tmp/'
    [os.mkdir(outpath + f) for f in ['', '/noise', '/enhanced']]
    files = glob.glob(inpath + '/*')
    for f in files:
        # Get the file name to store the noise files
        fname = f.split('/')[-1].split('.')[0]
        img = cv2.imread(f) / 255.0
        _, img, _ = cv2.split(img)
        # Noise extraction with DWT for PRNU estimation. Assume every file in inpath is an image, otherwise error will be thrown
        noise = img - denoise_func(img)
        # Store noise and enhanced noise to avoid several computations
        enhanced = cosine_enhancer(noise)
        np.save(outpath + '/noise/' + fname, noise)
        np.save(outpath + '/enhanced/' + fname, enhanced)
    return

def corr_matrix(inpath, outpath = None):
    '''
    Get correlation matrix for hierarchical distance
    :param inpath: Noisy image directory.
    :param outpath: If None, the correlation matrix is stored in tmp dir
    '''
    if outpath is None:
        outpath = 'tmp/corr.npy'
    # Get noise files from inpath dir
    files = sorted(glob.glob(inpath + '/*.npy'))
    nf = len(files)
    # Dissimilarity matrix computed as 1 - corr(noise[i], noise[j])
    C = np.ones((nf, nf))
    for i in range(nf-1):
        np1 = np.load(files[i])
        for j in range(i+1, nf):
            np2 = np.load(files[j])
            C[i, j] = C[j, i] = np.corrcoef(np1.flatten(), np2.flatten())[0,1]
    np.save(outpath, C)
    return C

def cluster_sc(C, corr):
    '''
    Computes the silhouette score  using the correlation matrix
    :param C: Cluster to be inspected
    :param corr: Correlation matrix
    :return: Global silhouette score for the cluster (average score)
    '''
    s = np.ones(corr.shape[0]) * np.inf
    for I in C:
        B = list(range(corr.shape[0])) # Elements outside the cluster
        for i in I: B.remove(i)
        for i in I:
            a = np.mean([corr[i, j] for j in I]) if len(I) > 1 else 0
            b = np.mean([corr[i, j] for j in B])
            s[i] = b - a
    SC = np.mean(s)
    return SC

def cluster_corr_similarity(A, B, corr):
    '''
    Similarity between two clusters based on correlation matrix
    '''
    m, n = len(A), len(B)
    corr_ij = sum(corr[i,j] for j in B for i in A)
    return corr_ij / (m+n)

def generate_clusters(corr):
    '''
    Optimal clustering algorithm for hierarchical distance
    :param C: Correlation matrix corr(n_i, n_j)
    :return: List of clusters and its corresponding silhoutte coefficients
    '''
    # 1. Initialization
    N = corr.shape[0]
    H = copy.deepcopy(corr)

    C = [[i] for i in range(N)] # Current cluster. Initialized as a single one
    CC = [copy.deepcopy(C)]                    # List of all possible clusters
    SC = [0]                     # Silhouette coefficients
    # 2. Loop over 1 <= q <= N-1
    for q in range(1, N):
        # 2.a. Search for the pair of clusters with maximum similarity
        max_sim, max_i, max_j = -1, 0, 0
        M = H.shape[0]
        for i in range(M-1):
            for j in range(i+1, M):
                if H[i, j] > max_sim:
                    max_sim = H[i, j]
                    max_i, max_j = i, j
        # 2.b. Delete from H the rows and columns referred to clusters Z <= (U,V)
        H = np.delete(np.delete(H, max_i, 0), max_j, 1)
        C[max_i].extend(C[max_j])         # Joining clusters with maximum similarity
        del C[max_j]
        CC.append(copy.deepcopy(C))
        SC.append(cluster_sc(C, corr))
        M = M-1
        for i in range(M-1):
            for j in range(i+1, M):
                H[i, j] = H[j, i] = cluster_corr_similarity(C[i], C[j], corr)
    return CC, SC

# Cluster technique applied to a set of images in
def caldelli_clustering(inpath, denoise, enhanced = True, outpath = None, n_clusters = 0):
    if outpath is None:
        outpath = 'tmp_' + str(time.time())
    # 1: Noise extraction
    noise_extraction(denoise, inpath = inpath, outpath = outpath)
    # 2: Correlation Matrix for noise and enhanced noise
    noise_corr_file = outpath + '/noise_corr.npy'
    enhanced_corr_file = outpath + '/enhanced_corr.npy'
    noise_corr = corr_matrix(outpath + '/noise', noise_corr_file)
    enhanced_corr = corr_matrix(outpath + '/enhanced', enhanced_corr_file)
    # 3: Create clusters
    CC1, SC1 = generate_clusters(noise_corr)
    CC2, SC2 = generate_clusters(enhanced_corr)
    if n_clusters > 0:
        min_clusters1 = CC1[-n_clusters]
        min_clusters2 = CC1[-n_clusters]
    else:
        min_clusters1 = CC1[SC1.index(min(SC1))]
        min_clusters2 = CC2[SC2.index(min(SC2))]
    return min_clusters1, min_clusters2


