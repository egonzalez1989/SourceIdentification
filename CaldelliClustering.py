import glob, os, copy, time
from DenoiseTechniques import *

def zero_mean(data):
    '''
    Subtract row-column mean value to center data around 0
    :param data: image data
    :return:
    '''
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

def noise_extraction(denoise_func, inpath, outpath = None, replace = False):
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
    try:
        [os.makedirs(os.path.join(outpath, f)) for f in ['', 'noise', 'enhanced']]
    except:
        print('File exists: ', os.path.join(outpath))
    files = glob.glob(inpath + '/*')
    for f in files:
        # Get the file name to store the noise files
        fname = os.path.basename(f)
        try:
            img = cv2.imread(f) / 255.0
            _, img, _ = cv2.split(img)
            # Noise extraction with DWT for PRNU estimation. Assume every file in inpath is an image, otherwise error will be thrown
            noise = img - denoise_func(img)
            # Store noise and enhanced noise to avoid several computations
            enhanced = cosine_enhancer(noise)
            np.save(outpath + '/noise/' + fname, noise)
            np.save(outpath + '/enhanced/' + fname, enhanced)
        except:
            print('Error reading: ', f)
    return

def corr_matrix(inpath, outfile = None):
    '''
    Get correlation matrix for hierarchical distance
    :param inpath: Directory with noise samples.
    :param outpath: If None, the correlation matrix is stored in tmp dir
    '''
    if outfile is None:
        outfile = 'tmp/corr.npy'
    # Get noise files from inpath dir
    files = sorted(glob.glob(inpath + '/*.npy'))
    nf = len(files)
    # Dissimilarity matrix computed as 1 - corr(noise[i], noise[j])
    C = np.ones((nf, nf))
    for i in range(nf-1 ):
        np1 = np.load(files[i])
        for j in range(i+1, nf):
            np2 = np.load(files[j])
            C[i, j] = C[j, i] = np.corrcoef(np1.flatten(), np2.flatten())[0,1]
    np.save(outfile, C)
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

#def cluster_corr_similarity(A, B, corr):
    '''
    Similarity between two clusters based on correlation matrix
    :param A: 
    :param B: 
    :param corr: 
    :return: 
    '''
#    m, n = len(A), len(B)
#    corr_ij = sum(corr[i,j] for j in B for i in A)
#    return corr_ij / (m*n)

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
        # a. Search for the pair of clusters with maximum similarity
        max_sim, max_i, max_j = -1, 0, 0
        M = H.shape[0]
        for i in range(M-1):
            for j in range(i+1, M):
                if H[i, j] > max_sim:
                    max_sim = H[i, j]
                    max_i, max_j = i, j
        # b. Delete from H the rows and columns referred to clusters Z <= (U,V)
        H = np.delete(np.delete(H, max_i, 0), max_j, 1)
        C[max_i].extend(C[max_j])         # Joining clusters with maximum similarity
        del C[max_j]
        CC.append(copy.deepcopy(C))
        SC.append(cluster_sc(C, corr))
        M = M-1
        # c. Update H calculating new similarities
        for i in range(M-1):
            for j in range(i+1, M):
                # Average correlation of clusters
                H[i, j] = H[j, i] = np.mean([corr[k,l] for l in C[j] for k in C[i]])

    return CC, SC


def caldelli_clustering(inpath, denoise = dwt_denoise, outpath = None,
                        n_clusters = 0, gen_noise = True, gen_corr = True):
    '''
    Implementation of Caldelli clustering using both noise only and enhanced noise
    :param inpath: Dataset path
    :param denoise: Denoise function to be applied
    :param outpath: Path to store results. If None is give, a temporal dir is genereted to store results
    :param n_clusters: Number of clusters to be produced
    :param gen_noise: If true it generates a dir with noise samples, otherwise it looks for them in the outpath dir
    :param gen_corr: If true it generates a correlation file, otherwise it looks for it in the outpath dir
    :return: Two different clusters, corresponding to correlation of noise only, and correlation
    of enhanced function.
    '''
    if outpath is None:
        outpath = 'tmp_' + str(time.time())
    # 1: Noise extraction
    if gen_noise:
        noise_extraction(denoise, inpath = inpath, outpath = outpath)

    # 2: Correlation Matrix for noise and enhanced noise
    noise_corr_file = os.path.join(outpath, 'noise_corr.npy')
    enhanced_corr_file = os.path.join(outpath, 'enhanced_corr.npy')
    if gen_corr:
        noise_corr = corr_matrix(os.path.join(outpath, 'noise'), noise_corr_file)
        enhanced_corr = corr_matrix(os.path.join(outpath, 'enhanced'), enhanced_corr_file)
    else:
        noise_corr = np.load(noise_corr_file)
        enhanced_corr = np.load(enhanced_corr_file)

    # 3: Create list of clusters and silhouette scores
    CC1, SC1 = generate_clusters(noise_corr)
    CC2, SC2 = generate_clusters(enhanced_corr)

    # Return according to the required number of clusters or the best silhouette score
    if n_clusters > 0:
        noise_clusters = CC1[-n_clusters]
        enhanced_clusters = CC1[-n_clusters]
    else:
        noise_clusters = CC1[SC1.index(min(SC1))]
        enhanced_clusters = CC2[SC2.index(min(SC2))]
    return noise_clusters, enhanced_clusters


