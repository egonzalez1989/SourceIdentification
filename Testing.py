import os.path

from skimage.graph import central_pixel

from CaldelliClustering import *
from sklearn import metrics

DENOISE_MAP = {#'Noise_Visu_L4': lambda X: dwt_denoise(X, level=4, shrinkage='VisuShrink'),}
               #'Noise_Bayes_L4': lambda X: dwt_denoise(X, level=4, shrinkage='BayesShrink'),
               'Noise_TV_L01': lambda X: tv_denoise(X)}
CROP = 0

  ################################
 ###  PREPROCESSING UTILITIES  ##
################################
def create_test(inpath, outpath, replace = False):
    '''
    Creates a dataset for testing purposes. Sources are known and images are included within a
    folder with the device or source name, and possibly additional information
    :param inpath: Main directory containing subdirectories for each source device
    :param outpath: Directory where to save the unified dataset. The name of source device (subdir name)
    will be used as a prefix for evaluation
    :param replace: Replaces (or not) a previous dataset, if exists
    :return:
    '''
    if os.path.isdir(outpath) and not replace:
        return
    elif not os.path.isdir(outpath):
        os.makedirs(outpath)
    subdirs = os.listdir(inpath)
    for subdir in subdirs:
        # Copy and move files
        files = glob.glob(os.path.join(inpath, subdir, '*'))
        for file in files:
            outfile = os.path.join(outpath, subdir + '_' + os.path.basename(file))
            img = cv2.imread(file)
            cv2.imwrite(outfile, img)
    return

def crop_images(inpath, outpath, crop_size, location = 'center'):
    '''
    Create a cropped version of the dataset, to unify the image dimensions.
    :param inpath: Dataset to crop
    :param outpath: Cropped dataset
    :param crop: Size of the resulting cropped dataset
    :param location: c = center, tl = top-left, bl = bottom-left, tr = top-right, br = bottom-right
    :return:
    '''
    img_files = glob.glob(os.path.join(inpath, '*'))
    for f in img_files:
        fname = os.path.basename(f)
        img = cv2.imread(f)
        h, w = img.shape[:2]
        cs = crop_size
        # Verify that crop_size is smaller
        if cs > h or cs > w:
            raise Warning("Image {} is smaller than crop size".format(fname))
            pass
        # Crop according to chosen region
        if location == 'c': img = img[(h - cs) // 2: (h + cs) // 2, (w - cs) // 2: (w + cs) // 2, :]
        elif location == 'tl': img = img[: cs, : cs, :]
        elif location == 'tr': img = img[: cs, -cs:, :]
        elif location == 'bl': img = img[-cs: , :cs, :]
        elif location == 'br': img = img[-cs:, -cs:, :]
        # Save cropped
        cv2.imwrite(os.path.join(outpath, fname), img)

def evaluate(dataset, result_file):
    '''
    TODO: Evaluation of the results obtained
    :param dataset:
    :param result_file:
    :return:
    '''
    pass

def run_test(inpath, outpath, gen_noise = True, gen_corr = True, replace = True):
    # Crop images
    if CROP > 0:
        outpath = outpath + '_{}'.format(CROP)
        # If directory exists this block is not executed
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        crop_images(inpath, outpath, CROP)

    for k in DENOISE_MAP:
        # Directory for clustering results for each possible denoise implementation
        testpath = os.path.join(outpath, k)
        test_exists = os.path.isdir(testpath)
        if not test_exists or (test_exists and replace):
            noise_clusters, enh_clusters = caldelli_clustering(inpath, DENOISE_MAP[k],
                        outpath=testpath, gen_noise=gen_noise, gen_corr=gen_corr, n_clusters=4)
            noise_clusters = [sorted(c) for c in noise_clusters]
            enh_clusters = [sorted(c) for c in enh_clusters]

            # Store found clusters
            # Pure noise
            outfile = os.path.join(testpath, 'noise_clusters.txt')
            f = open(outfile, 'w')
            f.write(str(noise_clusters))
            f.close()
            # Enhanced noise
            outfile = os.path.join(testpath, 'enhanced_clusters.txt')
            f = open(outfile, 'w')
            f.write(str(enh_clusters))
            f.close()

            # Evaluate results
            #evaluate(outpath, results)

if __name__ == '__main__':
    # Create test dataset. All images in subdirectories are joint in a single direrctory
    imgpath = '/home/edgar/Documents/ImageForensics/Datasets/RTD_Pristine'
    dataset = '/home/edgar/Documents/ImageForensics/ClusterTests/RTD/Dataset'
    #create_test(imgpath, dataset, replace = False)

    # Testing on a single dataset, using default values
    outpath = '/home/edgar/Documents/ImageForensics/ClusterTests/RTD/Results/Noise_NeighVar_L4'
    noise_clusters, enh_clusters = caldelli_clustering(dataset, outpath=outpath, n_clusters=4)

    # Running a more complex test, with several denoising techniques
    outpath = '/home/edgar/Documents/ImageForensics/ClusterTests/RTD/Results'
    #run_test(dataset, outpath, replace = True)

    #TODO: Evaluate results with accurate metrics
