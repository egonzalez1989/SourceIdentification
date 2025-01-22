import glob

from CaldelliClustering import *

DENOISE_MAP = {'Noise_Visu_L4': lambda X: dwt_denoise(X, level=4, shrinkage='VisuShrink'),
               'Noise_Bayes_L4': lambda X: dwt_denoise(X, level=4, shrinkage='BayesShrink'),
               'Noise_TV_L01': lambda X: tv_denoise(X)}
CROP = 0
IMGPATH = '/home/edgar/Downloads/Forensics/TamperTests/TamperTest'
OUTPATH = '/home/edgar/Documents/ImageForensics/ClusterTests/Results/'
dirname = IMGPATH.split('/')[-1]
OUTPATH = OUTPATH + dirname

def create_test(inpath, outpath):
    # Test dataset must contain the different sources in independent subdirectires
    pass


def crop_images(inpath, outpath, crop):
    img_files = glob.glob(inpath + '/*')
    for f in img_files:
        fname = os.path.basename(f)
        img = cv2.imread(f)
        h, w = img.shape[:2]
        img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2, :]
        cv2.imwrite(outpath + fname, img)

def run_test(inpath, outpath):
    # Crop images
    if CROP > 0:
        outpath = outpath + '_{}'.format(CROP)
        # If directory exists this block is not executed
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        crop_images(inpath, outpath, CROP)

    for k in DENOISE_MAP:
        # Directory for clustering results
        outpath = outpath + '/' + k
        caldelli_clustering(inpath, DENOISE_MAP[k], outpath=outpath)

if __name__ == '__main__':
    run_test(IMGPATH, OUTPATH)