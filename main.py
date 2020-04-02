import image_functions as imf
import colorspace_discretization as csd
import slic_superpixels as ssp
import extract_features as ef
import svm
from os import path
import numpy as np

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    SUPERPIXELS_NUM = 100
    TESTING_IMAGE = "Dataset\\Testing.png"

    print("TRAINING")

    # Load train image of castle
    rgb_images = []
    lab_images = []

    # Loading all training images
    for i in range(1, 14):
        temp_image = imf.load_rgb_image("Dataset\\" + str(i) + ".png")
        rgb_images.append(temp_image)
        temp_image = imf.rgb_to_lab(temp_image)
        lab_images.append(temp_image)

    # Fit kmeans and save it with pickle
    csd.fit_and_save_kmeans(lab_images)

    if path.exists("SVC.sav"):
        print("Fitted SVM already saved -- If you wish to fit again stop execution, delete the file and rerun!")
    else:
        slic_list = []
        surf_list = []
        gabor_list = []

        for img in rgb_images:
            # SLIC segmentation and feature extraction per superpixel
            slic = ssp.get_slic_superpixels(img, superpixels_number=SUPERPIXELS_NUM)
            slic_list.extend(slic)

            surf = ef.extract_surf_for_each_superpixel(slic)
            surf_list.extend(surf)

            gabor = ef.extract_gabor_for_each_superpixel(slic)
            gabor_list.extend(gabor)

        # Fitting and saving SVM
        svm.fit_and_save_svm(slic_list, surf_list, gabor_list)

    print("===========================================================================================================")
    print("TESTING")

    # Load fitted models
    svc = svm.load_svm()
    kmeans = csd.load_kmeans()

    # Load test image of castle and make it grayscale
    colored_img = imf.load_rgb_image(TESTING_IMAGE)
    gray_img = imf.rgb_to_gray(colored_img)

    # SLIC segmentation and feature extraction per superpixel
    slic = ssp.get_slic_superpixels(gray_img, gray=True, superpixels_number=SUPERPIXELS_NUM)
    surf = ef.extract_surf_for_each_superpixel(slic, True)
    gabor = ef.extract_gabor_for_each_superpixel(slic, True)

    print("Preparing test samples...")
    test_sample = svm.prepare_test_sample(slic, surf, gabor)
    print("Test samples prepared!")
    predictions = svc.predict(test_sample)
    print("Colors predicted!")
    colors = kmeans.cluster_centers_.astype("uint8")[predictions]

    imf.display_end_result(colored_img, gray_img, colors)
