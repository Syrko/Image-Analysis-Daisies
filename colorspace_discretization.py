from sklearn.cluster import KMeans
import pickle
import numpy as np
from os import path


def fit_and_save_kmeans(lab_images):
    if path.exists("fitted_kmeans.sav"):
        print("Fitted kmeans already saved -- If you wish to fit again stop execution, delete the file and rerun!")
        return
    print("Fitting kmeans...")
    kmeans = KMeans(n_clusters=16)
    # Reshaping lab images as feature vector
    ab_values = []
    for lab_image in lab_images:
        lab_image = lab_image.reshape((lab_image.shape[0] * lab_image.shape[1]), 3)
        lab_image = np.delete(lab_image, 0, 1)
        ab_values.extend(lab_image)
    ab_values = np.array(ab_values)
    fitted = kmeans.fit(ab_values)

    print("Done fitting!")
    pickle.dump(fitted, open("fitted_kmeans.sav", 'wb'))
    print("Saved kmeans!")


def load_kmeans():
    loaded_kmeans = pickle.load(open("fitted_kmeans.sav", 'rb'))
    return loaded_kmeans
