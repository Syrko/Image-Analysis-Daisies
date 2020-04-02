import urllib.request
from urllib.error import HTTPError
from pathlib import Path


def read_image_urls():
    file = open("dataset_urls.txt", 'r')
    url_list = file.readlines()
    file.close()
    return url_list


def download_dataset():
    try:
        Path("Dataset").mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Dataset downloaded already! If it is not delete 'Dataset' folder and re-run the program.")
        return

    url_list = read_image_urls()
    counter = 1
    image_counter = 0
    for url in url_list:
        try:
            urllib.request.urlretrieve(url[:-1], Path("Dataset", "image" + str(counter) + ".jpg"))
            counter += 1
            image_counter += 1
        except HTTPError as e:
            print("Error " + str(e.code) + " -- " + url[:-1])
            continue
        except:
            print("Error -- " + url[:-1])
            continue
    print("\nDownloaded " + str(image_counter) + " images!")