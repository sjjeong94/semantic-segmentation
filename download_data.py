import os
import json
from tqdm import tqdm
from urllib import request


download_url = 'https://raw.githubusercontent.com/commaai/comma10k/master/'
imgs_url = download_url + 'imgs/'
masks_url = download_url + 'masks/'


def download_data(
    data_root='data/comma10k/'
):
    print('Download Data...')
    with open('comma10k.json', 'r') as json_file:
        file_names = json.load(json_file)
    names = []
    names.extend(file_names['train'])
    names.extend(file_names['val'])

    imgs_root = os.path.join(data_root, 'imgs')
    os.makedirs(imgs_root, exist_ok=True)
    masks_root = os.path.join(data_root, 'masks')
    os.makedirs(masks_root, exist_ok=True)

    for name in tqdm(names):
        img_url = imgs_url + name
        img_path = os.path.join(imgs_root, name)
        mask_url = masks_url + name
        mask_path = os.path.join(masks_root, name)
        request.urlretrieve(img_url, img_path)
        request.urlretrieve(mask_url, mask_path)


def download_data2(
    data_root='data/comma10k/',
):
    print('Download Data...')
    with open('comma10k.json', 'r') as json_file:
        file_names = json.load(json_file)
    names = []
    names.extend(file_names['train'])
    names.extend(file_names['val'])

    imgs_root = os.path.join(data_root, 'imgs')
    os.makedirs(imgs_root, exist_ok=True)
    masks_root = os.path.join(data_root, 'masks')
    os.makedirs(masks_root, exist_ok=True)

    request.urlretrieve(imgs_url, imgs_root)
    request.urlretrieve(masks_url, masks_root)


if __name__ == '__main__':
    download_data()
