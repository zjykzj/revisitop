# -*- coding: utf-8 -*-

import os
import urllib.request
import tarfile


def download_datasets(data_root):
    """
    Download data to
    1. ${data_root}/datasets/roxford5k
    2. ${data_root}/datasets/rparis6k
    """
    datasets_dir = os.path.join(data_root, 'datasets')
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)

    dataset_list = ['roxford5k', 'rparis6k']
    for dataset_name in dataset_list:
        if 'roxford5k' == dataset_name:
            remote_url = 'https://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
            dl_files = ['oxbuild_images-v1.tgz']
        elif 'rparis6k' == dataset_name:
            remote_url = 'https://www.robots.ox.ac.uk/~vgg/data/parisbuildings'
            dl_files = ['paris_1-v1.tgz', 'paris_2-v1.tgz']
        else:
            raise ValueError('Unknown dataset: {}!'.format(dataset_name))

        dst_dir = os.path.join(datasets_dir, dataset_name, 'jpg')
        if not os.path.isdir(dst_dir):
            print('>> Dataset {} directory does not exist. Creating: {}'.format(dataset_name, dst_dir))
            os.makedirs(dst_dir)

        for dl_file_name in dl_files:
            src_file = os.path.join(remote_url, dl_file_name)
            dst_file = os.path.join(dst_dir, dl_file_name)

            print('>> Downloading dataset {} archive {}...'.format(dataset_name, dl_file_name))
            os.system('wget {} -O {}'.format(src_file, dst_file))

            print('>> Extracting dataset {} archive {}...'.format(dataset_name, dl_file_name))
            # create tmp folder
            dst_dir_tmp = os.path.join(dst_dir, 'tmp')
            os.system('mkdir {}'.format(dst_dir_tmp))
            # extract in tmp folder
            os.system('tar -zxf {} -C {}'.format(dst_file, dst_dir_tmp))
            # remove all (possible) subfolders by moving only files in dst_dir
            os.system('find {} -type f -exec mv -i {{}} {} \\;'.format(dst_dir_tmp, dst_dir))
            # remove tmp folder
            os.system('rm -rf {}'.format(dst_dir_tmp))

            print('>> Extracted, deleting dataset {} archive {}...'.format(dataset_name, dl_file_name))
            os.system('rm {}'.format(dst_file))

        gnd_src_dir = os.path.join('http://cmp.felk.cvut.cz/revisitop/data', 'datasets', dataset_name)
        gnd_dst_dir = os.path.join(data_root, 'datasets', dataset_name)
        gnd_dl_file = 'gnd_{}.pkl'.format(dataset_name)
        gnd_src_file = os.path.join(gnd_src_dir, gnd_dl_file)
        gnd_dst_file = os.path.join(gnd_dst_dir, gnd_dl_file)
        if not os.path.exists(gnd_dst_file):
            print('>> Downloading dataset {} ground truth file...'.format(dataset_name))
            os.system('wget {} -O {}'.format(gnd_src_file, gnd_dst_file))


def download_distractors(data_root):
    """
    Download data to
    1. ${data_root}/datasets/revisitop1m/jpg
    2. ${data_root}/datasets/revisitop1m/revisitop1m.txt
    """
    # Create datasets folder if it does not exist
    datasets_dir = os.path.join(data_root, 'datasets')
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)

    dataset = 'revisitop1m'
    nfiles = 100
    src_dir = 'http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg'
    dl_files = 'revisitop1m.{}.tar.gz'
    dst_dir = os.path.join(data_root, 'datasets', dataset, 'jpg')
    dst_dir_tmp = os.path.join(data_root, 'datasets', dataset, 'jpg_tmp')
    if not os.path.isdir(dst_dir):
        print('>> Dataset {} directory does exist.\n>> Manually deleting: {}'.format(dataset, dst_dir))
        return
    print('>> Dataset {} directory does not exist.\n>> Creating: {}'.format(dataset, dst_dir))
    if not os.path.isdir(dst_dir_tmp):
        os.makedirs(dst_dir_tmp)

    for dfi in range(nfiles):
        dl_file = dl_files.format(dfi + 1)
        src_file = os.path.join(src_dir, dl_file)
        dst_file = os.path.join(dst_dir_tmp, dl_file)
        dst_file_tmp = os.path.join(dst_dir_tmp, dl_file + '.tmp')
        if os.path.exists(dst_file):
            print('>> [{}/{}] Skipping dataset {} archive {}, already exists...'.
                  format(dfi + 1, nfiles, dataset, dl_file))
            continue

        while True:
            try:
                print(
                    '>> [{}/{}] Downloading dataset {} archive {}...'.format(dfi + 1, nfiles, dataset, dl_file))
                urllib.request.urlretrieve(src_file, dst_file_tmp)
                os.rename(dst_file_tmp, dst_file)
                break
            except:
                print('>>>> Download failed. Try this one again...')

    for dfi in range(nfiles):
        dl_file = dl_files.format(dfi + 1)
        dst_file = os.path.join(dst_dir_tmp, dl_file)
        print('>> [{}/{}] Extracting dataset {} archive {}...'.format(dfi + 1, nfiles, dataset, dl_file))
        tar = tarfile.open(dst_file)
        tar.extractall(path=dst_dir_tmp)
        tar.close()
        print('>> [{}/{}] Extracted, deleting dataset {} archive {}...'.format(dfi + 1, nfiles, dataset, dl_file))
        os.remove(dst_file)
    # rename tmp folder
    os.rename(dst_dir_tmp, dst_dir)

    # download image list
    gnd_src_dir = 'http://ptak.felk.cvut.cz/revisitop/revisitop1m/'
    gnd_dst_dir = os.path.join(data_root, 'datasets', dataset)
    gnd_dl_file = '{}.txt'.format(dataset)
    gnd_src_file = os.path.join(gnd_src_dir, gnd_dl_file)
    gnd_dst_file = os.path.join(gnd_dst_dir, gnd_dl_file)
    if not os.path.exists(gnd_dst_file):
        print('>> Downloading dataset {} image list file...'.format(dataset))
        urllib.request.urlretrieve(gnd_src_file, gnd_dst_file)


def download_features(data_root):
    """
    Download data to
    1. ${data_root}/features/roxford5k_resnet_rsfm120k_gem.mat
    2. ${data_root}/features/rparis6k_resnet_rsfm120k_gem.mat
    """
    # Create features folder if it does not exist
    features_dir = os.path.join(data_root, 'features')
    if not os.path.isdir(features_dir):
        os.makedirs(features_dir)

    # Download example features
    dataset_list = ['roxford5k', 'rparis6k']
    for dataset_name in dataset_list:
        feat_src_dir = os.path.join('http://cmp.felk.cvut.cz/revisitop/data', 'features')
        feat_dst_dir = os.path.join(data_root, 'features')
        feat_dl_file = '{}_resnet_rsfm120k_gem.mat'.format(dataset_name)
        feat_src_file = os.path.join(feat_src_dir, feat_dl_file)
        feat_dst_file = os.path.join(feat_dst_dir, feat_dl_file)
        if not os.path.exists(feat_dst_file):
            print('>> Downloading dataset {} features file {}...'.format(dataset_name, feat_dl_file))
            os.system('wget {} -O {}'.format(feat_src_file, feat_dst_file))


if __name__ == '__main__':
    DATA_ROOT = os.path.join(os.getcwd(), "data")
    download_features(DATA_ROOT)
    download_datasets(DATA_ROOT)
