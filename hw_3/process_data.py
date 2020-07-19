'''
@Descripttion: 
@version: 
@Author: ErCHen
@Date: 2020-06-12 21:25:51
@LastEditTime: 2020-07-19 17:53:24
'''

import os
import numpy as np
import sys
import gc
import joblib
import random
from PIL import Image


# dump numpy array to disk using joblib
def save_batch(batch, label, output_dir, k):
    bath_name = os.path.join(output_dir, 'batch_%d'%k)
    label_name = os.path.join(output_dir, 'label_%d'%k)
    tmp_batch = np.array(batch)
    tmp_label = np.concatenate(label, axis=0)

    joblib.dump(tmp_batch, bath_name)
    joblib.dump(tmp_label, label_name)
    del tmp_batch
    del tmp_label
    del batch
    del label


def resize_img(path, size=256):
    img_raw = Image.open(path)
    img_raw = img_raw.resize((size,size))
    img_arr = np.array(img_raw, dtype=np.float32) / 255.0
    return img_arr


# convert the image to a numpy 3-D array
# with resizing and normalizing
def img2array(base_dir, output_dir, batch_size=30, img_size=256):
    if not os.path.exists(base_dir):
        print("%s does't exits " % base_dir)
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_path = os.listdir(base_dir)
    random.shuffle(image_path)
    batch = []
    label = []
    num = 0
    k = 0

    for img in image_path:
        tmp_label = np.zeros((1, 11),dtype=np.float32)
        idx = img.split('_')[0]
        tmp_label[0, int(idx)] = 1.
        # print('image %s ,label %s' %(img,idx))
        path = os.path.join(base_dir, img)
        # img_raw = Image.open(path)

        img_arr = resize_img(path, img_size)
        num = num + 1
        
        batch.append(img_arr)
        label.append(tmp_label)

        if num % batch_size == 0:
            save_batch(batch, label, output_dir, k)            
            batch = []
            label = []
            k += 1
            print('%d images are prosessed' % num)
            # break
        gc.collect()

    if len(batch) > 0:
        save_batch(batch, label, output_dir, k)
    print('All Done. %d batchs in total, please check out dir %s' % (k, output_dir))

    info_path = os.path.join(output_dir, 'info.txt')
    with open(info_path, 'w') as f:
        text = "image size:%d\nbatch size: %d\nbatches: %d\n" % (img_size, batch_size, k)
        f.write(text)


# get max and min size image of the data set        
def check(base_dir):
    size_set = set()
    image_path = os.listdir(base_dir)
    max_size = -1
    min_size = 50000 * 50000
    max_wh = ''
    min_wh = ''
    count = [0 for _ in range(11)]
    for img in image_path:
        path = os.path.join(base_dir, img)
        idx = img.split('_')[0]
        count[int(idx)] += 1
        img_arr = Image.open(path)
        w = img_arr.size[0]
        h  = img_arr.size[1]

        p = w * h
        if p < min_size:
            min_size = p
            min_wh = '%d_%d'%(w,h)
        
        if p > max_size:
            max_size = p
            max_wh = '%d_%d'%(w,h)

    print('max:%s' % max_wh)
    print('min:%s' % min_wh)
    print(count)
        # size_set.add('%d_%d'%(w,h))
    

    # for size in size_set:
    #     print(size)

def data_augmention(base_path, output_path):
    img_path = os.listdir(base_path)
    cluster = [ [] for _ in range(11) ]
    for img in img_path:
        idx = int(img.split('_')[0])
        cluster[idx].append(img)
    
    count = []
    for c in cluster:
        count.append(len(c))
    
    max_num = max(count)
    
    aug_k = 0
    for j in range(len(cluster)):
        for i in range(max_num - count[j]):
            target = random.choice(cluster[j])
            op = random.randint(0, 3)
            path = os.path.join(base_path, target)
            img_data = Image.open(path)
            img_aug = None
            if op == 0:
                img_aug = img_data.rotate(-30)
            elif op == 1:
                img_aug = img_data.rotate(30)
            elif op == 2:
                img_aug = img_data.transpose(Image.FLIP_LEFT_RIGHT)
            elif op == 3:
                img_aug = img_data.transpose(Image.FLIP_TOP_BOTTOM)
                
            if img_aug is not None:
                aug_path = '%d_aug%d.jpg' %(j, aug_k)
                aug_path = os.path.join(output_path, aug_path)
                img_aug.save(aug_path)
            aug_k += 1
    print('image augmentation is done!')


def process_test(test_base_dir, test_output_dir):
    img_path = os.listdir(test_base_dir)
    img_path = sorted(img_path)
    k = 0
    num = 0
    batch = []
    for path in img_path:
        path = os.path.join(test_base_dir, path)
        img_raw = Image.open(path)
        img_raw = img_raw.resize((128, 128))
        img = np.array(img_raw, dtype=np.float32) / 255.0
        k += 1
        batch.append(img)
        if k % 128 == 0:
            bath_name = os.path.join(test_output_dir, 'batch_%d'%num)
            tmp_batch = np.array(batch)
            joblib.dump(tmp_batch, bath_name)
            batch = []
            num += 1
            k = 0
    if len(batch) > 0:
        bath_name = os.path.join(test_output_dir, 'batch_%d'%num)
        tmp_batch = np.array(batch)
        joblib.dump(tmp_batch, bath_name)
        batch = []
    print('done')

if __name__ == "__main__":

    # check(base_dir)

    if len(sys.argv) < 5:
        # print(sys.argv)
        print('need more parameters')
        exit(0)
    base_dir = sys.argv[1]
    output_dir = sys.argv[2]
    batch_size = int(sys.argv[3])
    img_size = int(sys.argv[4])

