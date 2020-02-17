import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np


def lenlen(number, name):
    name = str(name)
    while len(name) != number:
        result = '0' + name
        name = result
    return name


def mypsnr(noise, algorithm, dataset, sigma, img):
    number = 0
    tt1 = '0'
    tt2 = '0'
    ii1 = '0'
    ii2 = '0'
    dd = ''
    if dataset == 'BSD68':
        tt1 = 'test'
        tt2 = 'test'
        number = 3
    elif dataset == 'Kodak':
        tt1 = 'kodim'
        tt2 = 'kodim'
        number = 2
    elif dataset == 'Urban100':
        tt1 = 'img_'
        tt2 = tt1
        number = 3
    elif dataset == 'Set12':
        tt1 = ''
        tt2 = tt1
        number = 2
    if noise == 'y' and algorithm == 'BM3D':
        dd = '/noisy/noisy_'
        ii1 = lenlen(number, img)
        ii2 = lenlen(number, img)
    elif noise == 'n' and algorithm == 'BM3D':
        dd = '/denoised/denoised_'
        ii1 = lenlen(number, img)
        ii2 = lenlen(number, img)
    elif noise == 'n' and (algorithm == 'DnCNN' or algorithm == 'FFDNet' or algorithm == 'Proposed'):
        dd = '/denoise_'
        ii1 = lenlen(number, img)
        ii2 = str(img)
        tt2 = ''
    elif noise == 'n' and algorithm == 'WDnCNN':
        dd = '/'
        ii1 = lenlen(number, img)
        ii2 = str(img)
        tt2 = ''
        if dataset == 'Set12':
            ii1 = str(img)
    '''
    gt_path = 'D:/' + dataset + '/' + tt1 + ii1 + '.png'
    my_path = 'D:/IPIU/' + algorithm + '/' + dataset + '/sigma' + str(sigma) + dd + tt2 + ii2 + '.png'
    npImg1 = cv2.imread(gt_path)/255.
    # print(np.max(npImg1))
    npImg2 = cv2.imread(my_path)/255.
    # print(np.max(npImg2))
    '''
    if dataset == 'DIV2K':
        gt_path = 'D:/DIV2K_train_HR_patch_48_noise_50/' + str(img) + '.png'
        my_path = 'D:/DIV2K_train_LR_patch_48_noise_50/' + str(img) + '.png'
        npImg1 = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)/255.
        npImg2 = cv2.imread(my_path, cv2.IMREAD_GRAYSCALE)/255.
    else:
        gt_path = 'D:/' + dataset + '/' + tt1 + ii1 + '.png'
        my_path = 'D:/IPIU/' + algorithm + '/' + dataset + '/sigma' + str(sigma) + dd + tt2 + ii2 + '.png'
        npImg1 = cv2.imread(gt_path)/255.
        # print(np.max(npImg1))
        npImg2 = cv2.imread(my_path)/255.
    squared_error = np.square(npImg1 - npImg2)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    print(noise, algorithm, dataset, sigma, img, psnr)
    return psnr


def myssim(noise, algorithm, dataset, sigma, img):
    number = 0
    tt1 = '0'
    tt2 = '0'
    ii1 = '0'
    ii2 = '0'
    dd = ''
    if dataset == 'BSD68':
        tt1 = 'test'
        tt2 = 'test'
        number = 3
    elif dataset == 'Kodak':
        tt1 = 'kodim'
        tt2 = 'kodim'
        number = 2
    elif dataset == 'Urban100':
        tt1 = 'img_'
        tt2 = tt1
        number = 3
    elif dataset == 'Set12':
        tt1 = ''
        tt2 = tt1
        number = 2
    if noise == 'y' and algorithm == 'BM3D':
        dd = '/noisy/noisy_'
        ii1 = lenlen(number, img)
        ii2 = lenlen(number, img)
    elif noise == 'n' and algorithm == 'BM3D':
        dd = '/denoised/denoised_'
        ii1 = lenlen(number, img)
        ii2 = lenlen(number, img)
    elif noise == 'n' and (algorithm == 'DnCNN' or algorithm == 'FFDNet' or algorithm == 'Proposed'):
        dd = '/denoise_'
        ii1 = lenlen(number, img)
        # ii1 = str(img)
        ii2 = str(img)
        tt2 = ''
    elif noise == 'n' and algorithm == 'NLRN':
        dd = '/'
        ii1 = lenlen(number, img)
        ii2 = lenlen(number, img)
        if dataset == 'Set12':
            tt2 = ''
    elif noise == 'n' and algorithm == 'WDnCNN':
        dd = '/'
        ii1 = lenlen(number, img)
        ii2 = str(img)
        tt2 = ''
        if dataset == 'Set12':
            ii1 = str(img)
    gt_path = 'D:/' + dataset + '/' + tt1 + ii1 + '.png'
    my_path = 'D:/IPIU/' + algorithm + '/' + dataset + '/sigma' + str(sigma) + dd + tt2 + ii2 + '.png'

    npImg1 = cv2.imread(gt_path)
    npImg2 = cv2.imread(my_path)

    if dataset == 'DIV2K':
        gt_path = 'D:/DIV2K_train_HR_patch_48_noise_50/' + str(img) + '.png'
        my_path = 'D:/DIV2K_train_LR_patch_48_noise_50/' + str(img) + '.png'
        npImg1 = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        npImg2 = cv2.imread(my_path, cv2.IMREAD_GRAYSCALE)

    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0)/255.0

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)

    # Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
    ssim_value = pytorch_ssim.ssim(img1, img2)
    print(noise, algorithm, dataset, sigma, img, ssim_value.item())
    return ssim_value.item()


dataset = ['BSD68']
for h in ['n']:
    for i in ['Proposed']:
        for j in dataset:
            if j == 'BSD68':
                n = 68
            elif j == 'Kodak':
                n = 24
            elif j == 'Urban100':
                n = 100
            elif j == 'Set12' or j == 'DIV2K':
                n = 12
            for l in [15]:
                susu = []
                for k in range(1, n+1):
                    # print(h, i, j, l, k)
                    if h == 'y' and i != 'BM3D':
                        break
                    a = myssim(h, i, j, l, k)
                    susu.append(a)
                if h == 'y' and i != 'BM3D':
                    continue
                else:
                    print(h, i, j, l, 'avg.', round(sum(susu)/len(susu), 4))
