import os
import time

import PIL.Image
import numpy as np
import scipy.stats
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import json
import random
from einops import rearrange
import torch
import torch.nn as nn
from collections import namedtuple

def mkdir(path):
    if os.path.exists(path):
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)

# the function for printing neural network's weights
print_weight_cnt = 0
def print_weights(data : torch.Tensor, mode='a'):
    global print_weight_cnt
    if print_weight_cnt == 0:
        fout = open('weight.txt', 'w')  # just open-close to clear previous content
        fout.close()
    print_weight_cnt += 1
    buffer_str = str(data.cpu().detach().numpy().copy())
    with open('weight.txt', mode) as fout:
        fout.write('=============times {}=============\n'.format(print_weight_cnt))
        fout.write(str(buffer_str))
        fout.write('\n')
        fout.close()

def image_normalize(image, denormalize=False, copy=False, param_type='CLIP'):
    '''
    image: H x W x C, image pixel's value range should be 0~1
    '''
    if param_type == 'ImageNet':
        # ImageNet, ResNet, DINOv3
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        # CLIP, BLIP
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    if copy == True:
        image = image.clone() if isinstance(image, torch.Tensor) else image.copy()
    if denormalize == False:  # noralize
        for channel in range(3):
            image[:, :, channel] = (image[:, :, channel] - mean[channel]) / std[channel]
    else:  # de-normalize (after de-normalize, range is 0~1)
        for channel in range(3):
            image[:, :, channel] = image[:, :, channel] * std[channel] + mean[channel]
    return image

def batch_images_normalize(images, denormalize=False, copy=False, param_type='CLIP'):
    '''
    image: B x C x H x W, pixel value's range should be 0~1
    '''
    if param_type == 'ImageNet':
        # ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        # CLIP
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    if copy == True:
        images = images.clone() if isinstance(images, torch.Tensor) else images.copy()
    if denormalize == False:  # noralize
        for channel in range(3):
            images[:, channel, :, :] = (images[:, channel, :, :] - mean[channel]) / std[channel]
    else:  # de-normalize
        for channel in range(3):
            images[:, channel, :, :] = images[:, channel, :, :] * std[channel] + mean[channel]
    return images

def make_grid_images(tensor_image, denormalize=True, save_path=None):
    '''
    :param tensor: B x C x H x W, pixel value should be ranges from 0~1 !!!
    :param denormalize:
    :param save_path:
    :return: grid_image, H' x W' x 3
    '''
    # vmax, vmin = torch.max(tensor_image), torch.min(tensor_image)
    image_temp = torch.clone(tensor_image)
    # de-normalize
    if denormalize == True:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # mean = [0.48145466, 0.4578275, 0.40821073]
        # std = [0.26862954, 0.26130258, 0.27577711]
        for channel in range(3):
            image_temp[:, channel, :, :] = image_temp[:, channel, :, :] * std[channel] + mean[channel]
    grid_image = torchvision.utils.make_grid(image_temp, scale_each=0.2)  # 3 x H x W, make_grid will output image with 3 channels
    grid_image = grid_image.permute(1, 2, 0)  # H x W x 3
    if save_path != None:
        grid_image = grid_image.cpu().detach().numpy()[:, :, ::-1]  # convert RGB image to BGR image
        cv2.imwrite(save_path, grid_image * 255)  # convert 0~1 to be 0~255

    return grid_image

def save_batch_images(tensor_images, save_path, prefix='', suffix=''):
    '''
    tensor_images (RGB or single-channel format, 0~255): B x C x H x W
    '''
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    B, C, H, W = tensor_images.shape
    ims_temp = tensor_images.permute(0, 2, 3, 1)  # B x H x W x C
    ims_temp = ims_temp.numpy()
    for i in range(B):
        each_im = ims_temp[i]
        p = os.path.join(save_path, '%s_%s_%s.png'%(prefix, i, suffix))
        if C == 3:  # RGB
            cv2.imwrite(p, each_im[:, :, [2, 1, 0]])
        elif C == 1:  # grayscale
            cv2.imwrite(p, each_im[:, :, 0])

def make_uncertainty_map(sigmas_np, B):
    # sigmas: (B*N) * (L*L), numpy
    # we want to make B images and each image shows N keypoint hotspots.
    W, H = 368, 368
    sigmas = sigmas_np  # sigmas_tensor.cpu().detach().numpy()
    N = int(sigmas.shape[0] / B)
    L = int(np.sqrt(sigmas.shape[1]))
    im_combined = np.zeros((B, H, W))
    for im_j in range(B):
        for i in range(im_j * N, (im_j + 1) * N):
            im = sigmas[i, :].reshape(L, L)
            im_resize = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
            vmin, vmax = np.min(im_resize), np.max(im_resize)
            # print(vmin, vmax)
            if vmax > vmin:
                # im_resize = (im_resize - vmin) / (vmax-vmin)  # normalize to 0~1
                im_resize = (im_resize - vmin) / (vmax)
            # merging using max operation
            ind = im_combined[im_j] < im_resize
            im_combined[im_j][ind] = im_resize[ind]
    vmin, vmax = np.min(im_combined), np.max(im_combined)
    # plt.imshow(im_combined[0])
    # plt.show()

    return im_combined  # B x H x W


def save_plot_image(im, save_path, does_show=False):
    '''
    :param im: H x W x 3 or H x W , numpy
    :return:
    '''
    H, W = im.shape[0], im.shape[1]
    fig, ax = plt.subplots()
    # fig = plt.figure()
    # ax = fig.gca()
    fig.tight_layout()
    fig.patch.set_alpha(0.)  # set the figure face to be transparent
    # im = Image.open("/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images/dog/do86.jpeg").convert('RGB')
    # im = im.resize((square_image_length, square_image_length), PIL.Image.BILINEAR)
    # plt.imshow(im, cmap=plt.cm.jet)
    plt.imshow(im, cmap=plt.cm.viridis)  # default
    # plt.imshow(im)
    # plt.show()

    # remove ticks but the frame still exists
    plt.xticks([])
    plt.yticks([])
    ax.invert_yaxis()

    # Remove the white margin around image
    fig.set_size_inches(W / 100.0, H / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(save_path)
    if does_show == True:
        plt.show()

def show_cam_on_image(original_img, mask, save_path=None, mode='color'):
    # original_img: H x W x C, BGR format, value range 0~255
    # mask: H x W, value ranges 0~255
    if mode == 'color':
        heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET)  # set 0.72 to adjust the jet color ranges
        cam = np.float32(heatmap)*0.35 + np.float32(original_img)
    else:  # 'gray
        H, W = mask.shape
        heatmap = mask.reshape(H, W, 1).repeat(3, axis=2) * 0.72
        cam = np.float32(heatmap) * 0.7 + np.float32(original_img) * 0.3

    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    if save_path != None:
        cv2.imwrite(save_path, cam)
    return cam

def compute_eigenvalues(covar):
    '''
    :param covar: 2 x 2
    :return: eigenvalues, eigenvectors, orientation
    '''
    # eigenvalues e: 2 x 2, each row is an eigenvalue e[i,0]+j*e[i, 1],
    # eigenvectors v: 2 x 2, each column is a corresponding eigenvector v[:, i]
    e, v = torch.eig(covar, eigenvectors=True)
    _, indices = torch.sort(e[:, 0], descending=True, dim=0)
    e2, v2 = e[indices, :], v[:, indices]
    radian = torch.atan2(v2[1, 0], v2[0, 0])  # atan2(vy, vx)
    angle = radian / 3.1415926 * 180  # orientation of the eigenvector for major axis
    return e2, v2, angle

def mean_confidence_interval(accs, confidence=0.95):
    '''
    compute mean and standard error of mean for a sequence of observations
    using t-test
    '''
    if isinstance(accs, np.ndarray) == False:
        accs = np.array(accs)

    n = accs.shape[0]
    if n == 1:
        return accs[0], 0
    m, se = np.mean(accs), scipy.stats.sem(accs)  # sem = standard error of mean = sigma / sqrt(n)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)  # ppf here is the inverse of cdf (cumulative distributin function)
    return m, h

def mean_confidence_interval_multiple(accs_multiple, confidence=0.95):
    '''
    accs_multiple: K x N, K rows, each row will compute mean_confidence_interval
    '''
    K = len(accs_multiple)
    mean, interval = np.zeros(K), np.zeros(K)
    for i in range(K):
        mean[i], interval[i] = mean_confidence_interval(np.array(accs_multiple[i]), confidence=confidence)

    return mean, interval

def load_samples(ann_json_files, local_json_root):
    '''
    ann_json_files: a list
    local_json_root: a path
    return: a list of samples
    '''
    samples = []
    for p in ann_json_files:
        annotation_path = os.path.join(local_json_root, p)
        with open(annotation_path, 'r') as fin:
            # self.samples = json.load(fin)
            samples_temp = json.load(fin)
            # self.samples = dataset['anns']
            fin.close()
        samples += samples_temp

    return samples

def power_norm1(x, SIGMA):
    out = 2/(1 + torch.exp(-SIGMA*x)) - 1
    return out

def power_norm2(x, SIGMA):
    out = torch.sign(x) * torch.abs(x).pow(SIGMA)
    return out





def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



# count model learnable parameters
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000



def get_patches(ims: torch.Tensor, patch_size=(32, 32), save=False, prefix='s', saveroot='./episode_images/patched_ims'):
    '''
    ims: cpu Tensor, B x C x H x W
    return patched_ims: B x (grid_h*grid_w) x C x p1 x p2
    '''
    if save:
        if os.path.exists(saveroot) == False:
            os.makedirs(saveroot)
    B, _, H, W = ims.shape
    patched_ims = rearrange(ims, 'B C (h p1) (w p2) -> B C (h w) p1 p2', p1=patch_size[0], p2=patch_size[1])
    patched_ims = patched_ims.permute(0, 2, 1, 3, 4)  # B x (grid_h*grid_w) x C x p1 x p2

    p = patch_size[0]
    h = H // p
    mask1 = (patched_ims).flatten(2).mean(dim=2)  # B x (grid_h*grid_w) x C
    mask1 = mask1.reshape(B, h, h, -1)

    # mask2 = torch.nn.functional.avg_pool2d(ims, kernel_size=p, stride=p, padding=0)
    # mask2 = mask2.permute(0, 2, 3, 1)  # B x h x h x C
    #
    # mask3 = torch.nn.functional.interpolate(ims, size=(h, h))
    # mask3 = mask3.permute(0, 2, 3, 1)  # B x h x h x C

    if save:
        for i in range(B):
            # grid_iamge: C x H' x W'
            grid_image = torchvision.utils.make_grid(patched_ims[i], nrow=W // patch_size[1], padding=2, normalize=False, pad_value=0.8)
            grid_image = grid_image.permute(1, 2, 0)
            grid_image = grid_image.numpy()[:, :, ::-1]
            cv2.imwrite(os.path.join(saveroot, prefix+'_'+str(i)+'.jpg'), grid_image * 255)

            t = mask1.numpy()[:, :, ::-1]
            cv2.imwrite(os.path.join(saveroot, prefix+'_'+str(i)+'_avg1.jpg'), t * 255)
            # t = mask2.numpy()[:, :, ::-1]
            # cv2.imwrite(os.path.join(saveroot, prefix + '_' + str(i) + '_avg2.jpg'), t * 255)
            # t = mask3.numpy()[:, :, ::-1]
            # cv2.imwrite(os.path.join(saveroot, prefix + '_' + str(i) + '_dsize.jpg'), t * 255)


    return patched_ims

def draw_contours(thresh, mask, im, color='pink', thickness=1):
    '''
    mask: H x W
    im  : H x W x 3
    '''
    # thresh = int(0.1 * 255)
    H, W = im.shape[0:2]
    ret, imbinary = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(imbinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(type(contours))  # a list of coordinates

    # im_out = np.copy(im)
    # if len(im_out.shape) == 2 or (len(im_out.shape) == 3 and im_out.shape[2] == 1):  # single channel
    #     im_out = np.repeat(im_out[..., np.newaxis], repeats=3, axis=2)  # expand

    im_out = np.zeros((H, W, 3))
    if len(im.shape) == 2 or (len(im.shape) == 3 and im.shape[2] == 1):  # single channel
        im = im.reshape(H, W)
        im_out[:, :, 0] = im[:, :]
        im_out[:, :, 1] = im[:, :]
        im_out[:, :, 2] = im[:, :]
    else:
        im_out[:, :, :] = im[:, :, :]

    if color == 'pink':
        c = (255, 0, 255)
    elif color == 'red':
        c = (255, 0, 0)
    elif color == 'green':
        c = (0, 255, 0)
    elif color == 'blue':
        c = (0, 0, 255)
    elif color == 'white':
        c = (255, 255, 255)
    cv2.drawContours(im_out, contours, -1, c, thickness=thickness)

    return im_out

def ele_max(a, b=0):
    # sign = (a >= b).detach().float()
    # return sign * a + (1 - sign) * b
    return torch.clamp(a, min=b)

def apply_noise(x, noise, type='add'):
    '''
    noise~N(0, std)
    type: 'add' or 'mul'
    '''
    y = (x + noise) if type == 'add' else (x * (noise+1))
    return y

def compute_similarity(a, b, type='cosine', **kwargs):
    '''
    the larger the similarity, the better.
    '''
    # a: C or M x C
    # b: C or N x C
    if type == 'cosine': # sim range -1~1
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        b = torch.nn.functional.normalize(b, p=2, dim=-1)
        sim = torch.matmul(a, b.T)  # M x N
    elif type == 'rbf':  # sim range 0~1
        rbf_factor = 0.5  # 0.1 ~ 0.5
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        b = torch.nn.functional.normalize(b, p=2, dim=-1)
        sim = torch.matmul(a, b.T)  # M x N
        sim = torch.exp(-(1 - sim) / rbf_factor)
    else:
        raise NotImplementedError

    return sim  # 1 x 1 or M x N

def compute_similarity2(cfg, a, b, type='cosine'):
    # a: C or M x C
    # b: C or M x C
    # return 1 or M similarity values
    if type == 'cosine':  # sim range -1~1
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        b = torch.nn.functional.normalize(b, p=2, dim=-1)
        sim = torch.sum(a * b, dim=-1)  # M
    elif type == 'rbf':  # sim range 0~1
        rbf_factor = float(cfg.LOSS.DOMAIN_ALIGNMENT.SIMILARITY.RBF_TAU)  # 0.1 ~ 0.5
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        b = torch.nn.functional.normalize(b, p=2, dim=-1)
        sim = torch.sum(a * b, dim=-1)  # M
        sim = torch.exp(-(1 - sim) / rbf_factor)
    else:
        raise NotImplementedError

    return sim  # 1 or M

def compute_distance(a, b, type='l2-norm'):
    '''
    the smaller the distance, the better.
    '''
    # a: C or M x C
    # b: C or N x C
    if type == 'l2':
        C = a.shape[-1]
        a = a.unsqueeze(-2)
        b = b.unsqueeze(0)
        dist = ((a - b) ** 2).sum(-1) / C  # M x N
    elif type == 'l2-norm':  # basically, this is equal to 2 - 2 * 'cosine' similarity
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        b = torch.nn.functional.normalize(b, p=2, dim=-1)
        a = a.unsqueeze(-2)
        b = b.unsqueeze(0)
        dist = ((a - b) ** 2).sum(-1)  # M x N
    elif type == 'l1':
        C = a.shape[-1]
        a = a.unsqueeze(-2)
        b = b.unsqueeze(0)
        dist = (torch.abs(a - b)).sum(-1) / C  # M x N
    else:
        raise NotImplementedError
    return dist  # 1 x 1 or M x N

def summarize_losses(losses_list, weights_list=None):
    # if all losses None, return None; otherwise return summation of valid losses
    flag = False
    loss = 0
    w_sum = 0
    if weights_list is None:
        for i in range(len(losses_list)):
            if (losses_list[i] is not None) and (losses_list[i] != 0):
                loss += losses_list[i]
                flag = True
    else:
        for i in range(len(losses_list)):
            if (losses_list[i] is not None) and (losses_list[i] != 0):
                loss += (weights_list[i] * losses_list[i])
                w_sum += weights_list[i]
                flag = True
    if flag == False:
        loss = None
    if (weights_list is not None) and w_sum > 0:
        loss /= w_sum
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def get_model_summary(model, inputs, item_length=26, verbose=True):
    """
    :param model: an object which inherits nn.Module
    :param inputs: a list which contains input parameters
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            # here we can also add other modules if we need to compute their params & FLOPs,
            # added by Changsheng Lu, 2023.01.27
            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                # Lienar layer B X (.) X C1 --> B X (.) X C2, complexity is B x (.) X C2 x C1,
                # modified by Changsheng Lu, 2023.01.27
                flops = (torch.prod(torch.LongTensor(list(output.size()))) * input[0].size(-1)).item()
            elif class_name.find('SalAttention') != -1:  # only used in my SalViT based FSKD, added by Changsheng Lu, 2023.01.27
                flops = (torch.prod(torch.LongTensor(list(input[0].size()))) * input[0].size(1) * 2).item()
                output = output[0]
            else:
                pass

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            if flops != 'Not Available':  # skip unknown modules, modified by Changsheng Lu, 2023.01.27
                summary.append(
                    ModuleDetails(
                        name=layer_name,
                        input_size=list(input[0].size()),
                        output_size=list(output.size()),
                        num_parameters=params,
                        multiply_adds=flops)
                )
            else:
                summary.append(
                    ModuleDetails(
                        name=layer_name,
                        input_size='unknown',
                        output_size='unknown',
                        num_parameters=0,
                        multiply_adds=0)
                )
                # pass

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*inputs)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep
    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,}".format(flops_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


def display_all_args(args):
    '''
    args: the parsed args, type(args) == argparse.Namespace
    '''
    print('==========================================')
    print('Display all the hyper-parameters in args:')
    print('------------------------------------------')
    for arg in vars(args):
        value = getattr(args, arg)
        # if value is not None:
        print('%s: %s' % (str(arg), str(value)))
    print('==========================================')

def display_dict(d):
    print('==========================================')
    print('Display key-value pairs in dict:')
    print('------------------------------------------')
    for key in d:
        print('%s: %s'%(key, d[key]))
    print('==========================================')

def list2str(l, link_str='-'):
    return link_str.join([str(v) for v in l])

def list_transpose(l: list):
    # transpose a second-level list M x N to N x M
    if len(l) == 0:  # if empty list [], directly return
        return l
    
    result = []
    N = len(l[0])
    for n in range(N):
        result.append([sub_list[n] for sub_list in l])
    return result

class SoftArgmax():
    def __init__(self, h, w, scale_factor=1.0):
        super(SoftArgmax, self).__init__()
        y_range = torch.Tensor([i for i in range(int(h))])
        x_range = torch.Tensor([i for i in range(int(w))])
        yy, xx = torch.meshgrid(y_range, x_range)
        yy = ((yy + 0.5) / h - 0.5) * 2  # -1~1
        xx = ((xx + 0.5) / w - 0.5) * 2  # -1~1
        self.yy, self.xx = yy.reshape(-1), xx.reshape(-1)
        if torch.cuda.is_available():
            self.yy, self.xx = self.yy.cuda(), self.xx.cuda()
        # self.coords = torch.stack((xx, yy), dim=-1).cuda()  # H x W x 2 (each pixel stores an index (ix, iy))
        self.scale_factor = scale_factor
        self.h, self.w = h, w

    def __call__(self, heatmaps: torch.Tensor):
        '''
        heatmaps: B x N x H x W
        return coords B x N x 2 (each entry is (x,y) in range -1~1)
        '''
        B, N, H, W = heatmaps.shape
        assert H == self.h and W == self.w, "Error in SoftArgmax: The dim should be consistent."

        prob = torch.softmax(heatmaps.reshape(B, N, H*W) * self.scale_factor, dim=-1)  # B x N x (H*W)
        y_coords = (self.yy.reshape(1, 1, -1) * prob).sum(-1)
        x_coords = (self.xx.reshape(1, 1, -1) * prob).sum(-1)
        coords = torch.stack((x_coords, y_coords), dim=-1)  # B x N x 2

        return coords

def get_random_occlusion_map(num_patch_l=1, num_patch_h=5, im_prob=0.5, patch_grids=(12, 12), saliency_map=None, fg_thresh=0.4, p_thresh=0.1):
    '''
    Generate occlusion map by randomly occluding a number of foreground patches. The foreground patch is determined by
    the saliency and patch threshold.

    num_patch_l, num_patch_h: low ~ high number of patches to be occluded
    saliency_map: torch tensor, B x C x H x W, e.g., 5 x 1 x 384 x 384
    im_prob: probability to apply occlusion on an image
    fg_thresh: saliency larger than this value is regarded as a foreground pixel
    p_thresh: (the number of FG pixel)/(number of pixels in a patch) larger than this ratio is regarded a FG patch
    '''

    # im_H, im_W = saliency_map.shape[2:]

    patched_ims = rearrange(saliency_map, 'B C (h p1) (w p2) -> B C (h w) p1 p2', h=patch_grids[0], w=patch_grids[1])
    B2, _, num_total_grids, p1, p2 = patched_ims.shape
    patched_ims = patched_ims.reshape(B2, num_total_grids, p1, p2)  # B x (grid_h*grid_w) x p1 x p2
    fg_pixel_indicator = (patched_ims.flatten(2) >= fg_thresh)  # B x (grid_h*grid_w) x (p1 x p2)
    valid_patches = fg_pixel_indicator.sum(-1) >= (p_thresh * p1 * p2)  # B x (grid_h*grid_w)
    num_valid_patches = valid_patches.sum(-1)  # B

    # it is "white" (value=1) by default if no occlusion
    occlusion_maps = torch.ones((B2, patch_grids[0]*patch_grids[1]))  # B x (grid_h x grid_w)
    inds = np.arange(0, num_total_grids, step=1)
    for im_i in range(B2):
        if num_patch_l >= num_patch_h:
            num_patch = num_patch_h
        else:
            num_patch = random.randint(num_patch_l, num_patch_h)
        image_prob = random.uniform(0, 1)
        if (image_prob > im_prob) or (num_patch <= 0) or (num_patch > num_valid_patches[im_i]):
            pass
        else:
            inds_filtered = inds[valid_patches[im_i]]
            random.shuffle(inds_filtered)
            inds_sampled = inds_filtered[0:num_patch]
            occlusion_maps[im_i, inds_sampled] = 0

    return occlusion_maps  # B x (grid_h x grid_w)

def get_random_occlusion_map_by_ratio(num_patch_ratio=0.1, im_prob=0.5, patch_grids=(12, 12), saliency_map=None, fg_thresh=0.4, p_thresh=0.1):
    '''
    Generate occlusion map by a ratio of foreground patches. The foreground patch is determined by
    the saliency and patch threshold.

    num_patch_ratio: a given ratio number of foreground patches to be occluded
    saliency_map: torch tensor, B x C x H x W, e.g., 5 x 1 x 384 x 384
    im_prob: probability to apply occlusion on an image
    fg_thresh: saliency larger than this value is regarded as a foreground pixel
    p_thresh: (the number of FG pixel)/(number of pixels in a patch) larger than this ratio is regarded a FG patch
    '''

    # im_H, im_W = saliency_map.shape[2:]

    patched_ims = rearrange(saliency_map, 'B C (h p1) (w p2) -> B C (h w) p1 p2', h=patch_grids[0], w=patch_grids[1])
    B2, _, num_total_grids, p1, p2 = patched_ims.shape
    patched_ims = patched_ims.reshape(B2, num_total_grids, p1, p2)  # B x (grid_h*grid_w) x p1 x p2
    fg_pixel_indicator = (patched_ims.flatten(2) >= fg_thresh)  # B x (grid_h*grid_w) x (p1 x p2)
    valid_patches = fg_pixel_indicator.sum(-1) >= (p_thresh * p1 * p2)  # B x (grid_h*grid_w)
    num_valid_patches = valid_patches.sum(-1)  # B

    # it is "white" (value=1) by default if no occlusion
    occlusion_maps = torch.ones((B2, patch_grids[0]*patch_grids[1]))  # B x (grid_h x grid_w)
    inds = np.arange(0, num_total_grids, step=1)
    for im_i in range(B2):
        num_patch = int(num_patch_ratio * num_valid_patches[im_i])
        if num_patch > num_valid_patches[im_i]:
            num_patch = num_valid_patches[im_i]
        image_prob = random.uniform(0, 1)
        if (image_prob > im_prob) or (num_patch <= 0) or (num_patch > num_valid_patches[im_i]):
            pass
        else:
            inds_filtered = inds[valid_patches[im_i]]
            random.shuffle(inds_filtered)
            inds_sampled = inds_filtered[0:num_patch]
            occlusion_maps[im_i, inds_sampled] = 0

    return occlusion_maps  # B x (grid_h x grid_w)

#==============================================================
# Below code is useless
def train_parser():
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt", help="optimizer", choices=['adam', 'sgd'])
    parser.add_argument("--lr", help="initial learning rate", type=float)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int)
    parser.add_argument("--stage", help="number lr stages", type=int)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float)
    parser.add_argument("--gpu", help="gpu device", type=int, default=0)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=20)
    parser.add_argument("--resnet", help="whether use resnet18 as backbone or not", action="store_true")

    ## PN model related hyper-parameters
    parser.add_argument("--alpha", help="scalar for pose loss", type=int)
    parser.add_argument("--num_part", help="number of parts", type=int)
    parser.add_argument("--percent", help="percent of base images with part annotation", type=float)

    ## shared optional
    parser.add_argument("--batch_size", help="batch size", type=int)
    parser.add_argument("--load_path", help="load path for dynamic/transfer models", type=str)

    args = parser.parse_args()

    if args.resnet:
        name = 'ResNet18'
    else:
        name = 'Conv4'

    return args, name



