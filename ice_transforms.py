'''transforms for images / pytorch tensors'''
import torch
import affine_transforms as af


def clip_low(x):
    '''clip an image to values above median'''
    median = torch.median(x)
    return torch.clamp(x, min=median)


def clip_low_except_center(x, size=10):
    '''clip an image to values above median,
    but keep entire image in square of +/- size px
    about center of image'''

    median = torch.median(x)
    out = torch.clamp(x, min=median)
    cntr = x.size()[1] // 2
    im_cntr = x[:, cntr-size:cntr+size, cntr-size:cntr+size]
    out[:, cntr-size:cntr+size, cntr-size:cntr+size] = im_cntr
    return out


def center_crop(im, center=None, size=28):
    '''crop center of square images to output_size * 2
    center defaults to true center'''
    if type(im) is torch.FloatTensor:
        im = im.numpy()
    if center is None:
        x, y = im.shape[1] // 2, im.shape[2] // 2
    else:
        x, y = center
        if x < size:
            x = size
        elif x > im.shape[1] - size:
            x = im.shape[1] - size
        if y < size:
            y = size
        elif y > im.shape[2] - size:
            y = im.shape[2] - size

    return im[:, x-size:x+size, y-size:y+size]


def scale_to_angle(x, angle=.6):
    '''don't do this because it's broken'''
    return af.Zoom(1.6 - .75 * angle)(x)


def contrast_background(x):
    mask = x.lt(torch.median(x))    # element wise less than
    return x.masked_fill_(mask, torch.min(x))
