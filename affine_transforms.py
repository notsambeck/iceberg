"""
Affine transforms implemented on torch tensors, and
requiring only one interpolation

https://github.com/ncullen93/torchsample/blob/master/
torchsample/transforms/affine_transforms.py
"""

import math
import random
import torch as th
import numpy as np


def th_random_choice(a, n_samples=1, replace=True, p=None):
    """
    Parameters
    -----------
    a : 1-D array-like
        If a th.Tensor, a random sample is generated from its elements.
        If an int, the random sample is generated as if a was th.range(n)
    n_samples : int, optional
        Number of samples to draw. Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.
    Returns
    --------
    samples : 1-D ndarray, shape (size,)
        The generated random samples
    """
    if isinstance(a, int):
        a = th.arange(0, a)

    if p is None:
        if replace:
            idx = th.floor(th.rand(n_samples)*a.size(0)).long()
        else:
            idx = th.randperm(len(a))[:n_samples]
    else:
        if abs(1.0-sum(p)) > 1e-3:
            raise ValueError('p must sum to 1.0')
        if not replace:
            raise ValueError('replace must equal true if probabilities given')
        idx_vec = th.cat([th.zeros(round(p[i]*1000))+i for i in range(len(p))])
        idx = (th.floor(th.rand(n_samples)*999)).long()
        idx = idx_vec[idx].long()
    selection = a[idx]
    if n_samples == 1:
        selection = selection[0]
    return selection


def th_iterproduct(*args):
    return th.from_numpy(np.indices(args).reshape((len(args), -1)).T)


def th_nearest_interp2d(input, coords):
    """
    2d nearest neighbor interpolation th.Tensor
    """
    # take clamp of coords so they're in the image bounds
    x = th.clamp(coords[:, :, 0], 0, input.size(1)-1).round()
    y = th.clamp(coords[:, :, 1], 0, input.size(2)-1).round()

    stride = th.LongTensor(input.stride())
    x_ix = x.mul(stride[1]).long()
    y_ix = y.mul(stride[2]).long()

    input_flat = input.view(input.size(0), -1)

    mapped_vals = input_flat.gather(1, x_ix.add(y_ix))

    return mapped_vals.view_as(input)


def th_bilinear_interp2d(input, coords):
    """
    bilinear interpolation in 2d
    """
    x = th.clamp(coords[:, :, 0], 0, input.size(1)-2)
    x0 = x.floor()
    x1 = x0 + 1
    y = th.clamp(coords[:, :, 1], 0, input.size(2)-2)
    y0 = y.floor()
    y1 = y0 + 1

    stride = th.LongTensor(input.stride())
    x0_ix = x0.mul(stride[1]).long()
    x1_ix = x1.mul(stride[1]).long()
    y0_ix = y0.mul(stride[2]).long()
    y1_ix = y1.mul(stride[2]).long()

    input_flat = input.view(input.size(0), -1)

    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix))
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix))
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix))
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix))

    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))

    return x_mapped.view_as(input)


def th_affine2d(x, matrix, mode='bilinear', center=True):
    """
    2D Affine image transform on th.Tensor

    Arguments
    ---------
    x : th.Tensor of size (C, H, W)
        image tensor to be transformed
    matrix : th.Tensor of size (3, 3) or (2, 3)
        transformation matrix
    mode : string in {'nearest', 'bilinear'}
        interpolation scheme to use
    center : boolean
        whether to alter the bias of the transform
        so the transform is applied about the center
        of the image rather than the origin
    Example
    -------
    >>> import torch
    >>> from torchsample.utils import *
    >>> x = th.zeros(2,1000,1000)
    >>> x[:,100:1500,100:500] = 10
    >>> matrix = th.FloatTensor([[1.,0,-50],
    ...                             [0,1.,-50]])
    >>> xn = th_affine2d(x, matrix, mode='nearest')
    >>> xb = th_affine2d(x, matrix, mode='bilinear')
    """

    if matrix.dim() == 2:
        matrix = matrix[:2, :]
        matrix = matrix.unsqueeze(0)
    elif matrix.dim() == 3:
        if matrix.size()[1:] == (3, 3):
            matrix = matrix[:, :2, :]

    A_batch = matrix[:, :, :2]
    # print(A_batch, x)
    if A_batch.size(0) != x.size(0):
        A_batch = A_batch.repeat(x.size(0), 1, 1)
    b_batch = matrix[:, :, 2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(1), x.size(2))
    coords = _coords.unsqueeze(0).repeat(x.size(0), 1, 1).float()

    if center:
        # shift the coordinates so center is the origin
        coords[:, :, 0] = coords[:, :, 0] - (x.size(1) / 2. - 0.5)
        coords[:, :, 1] = coords[:, :, 1] - (x.size(2) / 2. - 0.5)
    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1, 2))+b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:, :, 0] = new_coords[:, :, 0] + (x.size(1) / 2. - 0.5)
        new_coords[:, :, 1] = new_coords[:, :, 1] + (x.size(2) / 2. - 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp2d(x.contiguous(), new_coords)
    elif mode == 'bilinear':
        x_transformed = th_bilinear_interp2d(x.contiguous(), new_coords)

    return x_transformed


class RandomAffine(object):

    def __init__(self,
                 rotation_range=None,
                 translation_range=None,
                 shear_range=None,
                 zoom_range=None,
                 interp='bilinear',
                 lazy=False):
        """
        Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.
        Arguments
        ---------
        rotation_range : one integer or float
            image will be rotated randomly between (-degrees, degrees)
        translation_range : a float or a tuple/list with 2 floats between [0, 1)
            first value:
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)
        shear_range : float
            image will be sheared randomly between (-degrees, degrees)
        zoom_range : list/tuple with two floats between [0, infinity).
            first float should be less than the second
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        """
        self.transforms = []
        if rotation_range is not None:
            rotation_tform = RandomRotate(rotation_range, lazy=True)
            self.transforms.append(rotation_tform)

        if translation_range is not None:
            translation_tform = RandomTranslate(translation_range, lazy=True)
            self.transforms.append(translation_tform)

        if shear_range is not None:
            shear_tform = RandomShear(shear_range, lazy=True)
            self.transforms.append(shear_tform)

        if zoom_range is not None:
            zoom_tform = RandomZoom(zoom_range, lazy=True)
            self.transforms.append(zoom_tform)

        self.interp = interp
        self.lazy = lazy

        if len(self.transforms) == 0:
            raise Exception('Must give at least one transform parameter')

    def __call__(self, *inputs):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](inputs[0])
        for tform in self.transforms[1:]:
            tform_matrix = tform_matrix.mm(tform(inputs[0]))
        self.tform_matrix = tform_matrix

        if self.lazy:
            return tform_matrix
        else:
            outputs = Affine(tform_matrix,
                             interp=self.interp)(*inputs)
            return outputs


class Affine(object):

    def __init__(self,
                 tform_matrix,
                 interp='bilinear'):
        """
        Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.
        Arguments
        ---------
        tform_matrix : a 2x3 or 3x3 matrix
            affine transformation matrix to apply
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        """
        self.tform_matrix = tform_matrix
        self.interp = interp

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple, list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        outputs = []
        for idx, _input in enumerate(inputs):
            input_tf = th_affine2d(_input,
                                   self.tform_matrix,
                                   mode=interp[idx])
            outputs.append(input_tf)
        return outputs if idx > 1 else outputs[0]


class AffineCompose(object):

    def __init__(self,
                 transforms,
                 interp='bilinear'):
        """
        Apply a collection of explicit affine transforms to an input image,
        and to a target image if necessary
        Arguments
        ---------
        transforms : list or tuple
            each element in the list/tuple should be an affine transform.
            currently supported transforms:
                - Rotate()
                - Translate()
                - Shear()
                - Zoom()
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        """
        self.transforms = transforms
        self.interp = interp
        # set transforms to lazy so they only return the tform matrix
        for t in self.transforms:
            t.lazy = True

    def __call__(self, *inputs):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](inputs[0])
        for tform in self.transforms[1:]:
            tform_matrix = tform_matrix.mm(tform(inputs[0]))

        if not isinstance(self.interp, (tuple, list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        outputs = []
        for idx, _input in enumerate(inputs):
            input_tf = th_affine2d(_input,
                                   tform_matrix,
                                   mode=interp[idx])
            outputs.append(input_tf)
        return outputs if idx > 1 else outputs[0]


class RandomRotate(object):

    def __init__(self,
                 rotation_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.
        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.rotation_range = rotation_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        degree = random.uniform(-self.rotation_range, self.rotation_range)

        if self.lazy:
            return Rotate(degree, lazy=True)(inputs[0])
        else:
            outputs = Rotate(degree,
                             interp=self.interp)(*inputs)
            return outputs


class RandomChoiceRotate(object):

    def __init__(self,
                 values,
                 p=None,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image from a list of values. If the image
        has multiple channels, the same rotation will be applied to each channel.
        Arguments
        ---------
        values : a list or tuple
            the values from which the rotation value will be sampled
        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1.
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if p is None:
            p = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        degree = th_random_choice(self.values, p=self.p)

        if self.lazy:
            return Rotate(degree, lazy=True)(inputs[0])
        else:
            outputs = Rotate(degree,
                             interp=self.interp)(*inputs)
            return outputs


class Rotate(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each chanl.
        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple, list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = math.pi / 180 * self.value
        rot_matrix = th.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
                                     [math.sin(theta), math.cos(theta), 0],
                                     [0, 0, 1]])
        if self.lazy:
            return rot_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       rot_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]


class RandomTranslate(object):

    def __init__(self,
                 translation_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly translate an image some fraction of total height and/or
        some fraction of total width. If the image has multiple channels,
        the same translation will be applied to each channel.
        Arguments
        ---------
        translation_range : two floats between [0, 1)
            first value:
                fractional bounds of total height to shift image
                image will be horizontally shifted between
                (-height_range * height_dimension, height_range * height_dim)
            second value:
                fractional bounds of total width to shift image
                Image will be vertically shifted between
                (-width_range * width_dimension, width_range * width_dimension)
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        if isinstance(translation_range, float):
            translation_range = (translation_range, translation_range)
        self.height_range = translation_range[0]
        self.width_range = translation_range[1]
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        # height shift
        random_height = random.uniform(-self.height_range, self.height_range)
        # width shift
        random_width = random.uniform(-self.width_range, self.width_range)

        if self.lazy:
            return Translate([random_height, random_width],
                             lazy=True)(inputs[0])
        else:
            outputs = Translate([random_height, random_width],
                                interp=self.interp)(*inputs)
            return outputs


class RandomChoiceTranslate(object):

    def __init__(self,
                 values,
                 p=None,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly translate an image some fraction of total height and/or
        some fraction of total width from a list of potential values.
        If the image has multiple channels,
        the same translation will be applied to each channel.
        Arguments
        ---------
        values : a list or tuple
            the values from which the translation value will be sampled
        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1.
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if p is None:
            p = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        random_height = th_random_choice(self.values, p=self.p)
        random_width = th_random_choice(self.values, p=self.p)

        if self.lazy:
            return Translate([random_height, random_width],
                             lazy=True)(inputs[0])
        else:
            outputs = Translate([random_height, random_width],
                                interp=self.interp)(*inputs)
            return outputs


class Translate(object):

    def __init__(self, 
                 value, 
                 interp='bilinear',
                 lazy=False):
        """
        Arguments
        ---------
        value : float or 2-tuple of float
            if single value, both horizontal and vertical translation
            will be this value * total height/width. Thus, value should
            be a fraction of total height/width with range (-1, 1)
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        """
        if not isinstance(value, (tuple, list)):
            value = (value, value)

        if value[0] > 1 or value[0] < -1:
            raise ValueError('Translation must be between -1 and 1')
        if value[1] > 1 or value[1] < -1:
            raise ValueError('Translation must be between -1 and 1')

        self.height_range = value[0]
        self.width_range = value[1]
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple, list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        tx = self.height_range * inputs[0].size(1)
        ty = self.width_range * inputs[0].size(2)

        translation_matrix = th.FloatTensor([[1, 0, tx],
                                             [0, 1, ty],
                                             [0, 0, 1]])
        if self.lazy:
            return translation_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       translation_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]


class RandomShear(object):

    def __init__(self,
                 shear_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly shear an image with radians (-shear_range, shear_range)
        Arguments
        ---------
        shear_range : float
            radian bounds on the shear transform

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        self.shear_range = shear_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        shear = random.uniform(-self.shear_range, self.shear_range)
        if self.lazy:
            return Shear(shear,
                         lazy=True)(inputs[0])
        else:
            outputs = Shear(shear,
                            interp=self.interp)(*inputs)
            return outputs


class RandomChoiceShear(object):

    def __init__(self,
                 values,
                 p=None,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly shear an image with a value sampled from a list of values.
        Arguments
        ---------
        values : a list or tuple
            the values from which the rotation value will be sampled
        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1.
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if p is None:
            p = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        shear = th_random_choice(self.values, p=self.p)

        if self.lazy:
            return Shear(shear,
                         lazy=True)(inputs[0])
        else:
            outputs = Shear(shear,
                            interp=self.interp)(*inputs)
            return outputs


class Shear(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple, list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = (math.pi * self.value) / 180
        shear_matrix = th.FloatTensor([[1, -math.sin(theta), 0],
                                       [0, math.cos(theta), 0],
                                       [0, 0, 1]])
        if self.lazy:
            return shear_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       shear_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]


class RandomZoom(object):

    def __init__(self,
                 zoom_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly zoom in and/or out on an image
        Arguments
        ---------
        zoom_range : tuple or list with 2 values, both between (0, infinity)
            lower and upper bounds on percent zoom.
            Anything less than 1.0 will zoom in on the image,
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in,
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        self.zoom_range = zoom_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])

        if self.lazy:
            return Zoom([zx, zy], lazy=True)(inputs[0])
        else:
            outputs = Zoom([zx, zy], 
                           interp=self.interp)(*inputs)
            return outputs


class RandomChoiceZoom(object):

    def __init__(self, 
                 values,
                 p=None,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly zoom in and/or out on an image with a value sampled from
        a list of values
        Arguments
        ---------
        values : a list or tuple
            the values from which the applied zoom value will be sampled
        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1.
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if p is None:
            p = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        zx = th_random_choice(self.values, p=self.p)
        zy = th_random_choice(self.values, p=self.p)

        if self.lazy:
            return Zoom([zx, zy], lazy=True)(inputs[0])
        else:
            outputs = Zoom([zx, zy], 
                           interp=self.interp)(*inputs)
            return outputs


class Zoom(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Arguments
        ---------
        value : float
            Fractional zoom.
            =1 : no zoom
            >1 : zoom-in (value-1)%
            <1 : zoom-out (1-value)%
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy: boolean
            If true, just return transformed
        """

        if not isinstance(value, (tuple, list)):
            value = (value, value)
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple, list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        zx, zy = self.value
        zoom_matrix = th.FloatTensor([[zx, 0, 0],
                                      [0, zy, 0],
                                      [0, 0,  1]])

        if self.lazy:
            return zoom_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       zoom_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]
