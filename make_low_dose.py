import os
import json
from itertools import islice
from math import ceil
import numpy as np
import odl
from tqdm import tqdm
from skimage.transform import resize
from pydicom.filereader import dcmread
import h5py
import multiprocessing
import matplotlib.pyplot as plt
import cv2
from glob import glob

# linear attenuations in m^-1
MU_WATER = 20
MU_AIR = 0.02
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER

MIN_PT = [-0.13, -0.13]
MAX_PT = [0.13, 0.13]

dcm_paths = glob('/home/do/tmp/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/*')

def lidc_idri_gen(dataset):
    seed = 0
    #array = dataset.pixel_array[75:-75, 75:-75].astype(np.float32).T
    array = dataset.pixel_array.astype(np.float32).T
    # rescale by dicom meta info
    array *= dataset.RescaleSlope
    array += dataset.RescaleIntercept

    # add noise to get continuous values from discrete ones
 #   array += r.uniform(0., 1., size=array.shape)

    # convert values
    array *= (MU_WATER - MU_AIR) / 1000
    array += MU_WATER
    array /= MU_MAX
    np.clip(array, 0., 1., out=array)

    yield array
    
NUM_ANGLES = 1000
RECO_IM_SHAPE = (512, 512)
IM_SHAPE = (512, 512)
DETECTOR_SHAPE = (727,)


reco_space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT,
                               shape=RECO_IM_SHAPE, dtype=np.float32)
space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE,
                          dtype=np.float64)

reco_geometry = odl.tomo.parallel_beam_geometry(
    reco_space, num_angles=NUM_ANGLES)
geometry = odl.tomo.parallel_beam_geometry(
    space, num_angles=NUM_ANGLES, det_shape=reco_geometry.detector.shape)

impl = 'astra_cpu'
IMPL = 'astra_cpu'
reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry, impl=IMPL)
ray_trafo = odl.tomo.RayTransform(space, geometry, impl=IMPL)

PHOTONS_PER_PIXEL = 512  ## 4096

rs = np.random.RandomState(3)

NUM_SAMPLES_PER_FILE = 1   #128

def forward_fun(im):
    # upsample ground_truth from 362px to 1000px in each dimension
    # before application of forward operator in order to avoid
    # inverse crime
    im_resized = resize(im * MU_MAX, IM_SHAPE, order=1)

    # apply forward operator
    data = ray_trafo(im_resized).asarray()

    data *= (-1)
    np.exp(data, out=data)
    data *= PHOTONS_PER_PIXEL
    return data

def normal2sino(dataset):
   gen = lidc_idri_gen(dataset)
   im_buf = [im for im in islice(gen, NUM_SAMPLES_PER_FILE)]
   with multiprocessing.Pool(20) as pool:
      data_buf = pool.map(forward_fun, im_buf)
   for i, (im, data) in enumerate(zip(im_buf, data_buf)):
      data = rs.poisson(data) / PHOTONS_PER_PIXEL
      np.maximum(0.1 / PHOTONS_PER_PIXEL, data, out=data)
      np.log(data, out=data)
      data /= (-MU_MAX)
   return data.T


def normal2low(dataset):
   x = normal2sino(dataset)
   space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=RECO_IM_SHAPE, dtype=np.float64)
   geometry = odl.tomo.parallel_beam_geometry(space, num_angles=NUM_ANGLES, det_shape=DETECTOR_SHAPE)
   result = odl.tomo.RayTransform(space, geometry, impl=impl)
   fbp_op = odl.tomo.fbp_op(result, filter_type='Hann', frequency_scaling=1)
   return np.asarray(fbp_op(x.T)).T


if __name__ == "__main__":
   dicom = dcmread('/home/do/tmp/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/000051.dcm')
   
   PHOTONS_PER_PIXEL_LIST = [32, 64, 128, 256, 512, 1024, 2048, 4096]
   array = dicom.pixel_array.astype(np.float32).T
   array *= dicom.RescaleSlope
   array += dicom.RescaleIntercept
   print('RescaleSlop: ', dicom.RescaleSlope)
   print('RescaleIntercept: ', dicom.RescaleIntercept)
   img = array.T
   img *= (MU_WATER - MU_AIR) / 1000
   img += MU_WATER
   img /= MU_MAX
   np.clip(img, 0., 1., out=img)
   print(img.min(), img.max())
   plt.imsave("normal.png", img, cmap='gray')
   for PHOTONS_PER_PIXEL in PHOTONS_PER_PIXEL_LIST:   
      plt.imsave("normal2low_PHOTONS{}.png".format(PHOTONS_PER_PIXEL), normal2low(dicom), cmap='gray')
   