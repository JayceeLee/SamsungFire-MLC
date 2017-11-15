# -*- coding: utf-8 -*-

import torch.utils.data as data
import torch.utils.data.dataloader as loader
from PIL import Image, ImageEnhance
import os
import xlrd
import csv
from os import sys
import dic
import numpy as np
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def my_collate(batch):
	batch = list(filter (lambda x:x is not None, batch))
	return loader.default_collate(batch)

def make_dataset(dir, excel_file):
	assert os.path.exists(excel_file), excel_file
	images = []
	
	wb = xlrd.open_workbook(excel_file)
	sh = wb.sheet_by_name('MASTER')
	num_rows = sh.nrows
	for row_index in range(num_rows):
		if row_index == 0:
			continue
		
		row_val = sh.row_values(row_index)
		try:
			img_idx = str(int(row_val[0])) #
		except:
			img_idx = str(row_val[0])
		fname = img_idx + '.jpg'#'%d.jpg' % img_idx
		strings = ''
		for sign_idx in range(int(row_val[1])):
			strings = strings + row_val[2+sign_idx]
		idx_of_string = dic.convert_hangul_to_index(strings)
		label = - np.ones([dic.ClassNum])
		for i in range(len(idx_of_string)):
			label[idx_of_string[i]] = 1
		
		if is_image_file(fname):
				path = os.path.join(dir, fname)
				if os.path.exists(path):
					item = (path, label)
					images.append(item)
				else:
					continue

	return images


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def accimage_loader(path):
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		print('acc_loader',IOError)
		try:
			return pil_loader(path)
		except IOError:
			print('pil_loader', IOError)


def default_loader(path):
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)		

	
class samsungMLC(data.Dataset):
	
	def __init__(self, root, transform=None, target_transform=None,loader = default_loader, excel_file = './Lexicon.xlsx'):
		imgs = make_dataset(root, excel_file)
		if len(imgs) == 0:
			raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
				"Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

		self.root = root
		self.imgs = imgs
		
		
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
	def __getitem__(self, index):
		"""
		Args:
		    index (int): Index

		Returns:
		    tuple: (image, target) where target is class_index of the target class.
		"""
		try:
			path, target = self.imgs[index]
			img = self.loader(path)
			if self.transform is not None:
				img = self.transform(img)
			if self.target_transform is not None:
				target = self.target_transform(target)
			return (img, path), np.double(target)
		
		except Exception as e:
			print(e)





	def __len__(self):
		return len(self.imgs)

class RandomAug(object):


	def __init__(self, p = 0.2):
		self.p = p

	def __call__(self, img):
		
		if random.random() <= self.p:
			contrast = ImageEnhance.Contrast(img)
			img = contrast.enhance(0.5+ 1.5*random.random())
		if random.random() <= self.p:
			bright = ImageEnhance.Brightness(img)
			img = bright.enhance(0.5 + 0.5*random.random())
		if random.random() <= self.p:
			sharp = ImageEnhance.Sharpness(img)
			img = sharp.enhance(2* random.random())


		return img
