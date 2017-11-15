#-*- encoding: utf-8 -*-
import numpy as np
from hanja import hangul
FirNum = 19  # add for 'space'
SecNum = 21 
LasNum = 27 + 1 
ClassNum = FirNum * SecNum * LasNum + 1
# SpaceNum = 1

def convert_hangul_to_index(string):#, size):
	#string = unicode(string)
	list = []#np.ndarray([size, 3])
	for i in range(len(string)):
		#exception
		if not hangul.is_hangul(string[i]):
			continue
		char3 = hangul.separate(string[i])
		idx = char3[0] +  char3[1] * FirNum + char3[2] * FirNum * SecNum
		list.append([idx])
	if len(list)==0:
		list.append([ClassNum - 1])

	return np.array(list)
		
def convert_index_to_hangul(list):
	str = ''
	#list = list.view(-1, 3)
	for i in range(len(list)):
		remain1 = list[i]
		char3 = remain1 // (FirNum * SecNum)
		remain2 = remain1 - char3 * FirNum * SecNum
		char2 = remain2 // FirNum
		remain3 = remain2 - char2 * FirNum
		char1 = remain3
		
		hg = hangul.synthesize(char1, char2, char3)
		if hangul.is_hangul(hg):
			str = str + hg

	
	return str
