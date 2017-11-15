import torch.nn as nn
import torch.nn.functional as F
import dic
import torch


class HardNegLoss(nn.Module):
	def __init__(self, ratio=3, num_classes=dic.ClassNum):

		super(HardNegLoss, self).__init__()
		self.ratio = ratio
		self.num_classes = num_classes
		self.softmax = nn.Softmax()
	def forward(self, pred, target):
		"""
		pred : num x class
		target : num x class
		"""
		loss = 0
		num = pred.size(0)
		classes = pred.size(1)
		assert classes == self.num_classes

		for idx in range(num):
			mask = target[idx].clone()#torch.zeros(classes).cuda().double()#type_as(pred)
			pos =  int(torch.sum(target[idx]).data[0])
			neg = int(min(pos * self.ratio, classes - pos))
			prob = self.softmax(pred[idx].unsqueeze(0)).squeeze()
			_, index = (prob-target[idx]).topk(neg,-1)
			mask[index.data] = 1 

			loss += F.multilabel_soft_margin_loss(pred[idx], target[idx], weight=mask, size_average=False)


		return loss / torch.sum(target).data[0]

