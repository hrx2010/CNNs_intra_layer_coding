import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

rgb_avg = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

transdata = transforms.Compose(
	[transforms.Resize(256,interpolation=1),
	 transforms.CenterCrop(224),
	 transforms.ToTensor(),
	 transforms.Normalize(rgb_avg, rgb_std)])

def loadnetwork(archname):
	# load the network
	if archname == 'alexnet':
		net = models.alexnet(pretrained=True)
	elif archname == 'resnet50':
		net = models.resnet50(pretrained=True)
	elif archname == 'mobilenetv2':
		net = models.mobilenet.mobilenet_v2(pretrained=True)
	# load the dataset
	dataset = datasets.ImageNet \
		(root='~/Developer/ILSVRC2012_devkit_t12',split='val',transform=transdata)

	return neural, dataset
