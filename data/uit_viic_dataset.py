import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

class uit_viic_dataset_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        anno_file = 'uitviic_train_vi.json'
        
        self.annotations = json.load(open(os.path.join(ann_root,anno_file),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root,self.annotations[index]['image'])
        image_id = image_path.split('/')[-1].split('.')[0]
        while image_id[0] == '0':
            image_id = image_id[1:]
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = self.prompt+pre_caption(self.annotations[index]['caption'], self.max_words)

        return image, caption, image_id

class uit_viic_dataset_val(Dataset):
    def __init__(self, transform, image_root, ann_root, split='val', max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        anno_file = 'uitviic_{}_vi.json'.format(split)
        
        self.annotations = json.load(open(os.path.join(ann_root,anno_file),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):    
        
        image_path = self.annotations[index]['image']
        image_id = image_path.split('/')[-1].split('.')[0]
        while image_id[0] == '0':
            image_id = image_id[1:]
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        return image, int(image_id)
    