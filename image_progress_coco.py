import torch
from PIL import Image
import torchvision.transforms as T
import os
import argparse
import json
from tqdm import tqdm
import numpy as np
from transformers import DetrFeatureExtractor, DetrForObjectDetection
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device =  torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--img_type', type=str, default="detr", choices=['detr', 'vit'], help='type of image features')
    args = parser.parse_args()
    return args

def extract_features_batch(img_type, input_images, preprocessor, detection_model, vf_extractor):
    img_batch = []
    for input_image in input_images:
        img = Image.open(input_image).convert("RGB")
        img_batch.append(img)
        
    inputs = preprocessor(images=img_batch, return_tensors="pt")
    vision_tokenizer = vf_extractor(inputs['pixel_values'].cuda(), inputs['pixel_mask'].cuda())
    return vision_tokenizer[0]

def init_mydata(args):
    # args = parse_args()
    print("args",args)
    all_images = os.listdir(args.data_root)
    preprocessor = DetrFeatureExtractor.from_pretrained('')  
    detection_model = DetrForObjectDetection.from_pretrained('')
    vf_extractor = detection_model.model.to(device)  
    for param in vf_extractor.parameters():
        param.requires_grad = False

    return all_images,preprocessor,detection_model,vf_extractor,args.data_root,args.output_dir,args.img_type

def deal_with_image(all_images, preprocessor, detection_model, vf_extractor, data_root, output_dir, img_type):
    name_map = {}
    tmp = None
    batch_size = 2
    for idx in tqdm(range(0, len(all_images), batch_size)):
        if idx % 100 == 0:
            print(idx)
        
        batch = all_images[idx: idx + batch_size]
        batch_images = []

        for image in batch:
            if os.path.exists(os.path.join(data_root, image, "image.png")):
                curr_dir = os.path.join(data_root, image, "image.png")
            else:
                curr_dir = os.path.join(data_root, image, "choice_0.png")
            batch_images.append(curr_dir)
        
        features = extract_features_batch(img_type, batch_images, preprocessor, detection_model, vf_extractor)
        
        if tmp is None:
            tmp = features.detach().cpu()
        else:
            tmp = torch.cat((tmp, features.detach().cpu()), dim=0)

        for i in range(features.shape[0]):
            name_map[str(batch[i])] = idx + i

    print(tmp.shape)
    tmp_np = tmp.numpy()
    np.save(os.path.join(output_dir, 'tmp.npy'), tmp_np)

    with open(os.path.join(output_dir, 'name_map.json'), 'w') as outfile:
        json.dump(name_map, outfile)


if __name__ == '__main__':
    all_images, preprocessor, detection_model, vf_extractor, data_root, output_dir, img_type = init_mydata()
    deal_with_image(all_images, preprocessor, detection_model, vf_extractor, data_root, output_dir, img_type)
