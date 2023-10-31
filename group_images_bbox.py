# import required packages
from config import Config
import pickle as pkl
from generate_images import ImgGenerator
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
import argparse

def save_image(final_image,img_bbox,file_name,output_path):
    i=np.concatenate(final_image)
    y=i.shape[0]
    print(f'SAVING {file_name}')
    cv2.imwrite(os.path.join(output_path,file_name)+'.jpg',cv2.resize(i,(1024,y)))
    with open(os.path.join(output_path,file_name)+'.txt','w+',encoding='utf8') as f:
        f.writelines([i+'\n' for i in img_bbox])


def gen_images(generator,input_words,output_path):
    final_image=[]
    img_bbox=[]
    sentence_img = []
    n_lines=0
    page_w=1024
    word_h=64
    space=int(word_h/2)
    saved_pages=0
    max_lines=int(page_w/word_h)
    sentence_img.append(np.ones((word_h, space))*255)
    line_x=space
    for word in input_words:
        try:
            word=word.split()[-1]
            generated_imgs, _, word_labels = generator.generate(word_list=[word])
            img=generated_imgs[0]
            img = img[:, img.sum(0) < 31.5]*255
            y,x=img.shape[:2]
            new_x=int(word_h/y*x)
            img=cv2.resize(img,(new_x,word_h))
            line_x+=new_x+space

            if line_x>page_w:
                sentence_img = np.hstack(sentence_img)
                residual=page_w-sentence_img.shape[1]
                sentence_img=np.hstack([sentence_img,np.ones((word_h,residual))*255])
                sentence_img=cv2.resize(sentence_img,(page_w,word_h))
                final_image.append(sentence_img)
                n_lines+=1
                if n_lines%max_lines==0:
                    saved_pages+=1
                    save_image(final_image,img_bbox,f'page_{saved_pages}',output_path)
                    final_image=[]
                    img_bbox=[]
                line_x=space+new_x+space
                sentence_img=[]
                sentence_img.append(np.ones((img.shape[0], space))*255)

            bbox_x1=line_x-new_x-space
            bbox_y1=word_h*(n_lines%max_lines)
            bbox_x2=line_x-space
            bbox_y2=word_h*((n_lines%max_lines)+1)

            t_bbox=f'{word_labels[0]} {bbox_x1} {bbox_y1} {bbox_x2} {bbox_y2}'
        except Exception as A:
            print(A)
            continue
        sentence_img.append(img)
        sentence_img.append(np.ones((img.shape[0], space))*255)
        img_bbox.append(t_bbox)

    if len(sentence_img)>0:
        try:
            sentence_img = np.hstack(sentence_img)
            residual=page_w-sentence_img.shape[1]
            sentence_img=np.hstack([sentence_img,np.ones((word_h,residual))*255])
            sentence_img=cv2.resize(sentence_img,(page_w,word_h))
            final_image.append(sentence_img)
            n_lines+=1
            
            saved_pages+=1
            save_image(final_image,img_bbox,f'page_{saved_pages}',output_path)
        except:
            pass

def main(args):
    images_folder = args.output_folder_name
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    # loading the model
    config = Config
    print('LOADING THE MODEL')
    with open(config.data_file, 'rb') as f:  # data file path
        char_map = pkl.load(f)
    char_map=char_map['char_map']
    generator = ImgGenerator(checkpt_path=args.model_path, config=config, char_map=char_map)

    with open(args.input_file,'r') as f:
        input_words=f.read().split('\n')
    gen_images(generator,input_words,args.output_folder)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_file", type=str, default=None, help="path to the input img file")
    parser.add_argument("-o", "--output_folder", type=str, default="", help="path to the output img directory")
    parser.add_argument("-m", "--model_path", type=str, default='./weights/model_checkpoint_epoch_100.pth.tar', help="path to the model")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)