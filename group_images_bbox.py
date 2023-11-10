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
import math
import random
from tqdm import tqdm

def preprocess(file_path,border_x,border_y):
    img=cv2.imread(file_path)
    y,x=img.shape[:2]
    border_cut_y=int(border_y/100*y)
    border_cut_x=int(border_x/100*x)
    img=img[border_cut_y:y-border_cut_y,border_cut_x:x-border_cut_x]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    iy,iw=gray.shape
    thresh= cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    blur=cv2.GaussianBlur(gray,(13,13),100)
    thresh_inv=cv2.threshold(blur,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    cnts=cv2.findContours(thresh_inv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if len(cnts)==2 else cnts[1]
    cnts=sorted(cnts,key=lambda x:cv2.boundingRect(x)[1])
    xl,yl,xh,yh=0,0,0,0
    for c in cnts:
        x,y,w,h=cv2.boundingRect(c)
        if not (((abs(x-0)<5 or abs(x-iw)<5) or (abs(y-0)<5 or abs(y-iy)<5)) and h<30 and w<50):
            if xh==0:
                xl,yl,xh,yh=x,y,x+w,y+h
            else:
                xl=min(xl,x)
                yl=min(yl,y)
                xh=max(xh,x+w)
                yh=max(yh,y+h)
                
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
    crp=thresh[yl:yh,xl:xh]
    return crp

def save_image(final_image,img_bbox,file_name,output_path):
    i=np.concatenate(final_image)
    i = i.astype("uint8")
    # print(f'SAVING {file_name}')
    cv2.imwrite(os.path.join(output_path,'images',file_name)+'.jpg',i)
    with open(os.path.join(output_path,'txt',file_name)+'.txt','w+',encoding='utf8') as f:
        f.writelines([i+'\n' for i in img_bbox])

def gen_images(generator, input_words, language,output_folder,min_word_h,max_word_h,page_w,page_h,space_x,space_y, saved_pages):
    
    final_image=[]
    img_bbox=[]
    sentence_img=[]
    n_lines=0
    line_x=int(max_word_h/3)
    saved_pages=0
    max_lines=math.floor(page_h/(max_word_h+space_y))
    page_left_from_bottom=page_h-((max_word_h+space_y)*max_lines)
    sentence_img.append(np.ones((max_word_h, int(max_word_h/3)))*255)
    skipped_words=[]
    
    for word in tqdm(input_words):
        try:
            generated_imgs, _, word_labels = generator.generate(word_list=[word])
            img=generated_imgs[0]
            img = img[:, img.sum(0) < 31.5]*255
            # img=preprocess(os.path.join(input_folder,img_path),border_cut_x,border_cut_y)
            y,x=img.shape[:2]
            w_h=random.randint(min_word_h,max_word_h-1)
            new_x=int(w_h/y*x)
            img=cv2.resize(img,(new_x,w_h))
            y,x=img.shape[:2]
            img=np.concatenate([img,np.ones((max_word_h-w_h,x))*255])
            line_x+=x+space_x
            y,x=img.shape[:2]

            if line_x>page_w:
                sentence_img = np.hstack(sentence_img)
                yt,xt=sentence_img.shape[:2]

                if xt>page_w:
                    sentence_img=[]
                    sentence_img.append(np.ones((max_word_h, int(max_word_h/3)))*255)
                    line_x=int(max_word_h/3)
                    print(word,"skipped due to large width")
                    skipped_words.append(word)
                    continue
                    
                residual=page_w-sentence_img.shape[1]
                sentence_img=np.hstack([sentence_img,np.ones((max_word_h,residual))*255])
                sentence_img=cv2.resize(sentence_img,(page_w,max_word_h))
                sentence_img=np.concatenate([sentence_img,np.ones((space_y,page_w))*255])
                final_image.append(sentence_img)
                n_lines+=1
                if n_lines%max_lines==0:
                    saved_pages+=1
                    final_image.append(np.ones((page_left_from_bottom,page_w))*255)
                    save_image(final_image,img_bbox,f'{language}_page_{saved_pages}',output_folder)
                    final_image=[]
                    img_bbox=[]
                sentence_img=[]
                sentence_img.append(np.ones((max_word_h, int(max_word_h/3)))*255)
                line_x=int(max_word_h/3)+new_x+space_x
                # print('NEW LINE')
            bbox_x1=max(0, line_x-new_x-space_x - 8)
            bbox_y1=max(0, (max_word_h+space_y)*(n_lines%max_lines) - 8)
            bbox_x2=line_x-space_x + 8
            bbox_y2=(max_word_h+space_y)*((n_lines%max_lines)+1)-space_y - (max_word_h-w_h) + 8
            
            t_bbox=f'{word} {bbox_x1} {bbox_y1} {bbox_x2} {bbox_y2}'
        except Exception as A:
            print(A)
            continue
        sentence_img.append(img)
        sentence_img.append(np.ones((max_word_h, space_x))*255)
        img_bbox.append(t_bbox)

    if len(sentence_img)>0:
        try:
            sentence_img = np.hstack(sentence_img)
            residual=page_w-sentence_img.shape[1]
            sentence_img=np.hstack([sentence_img,np.ones((max_word_h,residual))*255])
            sentence_img=cv2.resize(sentence_img,(page_w,max_word_h))
            final_image.append(sentence_img)
            n_lines+=1
            
            saved_pages+=1
            final_image.append(np.ones((page_left_from_bottom,page_w))*255)
            save_image(final_image,img_bbox,f'{language}_page_{saved_pages}',output_folder)
        except:
            pass


    with open(os.path.join(output_folder,'skipped_words')+'.txt','w+',encoding='utf8') as f:
        f.writelines([i+'\n' for i in skipped_words])
    print('SKIPPED WORDS LIST SAVED')
    print('PROCESS FINISHED')
    
    
def main(args):
    
    
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder + 'images/')
        os.makedirs(args.output_folder + 'txt/')
        
        # loading the model
    config = Config
    print('LOADING THE MODEL')
    with open(config.data_file, 'rb') as f:  # data file path
        char_map = pkl.load(f)
    char_map=char_map['char_map']
    generator = ImgGenerator(checkpt_path=args.model_path, config=config, char_map=char_map)
    
    with open(args.input_file,'r') as f:
        input_words=f.read().split('\n')
    # gen_images(generator,input_words,args.output_folder)


    language = Config.dataset
    
    MIN_WORD_H, MAX_WORD_H = 31, 64    
    # WORD_H = 20
    PAGE_W, PAGE_H = 1024, 1024
    SPACE_X, SPACE_Y = 32,32
    

    img=gen_images(generator, input_words, language,args.output_folder, MIN_WORD_H, MAX_WORD_H, PAGE_W, PAGE_H, SPACE_X, SPACE_Y, 0)


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