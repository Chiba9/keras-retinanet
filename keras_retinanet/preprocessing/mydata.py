import json
import cv2
import os

image_path = 'content 
f = open('./665-verified.json')
content = f.read()
f.close()
mydic = json.loads(content)
cocodic = {}
f = open('./instances_train2017.json')
content = f.read()
f.close()
coco_example = json.loads(content)

#coco_example = cocodic

cocodic = {}
cocodic['categories'] = coco_example['categories']
cocodic['images'], cocodic['annotations'] = [], []

img_id, anno_id = 0, 0

for (key, image_info) in mydic['_via_img_metadata'].items():
    imagedic_coco = {}
    imagedic_coco['file_name'] = image_info['filename']
    imagedic_coco['id'] = img_id
    
    image = cv2.imread(os.path.join(image_path, image_info['filename']))
    imagedic_coco['height'], imagedic_coco['width'] = image.shape[0], image.shape[1]
    cocodic['images'].append(imagedic_coco)
    
    for region in image_info['regions']:
        anno_coco = {}
        if region['region_attributes']['name'] == 'person':
            anno_coco['category_id'] = 1
        else:
            anno_coco['category_id'] = -1
        anno_coco['id'] = anno_id
        anno_id += 1
        mybbox = region['shape_attributes']
        anno_coco['bbox'] = [mybbox['x'], mybbox['y'], mybbox['width'] - 1, mybbox['height'] - 1]
        anno_coco['image_id'] = img_id
        anno_coco['iscrowd'] = 0
        cocodic['annotations'].append(anno_coco)
    
    img_id += 1

coco_json = json.dumps(cocodic)
with open('coco_json.json','w',encoding='utf-8') as f:#使用.dumps()方法时，要写入
    f.write(coco_json)
    f.close()