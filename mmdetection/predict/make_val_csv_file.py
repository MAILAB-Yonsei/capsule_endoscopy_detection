import os

def convert_coco_json_to_csv(filename):
    import pandas as pd
    import json
    
    s = json.load(open(filename, 'r'))
    out_file =  'predict/val_answer.csv'
    out = open(out_file, 'w')
    out.write('file_name, class_id, confidence, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y\n')

    for ann in s['annotations']:
        a = str(ann['image_id'])
        image_id = f'train_{a}.json'
        
        label = ann['category_id']
        confidence = 1.0
        
        x1 = ann['bbox'][0]
        y1 = ann['bbox'][1]
        
        x2 = ann['bbox'][0] + ann['bbox'][2]
        y2 = ann['bbox'][1]
        
        x3 = ann['bbox'][0] + ann['bbox'][2]
        y3 = ann['bbox'][1] + ann['bbox'][3]
        
        x4 = ann['bbox'][0]
        y4 = ann['bbox'][1] + ann['bbox'][3]
        
        out.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(image_id,label,confidence,x1,y1,x2,y2,x3,y3,x4,y4))

    out.close()

    # Sort file by image id
    s1 = pd.read_csv(out_file)
    s1.sort_values('file_name', inplace=True)
    s1.to_csv(out_file, index=False)
    
base_path = "../data_coco" 
val_img =  "valid" 
val_anno = "valid_annotations.json" 

val_img_path = os.path.join(base_path, val_img)
val_anno_path = os.path.join(base_path, val_anno)

convert_coco_json_to_csv(val_anno_path)