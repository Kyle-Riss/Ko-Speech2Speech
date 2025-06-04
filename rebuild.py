import os
import json
import shutil

source_ann_dir = '/Users/hayubin/Downloads/BDD100K/train/ann/'
source_img_dir = '/Users/hayubin/Downloads/BDD100K/train/img/'
target_base_dir = '/Users/hayubin/Downloads/Project dataset/train/'

target_classes = ["car", "person", "traffic light", "traffic sign"]
max_images_per_class = 750  
images_collected_count = {cls: 0 for cls in target_classes}

for cls_name in target_classes:
    class_folder_path = os.path.join(target_base_dir, cls_name)
    if not os.path.exists(class_folder_path):
        os.makedirs(class_folder_path)
        print(f"폴더 생성: {class_folder_path}")
    else:
        print(f"폴더 이미 존재 (또는 이전 삭제 후 재생성됨): {class_folder_path}")


print("이미지 선별 및 복사를 시작합니다 (목표: 클래스당 750장)...")
processed_files = 0
annotation_files = os.listdir(source_ann_dir)

for ann_filename in annotation_files:
    if not ann_filename.endswith('.json'):
        continue

    if all(count >= max_images_per_class for count in images_collected_count.values()):
        print("모든 클래스에 대해 목표 이미지 수를 수집했습니다. 종료합니다.")
        break

    ann_file_path = os.path.join(source_ann_dir, ann_filename)
    image_actual_filename = ann_filename.replace('.json', '') 
    source_image_path = os.path.join(source_img_dir, image_actual_filename)
    
    if not os.path.exists(source_image_path):
        continue
    
    processed_files += 1
    if processed_files % 2000 == 0: 
        print(f"{processed_files}개 어노테이션 파일 처리 중...")
        for cls_name_temp, count_temp in images_collected_count.items():
            print(f"  - 클래스 '{cls_name_temp}': {count_temp} / {max_images_per_class} 장")

    try:
        with open(ann_file_path, 'r') as f:
            annotation = json.load(f)

        found_classes_in_this_image = set()
        if 'objects' in annotation:
            for obj in annotation['objects']:
                obj_class_title = obj.get('classTitle')
                if obj_class_title in target_classes:
                    found_classes_in_this_image.add(obj_class_title)
        
        for cls_name in found_classes_in_this_image:
            if images_collected_count[cls_name] < max_images_per_class:
                target_class_folder = os.path.join(target_base_dir, cls_name)
                target_image_path = os.path.join(target_class_folder, image_actual_filename)
                
                if not os.path.exists(target_image_path): 
                    shutil.copy2(source_image_path, target_image_path)
                    images_collected_count[cls_name] += 1

    except Exception as e:
        print(f"오류 발생 (파일: {ann_filename}): {e}")
        continue

print("\n--- 이미지 선별 작업 완료 ---")
for cls_name, count in images_collected_count.items():
    print(f"클래스 '{cls_name}': {count} 장 복사됨")
print(f"총 처리된 어노테이션 파일 수: {processed_files}")