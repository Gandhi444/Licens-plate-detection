import argparse
import json
from pathlib import Path

import cv2

from procesing.main import perform_processing


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('images_dir', type=str)
    # parser.add_argument('results_file', type=str)
    # args = parser.parse_args()

    #images_dir =Path(args.images_dir)
    #results_file = Path(args.results_file)

    images_dir =Path("./data")
    results_file =Path("res.json")
    template_path=Path("procesing/templates")
    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    templates_paths = sorted([image_path for image_path in template_path.iterdir() if image_path.name.endswith('.png')])
    results = {}
    template_path="templates"
    templates=[]
    for template_path in templates_paths:
        template = cv2.imread(str(template_path),cv2.IMREAD_GRAYSCALE)
        templates.append((template_path.name[-5],template))
        if template is None:
            print(f'Error loading template {template_path}')
            continue
    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue
        try:
            results[image_path.name] = perform_processing(image,templates)
        except:
            print('exception while proccesing')
            results[image_path.name]="POAAAAA"
    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()