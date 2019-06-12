import json

categories = [
    {'id': 1,'supercategory': 'clothing', 'name': 'bag'}, 
    {'id': 2,'supercategory': 'clothing', 'name': 'boots'},
    {'id': 3,'supercategory': 'clothing', 'name': 'footwear'},
    {'id': 4,'supercategory': 'clothing', 'name': 'outer'},
    {'id': 5,'supercategory': 'clothing', 'name': 'sunglasses'},
    {'id': 6,'supercategory': 'clothing', 'name': 'pants'},
    {'id': 7,'supercategory': 'clothing', 'name': 'top'},
    {'id': 8,'supercategory': 'clothing', 'name': 'shorts'},
    {'id': 9,'supercategory': 'clothing', 'name': 'headwear'},
    {'id': 10,'supercategory': 'clothing', 'name': 'scarf&tie'}
]

class COCOSet:
    def __init__(self, parameters_info = {}):
        self.dataset = {
            "info": {
                "description": "generated dataset",
                "parameters": parameters_info
                },
            "images": [],
            "annotations": [],
            "categories": categories,
        }
    
    def allocate_image(self, image_size):
        image_id = len(self.dataset['images'])
        image = {"id": image_id,
                "file_name": "{}.jpg".format(image_id),
                "width": image_size[0],
                "height": image_size[1]
                }
        self.dataset['images'].append(image)
        return image

    def add_annotation(self, annotation):
        annotation['id'] = len(self.dataset['annotations'])
        self.dataset['annotations'].append(annotation)

    def write_annotations(self, path):
        with open(path, 'w') as f:
            json.dump(self.dataset, f)
