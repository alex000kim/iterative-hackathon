import hashlib
import json
from fastai.vision.all import *


def inference(data_path, model_pickle_path, predictions_folder):
    files = get_image_files(data_path/"labelbook")
    learn = load_learner(model_pickle_path, cpu=True)
    for file_path in files:
        with open(file_path, 'rb') as file_to_check:
            # read contents of the file
            data = file_to_check.read()
            # pipe contents of the file through
            md5 = hashlib.md5(data).hexdigest()

        pred_label, _, probs = learn.predict(file_path)
        confidence = np.array(probs).max()
        #true_label = class_names[ test_labels[index] ]
        file_name = file_path.stem

        json_string = f'{{ "annotation": {{ "inference": {{ "label": "{pred_label}", "confidence": {confidence} }} }}, "data-object-info": {{ "md5": "{md5}" }} }}'
        # print(json_string)
        json_data = json.loads(json_string)
        with open(predictions_folder/f'{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    data_path = Path('data')
    predictions_folder = Path('predictions')
    predictions_folder.mkdir(exist_ok=True)
    model_dir = Path('model')
    model_dir.mkdir(exist_ok=True)
    model_pickle_path = model_dir/'model.pkl'
    inference(data_path,
              model_pickle_path,
              predictions_folder)
