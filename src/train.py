import argparse
from fastai.vision.all import *
from dvclive.fastai import DvcLiveCallback
from load_params import load_params


def train_model(data_path, model_pickle_path, params_path):
    params = load_params(params_path)
    dls = ImageDataLoaders.from_folder(data_path,
                                       train='train',
                                       valid='val',
                                       bs=params.batch_size,
                                       item_tfms=Resize(256))

    learn = vision_learner(dls, resnet50, metrics=accuracy)
    early_stopping_cb = EarlyStoppingCallback(
        monitor='accuracy',
        min_delta=0.1,
        patience=2
    )
    dvclive_cb = DvcLiveCallback(path='dvclive')
    learn.fine_tune(params.n_epochs, cbs=[early_stopping_cb, dvclive_cb])
    learn.export(fname=Path(model_pickle_path).absolute())


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_path = 'data'
    model_dir = Path('model')
    model_dir.mkdir(exist_ok=True)
    model_pickle_path = model_dir/'model.pkl'
    train_model(data_path, model_pickle_path, params_path=args.config)
