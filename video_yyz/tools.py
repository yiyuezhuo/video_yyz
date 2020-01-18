'''
Provide quick reference to some CLI tools. Some of them may be exposed lately.

For examples,
ipython -i -m video_yyz.tools
>>> split_video(...)

require a installed ffmpeg, PyAV conda package will install a ffmpeg, or install
it from scratch.

'''
from pathlib import Path
import subprocess
import sys
import os
from os import environ as ENV
from pathlib import Path
import json
import random
from collections import Counter


def split_video(root, target_root, template, verbose=True):
    '''
    Use ffmpeg 
    template example: 'image-%05d.jpg'

    cat/cat1.mp4
    ->
    cat_split/cat1/image-00001.jpg
    cat_split/cat1/image-00002.jpg
    ...

    '''
    root_path = Path(root)
    target_root_path = Path(target_root)

    for path in root_path.glob('**/*.mp4'):
        r_path = path.relative_to(root_path)
        target_path = target_root_path / r_path.with_suffix('')  # stem is used to remove '.mp4'
        target_path.mkdir(exist_ok=True, parents=True)
        
        subprocess.run(['ffmpeg', '-i', str(path), '-f', 'image2', f'{str(target_path)}/{template}'])
        if verbose:
            print(f"{path} -> {target_path}")


def collect_index(root, extension='mp4'):
    root_path = Path(root)

    glob_pattern = f'**/*.{extension}'

    label_list = []
    r_path_list = []
    for path in root_path.glob(glob_pattern):
        r_path = path.relative_to(root_path)
        label = r_path.parts[0]

        r_path_list.append(r_path)
        label_list.append(label)

    classes = sorted(list(set(label_list)))
    class_to_idx = {label:idx for idx, label in enumerate(classes)}

    samples = [[str(r_path), class_to_idx[label]] for r_path, label in zip(r_path_list, label_list)]
    return dict(
        classes=classes,
        samples=samples
    )


def generate_index(root, target_root=None, train_name='train.json', val_name='val.json',
                   percent=0.7, extension='mp4', verbose=True):
    '''
    Generate index files (train.json, val.json).
    Path is relative to root to maintain some portability. 

    Assumed structure:
        cat/cat1.mp4
        cat/cat2.mp4
        dog/dog1.mp4
        ....

    Generated index:
        classes -> ['cat', 'dot', ...]  (alphabetic ordering)
        samples -> List[(relative_path, class_idx)]  (class_idx is an int, 0 denotes cat, etc)
    '''
    if target_root is None:
        target_root = root
    root_path = Path(root)
    train_path = target_root / train_name
    val_path = target_root / val_name

    index = collect_index(root, extension=extension)
    classes, samples = index['classes'], index['samples']

    if verbose:
        print(f"Collect {len(samples)} items, {len(classes)} classes")
        if len(classes) < 10:
            print("classes: ", classes)
    
    random.shuffle(samples)
    cutoff = int(len(samples)*percent)
    samples_train = samples[:cutoff]
    samples_val = samples[cutoff:]

    index_train = dict(classes=classes, samples=samples_train)
    index_val = dict(classes=classes, samples=samples_val)

    if verbose:
        print(f'train size: { len(index_train["samples"]) }')
        print(f'val size: { len(index_val["samples"]) }')
    
    with train_path.open('w', encoding='utf8') as f:
        json.dump(index_train, f)
    with val_path.open('w', encoding='utf8') as f:
        json.dump(index_val, f)
    
    if verbose:
        print(f'Created: {train_path} {val_path}')

    if verbose:
        for name, _index in zip(['train', 'val'], [index_train, index_val]):
            counter = Counter([label_idx for path, label_idx in _index['samples']])
            print(name, counter)


if __name__ == '__main__':
    # export some name for convenience
    SVD = ENV.get('SVD')  # steel_video_dataset
    if SVD:
        SVD = Path(SVD)
        video_sample = SVD / 'video_sample'
        video_sample_mini = SVD / 'video_sample_mini'
        video_sample_split = SVD / 'video_sample_split'
        video_sample_mini_split = SVD / 'video_sample_mini_split'
