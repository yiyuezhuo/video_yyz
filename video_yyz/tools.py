'''
Provide quick reference to some CLI tools. Some of them may be exposed lately.

For examples,
ipython -i -m video_yyz.tools
>>> split_video(...)

require a installed ffmpeg, PyAV conda package will install a ffmpeg, or install
it from scratch.

Selected tools:
    split_video: split video into frames
    generate_index: generat index used by fast dataset class
    resize_video: resize video into new size

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
import shutil


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
        target_path = target_root_path / r_path.with_suffix('')  # with_suffix is used to remove '.mp4'
        target_path.mkdir(exist_ok=True, parents=True)
        
        subprocess.run(['ffmpeg', '-i', str(path), '-f', 'image2', f'{str(target_path)}/{template}'])
        if verbose:
            print(f"{path} -> {target_path}")


def collect_index(root, extension='mp4'):
    root_path = Path(root)

    # glob_pattern = f'**/*.{extension}'  # don't work if symbol link is used
    glob_pattern = f'*/*.{extension}'  # further assume data structure

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
                   train_val_name='train_val.json', percent=0.7, extension='mp4', verbose=True):
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
    target_root = Path(root)

    train_path = target_root / train_name
    val_path = target_root / val_name
    train_val_path = target_root / train_val_name

    index = collect_index(root, extension=extension)

    with train_val_path.open('w', encoding='utf8') as f:
        json.dump(index, f)

    classes, samples = index['classes'], index['samples']
    if len(classes)==0:
        raise ValueError("dataset contain zero video!")

    if verbose:
        print(f"Collect {len(samples)} items, {len(classes)} classes")
        if len(classes) <= 10:
            print("classes: ", classes)
        else:
            print("classes (first 10):", classes[:10])
    
    random.shuffle(samples)
    cutoff = int(len(samples)*percent)
    samples_train = samples[:cutoff]
    samples_val = samples[cutoff:]

    if len(samples_train)==0:
        raise ValueError("Train set contain zero video!")
    if len(samples_val)==0:
        raise ValueError("Val set contain zero video!")

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
        for name, _index in zip(['train', 'val', 'train_val'], [index_train, index_val, index]):
            counter = Counter([label_idx for path, label_idx in _index['samples']])
            print(name, counter)


def resize_video(root, target_root, size, extension='mp4', verbose=True):
    '''
    resize video and save them into a new folder
    Size: (width, height)
    
    For example, to convert 1920x1080 video into 960x540:
        >>> resize_video(root, target_root, (960, 540))
    '''
    root_path = Path(root)
    target_root_path = Path(target_root)

    scale = f'scale={size[0]}:{size[1]}'

    # for path in root_path.glob(f'**/*.{extension}'):
    for path in root_path.glob(f'*/*.{extension}'):
        r_path = path.relative_to(root_path)
        target_path = target_root_path / r_path

        target_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg', '-i', str(path), '-vf', scale, f'{str(target_path)}'])

        if verbose:
            print(f'{path} -> {target_path}')


def merge_dataset(root1, root2, target_root, extension='mp4'):
    '''
    Require Administrator in Windows, what sad!
    Assumed file structure:
        root1
            train.json
            ...
            dry
                1.mp4
                ...
            melt
                ...
            normal
                ...
        root2
            ...
        
        ->
        target_root
            train.json (merged)
            ...
            dry
                1.mp4
            ...
    '''
    root1 = Path(root1)
    root2 = Path(root2)
    target_root = Path(target_root)
    print("root1:", root1)
    print("root2:", root2)
    print("target_root:", target_root)

    target_root.mkdir(parents=True, exist_ok=True)

    for name in ['train.json', 'val.json', 'train_val.json']:
        with (root1 / name).open() as f:
            dat1 = json.load(f)
        with (root2 / name).open() as f:
            dat2 = json.load(f)
        
        assert dat1['classes'] == dat2['classes']

        sam1 = map(tuple, dat1['samples'])
        sam2 = map(tuple, dat2['samples'])
        set_sam1 = set(sam1)
        set_sam2 = set(sam2)
        set_sam1_sam2 = set_sam1 | set_sam2
        sam_merged = sorted(list(set_sam1_sam2))
        dup = len(set_sam1) + len(set_sam2) - len(set_sam1_sam2)
        print('# sam1:', len(set_sam1), "# sam2:", len(set_sam2))
        print("# merged:", len(sam_merged), "# duplicated:", dup)
        if dup > 0:
            ex = next(iter(set_sam1 & set_sam2))
            print("Duplicate example:", ex)
        
        with (target_root / name).open('w') as f:
            merged = dict(classes=dat1['classes'], samples=sam_merged)
            json.dump(merged, f)
    
    skip_count = 0
    link_count = 0
    for root in [root1, root2]:
        for p in root.glob(f"*/*.{extension}"):
            target_path = target_root / p.relative_to(root)
            if target_path.exists():
                print("Skip existed", target_path)
                skip_count += 1
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            #p.symlink_to(target_path)
            target_path.symlink_to(p)
            link_count += 1
    print("Linked:", link_count, "Skip:", skip_count)


def clone_dataset(root, target_root):
    '''
    symbol link all folder, and copy all json files to target_root
    '''
    root_path = Path(root)
    target_root_path = Path(target_root)
    root_path = root_path.absolute()
    target_root_path = target_root_path.absolute()

    target_root_path.mkdir(parents=True, exist_ok=True)

    for json_path in root_path.glob("*.json"):
        target_json_path = target_root_path / json_path.relative_to(root_path)
        shutil.copy(json_path, target_json_path)
        print(f"Copied {json_path} -> {target_json_path}")
    for p in root_path.iterdir():
        if p.is_dir():
            p = p.absolute()
            target_dir_path = target_root_path / p.relative_to(root_path)
            target_dir_path.symlink_to(p)
            print(f"Made symbolink {target_dir_path} -> {p}")
    # Yes, the "->" is reversed, traditionally.


def shuffle_dataset(root):
    '''
    Shuffle label only.
    Used to check whether model alway overfit, even under unreasonble random shuffle.
    '''
    root_path = Path(root)
    with (root_path / "train_val.json").open() as f:
        train_val = json.load(f)
    path_list, label_list = zip(*train_val['samples'])
    label_list = list(label_list)  # tuple -> list
    random.shuffle(label_list)
    path_to_label = {p: l for p, l in zip(path_list, label_list)}
    with (root_path / "train_val.json").open('w') as f:
        new_train_val = {
            'classes': train_val['classes'],
            'samples': list(zip(path_list, label_list))
        }
        json.dump(new_train_val, f)

    for name in ['train.json', 'val.json']:
        with (root_path / name).open() as f:
            dat = json.load(f)
        path_list, label_list = zip(*dat['samples'])
        label_list = [path_to_label[p] for p in path_list]
        with (root_path / name).open('w') as f:
            new_dat = {
                'classes': dat['classes'],
                'samples': list(zip(path_list, label_list))
            }
            json.dump(new_dat, f)
    #import pdb;pdb.set_trace()


#if __name__ == '__main__':
# export some name for convenience
SVD = ENV.get('SVD')  # steel_video_dataset
if SVD is None:
    print("Not set SVD")
if SVD:
    SVD = Path(SVD)
    
    video_sample = SVD / 'video_sample'
    video_sample_split = SVD / 'video_sample_split'

    video_sample_mini = SVD / 'video_sample_mini'
    video_sample_mini_split = SVD / 'video_sample_mini_split'

    # small x0.5  -> (960, 540)
    video_sample_mini_small = SVD / 'video_sample_mini_small'
    video_sample_mini_small_split = SVD / 'video_sample_mini_small_split'

    # tiny x0.25  -> (480, 270)
    video_sample_mini_tiny = SVD / 'video_sample_mini_tiny'
    video_sample_mini_tiny_split = SVD / 'video_sample_mini_tiny_split'