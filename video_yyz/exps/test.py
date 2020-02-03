import subprocess

subprocess.run([
    'python', '-m', 'video_yyz.train',
    '--train', 'train_video_dataset_1', 'transform_train_1', 'val2vl', 'video_random_1',
    '--test', 'test_video_dataset_1', 'transform_test_1', 'val2vl', 'video_uniform_1',
    '--model', 'r2plus1d_18_1',
    '--optimizer', 'sgd_1',
    '--num-epoch', 30,
])
