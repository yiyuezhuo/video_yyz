
from torch.utils.data.dataloader import default_collate


def _val2vl(batch):
    # (video, audio, label) -> (video, label)
    # remove audio from the batch
    batch = [(d[0], d[2]) for d in batch]
    return default_collate(batch)

def val2vl():
    return _val2vl