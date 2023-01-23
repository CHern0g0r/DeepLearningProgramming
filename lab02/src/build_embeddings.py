import os
import torch
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from decord import VideoReader, cpu
from transformers import VideoMAEImageProcessor, VideoMAEModel
from tqdm import tqdm


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def build_embedding(video, model, image_processor, aggregate, device):
    inputs = image_processor(list(video), return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    emb = None
    if aggregate:
        emb = last_hidden_states.cpu().detach().mean(1).numpy().squeeze(0)
    else:
        emb = last_hidden_states.cpu().detach()[:, 0].numpy().squeeze(0)
    return emb


def get_video(file_path):
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)
    indices = sample_frame_indices(
        clip_len=16,
        frame_sample_rate=4,
        seg_len=len(videoreader)
    )
    video = videoreader.get_batch(indices).asnumpy()
    return video


def main(*,
         videos_path,
         output_file,
         model_path,
         cuda,
         aggregate):
    np.set_printoptions(threshold=1200)
    
    image_processor = VideoMAEImageProcessor.from_pretrained(model_path)
    model = VideoMAEModel.from_pretrained(model_path)

    device = torch.device('cpu')
    if not cuda == -1:
        device = torch.device(f'cuda:{cuda}')
        model.to(device)

    data = {
        'id': [],
        'emb': []
    }
    pd.DataFrame(data).to_csv(output_file, index=False)
    for i, dir_name in enumerate(tqdm(os.listdir(videos_path))):
        dir_path = os.path.join(videos_path, dir_name)
        dirlist = os.listdir(dir_path)
        if not dirlist:
            continue
        filename = dirlist[0]
        file_path = os.path.join(dir_path, filename)
        try:
            video = get_video(file_path)
        except Exception:
            continue
        emb = build_embedding(
            video,
            model,
            image_processor,
            aggregate,
            device
        )

        data['id'] += [dir_name]
        data['emb'] += [emb]
        if i % 100 == 0:
            pd.DataFrame(data).to_csv(output_file, index=False)
    print(f'Write to: {output_file}')
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('videos_path')
    parser.add_argument('output_file')
    parser.add_argument(
        '--model_path',
        default="/srv/nfs/VESO/fedor/labs/prog2/models/videomae-base"
    )
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--aggregate', action='store_true')

    args = parser.parse_args()

    main(**vars(args))
