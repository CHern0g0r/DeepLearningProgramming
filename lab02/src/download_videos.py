## Pre-requisities: run 'pip install youtube-dl' to install the youtube-dl package.
## Specify your location of output videos and input json file.
import json
import os

from tqdm import tqdm
from pytube import YouTube


def download(output_path, json_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    data = json.load(open(json_path, 'r'))['database']

    for i, youtube_id in enumerate(tqdm(data)):
        info = data[youtube_id]
        clas = info['recipe_type']
        url = info['video_url']
        vid_loc = os.path.join(output_path, f'{youtube_id}')
        # if not os.path.exists(vid_loc):
        #     os.mkdir(vid_loc)

        try:
            yt = YouTube(url).streams.filter(
                res="360p",
                progressive=True,
                file_extension='mp4'
            ).order_by('resolution').desc().first()
            yt.download(vid_loc)
        except AttributeError as ae:
            yt = YouTube(url).streams.filter(
                progressive=True,
                file_extension='mp4'
            ).order_by('resolution').desc().first()
            yt.download(vid_loc)
        except Exception as e:
            print(e)
            print('Failed:', youtube_id)
            print('-'*50)
        
#     os.system('youtube-dl -o ' + vid_loc + '/' + youtube_id + '.mp4' + ' -f best ' + url)
    
if __name__ == '__main__':
    output_path = '../data/videos'
    json_path = '../data/COIN.json'

    import ssl
    import os
    ssl._create_default_https_context = ssl._create_unverified_context

    # os.environ |= {
    #     'https_proxy': 'http://10.198.126.162:3128',
    #     'http_proxy': 'http://10.198.126.162:3128',
    #     'cntlm_proxy': '10.198.126.162:3128',
    #     'GLOBAL_AGENT_HTTP_PROXY': 'http://10.198.126.162:3128',
    #     'CURL_CA_BUNDLE': '',
    #     'TRANSFORMERS_CACHE': '/srv/data/models'
    # }

    download(output_path, json_path)
