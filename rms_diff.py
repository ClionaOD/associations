import os
import math
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import skvideo.io
import boto3
import cv2

s3 = boto3.client('s3')

def load_video(infn, clipheight=None):
    # Load metadata
    metadata = skvideo.io.ffprobe(infn)
    dur=float(metadata['video']['@duration'])       
    nframes=int(float(metadata['video']['@nb_frames']))
    fps=nframes/dur
    print('Video %s duration %f nframes %d fps %f'%(infn,dur,nframes,fps))

    # Load a video
    singlevideo=skvideo.io.vread(infn)
    singlevideo=singlevideo.astype(np.uint8)

    # Clip height?
    if clipheight:
        h=singlevideo.shape[1]
        singlevideo=singlevideo[:,round(h/2-clipheight/2):round(h/2+clipheight/2),:,:]
        metadata['video']['@height']=clipheight
        print('Clipped height to %s'%clipheight)
    
    return metadata,singlevideo,dur,fps

def rmsdiff(im1, im2, frame_id):
    if isinstance(im1, np.ndarray):
        im1 = Image.fromarray(im1).convert('RGB')
        im2 = Image.fromarray(im2).convert('RGB')
    diff = ImageChops.difference(im1,im2)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return h,rms

framewise_rms = {}
nsecs = 60
secinterval = 1

result = s3.list_objects_v2(Bucket='movie-associations', Prefix='Rekognition Tagging/allmovies')
for o in result.get('Contents'):
    key= o.get('Key')
    if '.mp4' in key:
        _mov = key.split('/')[-1].split('.')[0]
        print(f'working on {_mov}')

        url = s3.generate_presigned_url('get_object', 
                                        Params = {'Bucket': 'movie-associations', 'Key': key}, 
                                        ExpiresIn = 600) #this url will be available for 600 seconds
        cap = cv2.VideoCapture(url)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        first_frame = None
        rms_results = []
        
        while True:
            
            for i in range(0, nsecs, secinterval):
                # if len(rms_results) % 100 == 0:
                #     print(f'..{len(rms_results)} frames done')
                
                frame_id = i*fps
                cap.set(1,frame_id)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if first_frame is None:
                    first_frame = frame

                _, rms = rmsdiff(first_frame, frame, frame_id)
                rms_results.append(rms)
    
            break

        cv2.waitKey(100) #Wait 100msec (for debugging)
        cap.release() #Release must be inside the outer loop
        
        framewise_rms[_mov] = rms_results
    cv2.destroyAllWindows()

    with open(f'framewise_rms_allmovies_{nsecs}sec_every{secinterval}.pickle','wb') as f:
        pickle.dump(framewise_rms,f)

rms_df = pd.DataFrame.from_dict(framewise_rms)
rms_df.to_csv(f'./framewise_rms_allmovies_{nsecs}sec_every{secinterval}.csv')