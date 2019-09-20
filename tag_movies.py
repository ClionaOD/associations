'''
Tag movies
Rhodri Cusack cusacklab.org 2018-05-04
Brainhack Ireland
'''

import boto3
import hashlib
import json
import os
from botocore.exceptions import ClientError
from pathlib import Path
import pickle
import numpy as np
from scipy import stats
import s3tools #local file
import videotools #local file
import skvideo
from PIL import Image, ImageDraw, ImageFont
import h5py

def send_movies_to_rekognition_labels(bucket,prefix):
    '''
    Launch label detection on Amazon Rekognition
    Response is sent to SNS which broadcasts to SQS queue
    '''

    # List all objects in the bucket
    s3 = boto3.client('s3')
    allfiles = s3.list_objects(Bucket=bucket, Prefix=prefix)

    # Create rekognition object
    rekognition=boto3.client('rekognition')

    for movie in allfiles['Contents']:
        filename=movie['Key']
        if filename[-4:]=='.mp4': #change according to file type
            print('Working on %s'%filename)
            response = rekognition.start_label_detection(
                Video={'S3Object': {'Bucket': bucket, 'Name': filename}},
                ClientRequestToken=hashlib.md5(str('s3://' + bucket + '/' + filename).encode()).hexdigest(),
                NotificationChannel={
                    'SNSTopicArn': 'arn:aws:sns:eu-west-1:807820536621:AmazonRekognition-movie-associations',
                    'RoleArn': 'arn:aws:iam::807820536621:role/movie-association-role' #change according to SNS and Role Arn
                }
            )
            print('Done this one')



def process_sqs_responses(bucket,sqsqueuename,doevenifdone=False):
    """
    When rekognition finishes running, it posts to an SNS which I have configured to write to an SQS queue
    This script reads the queue, loads the results of rekognition, and formats them into a python structure
    :param bucket: bucket for output data
    :param sqsqueuename: name of SQS queue containing results
    :return:
    """

    sqs=boto3.resource('sqs')
    q=sqs.get_queue_by_name(QueueName=sqsqueuename)
    all_messages=[]
    rs = q.receive_messages()
    while len(rs) > 0:
        all_messages.extend(rs)
        rs = q.receive_messages()
    print("Got %d messages"%len(all_messages))

    # Process them all
    for message in all_messages:
        print(message.body)
        j=json.loads(message.body)
        print(j)
        jm=json.loads(j['Message'])

        if jm['Status']=='SUCCEEDED':
            process_rekognition_video(bucket,jm,doevenifdone = doevenifdone)



def process_rekognition_video(bucket, compmsg, doevenifdone = False):
    """
    Extract details from processed video,
    :param compmsg: SQS message returned by rekogntion
    :return:
    """

    # Get result from rekognition
    rekognition = boto3.client('rekognition')
    jobid=compmsg['JobId']
    vid = compmsg['Video']

    # Coding filename
    outkey_coding = "labels/" + os.path.splitext(vid['S3ObjectName'])[0] + '.pickle'
    outfn_coding = os.path.join(Path.home(), ".s3cache-out", outkey_coding)
    if not os.path.exists(os.path.dirname(outfn_coding)):
        os.makedirs(os.path.dirname(outfn_coding))
    
    s3bucket= boto3.resource('s3').Bucket(vid['S3Bucket'])
    s3client = boto3.client('s3')
    if doevenifdone or not 'Contents' in s3client.list_objects(Bucket=vid['S3Bucket'],Prefix=outkey_coding):
        try:
            if not jobid is None:
                response = rekognition.get_label_detection(JobId=jobid)

                assert response['JobStatus']=='SUCCEEDED', "Rekogntion job status not SUCCEEDED but %s"%response['JobStatus']

                alllabels = response['Labels']
                while 'NextToken' in response:
                    response = rekognition.get_label_detection(JobId=jobid, NextToken=response['NextToken'])
                    alllabels.extend(response['Labels'])
                print("%d labels detected"%len(alllabels))
            else:
                alllabels=compmsg['alllabels']

            # Work out what sampling rekogition seems to be using
            ts= [face['Timestamp'] for face in alllabels]
            difft=np.ediff1d(ts)
            difft=[x for x in difft if not x==0]
            deltat=stats.mode(difft).mode
            print("Delta t is %f"%deltat)

            # Create summary dict
            summary = {'alllabels': alllabels,
                       'vid': vid, 'compmsg': compmsg, 'deltat': deltat}

            # Write coding file
            with open(outfn_coding, 'wb') as f:
                pickle.dump(summary, f)

            # Write annotated video and coding files to s3
            s3bucket.upload_file(outfn_coding, outkey_coding)

            print("Done loading labels")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print("No response from rekognition available for jobid %s" % compmsg['JobId'])
                return False
            else:
                raise

def select_frames(bucket,prefix,TR):
    '''
    Load pickle of label data and downsample to a given TR - typically 1s for HCP
    :param bucket: bucket containing pickle file
    :param prefix: prefix of pickle files
    :param TR: TR to be downsampled to (milliseconds)
    :return:
    '''

    # List all objects in the bucket
    s3 = boto3.client('s3')
    allfiles = s3.list_objects(Bucket=bucket, Prefix=prefix)

    # Create rekognition object
    rekognition=boto3.client('rekognition')

    for movie in allfiles['Contents']:
        key_coding = movie['Key']
        fn_coding = s3tools.getpath({'S3Bucket':bucket, 'S3ObjectName':key_coding})
        if os.path.exists(fn_coding):
            with open(fn_coding, 'rb') as f:
                obj = pickle.load(f)
                fn_labels_ds = "labels_ds/" + os.path.splitext(obj['vid']['S3ObjectName'])[0] + '.pickle'
                if not os.path.exists(os.path.dirname(fn_labels_ds)):
                    os.makedirs(os.path.dirname(fn_labels_ds))

            alllabels_ds=[]
            listlabels=[]

            frametime=0
            ts = [face['Timestamp'] for face in obj['alllabels']]
            ts_unique=sorted(set(ts))
            ind=0
            # Each frame
            while True:
                # Get closest timepoint
                while (ind+1)<len(ts_unique) and (ts_unique[ind+1]-frametime)<(frametime-ts_unique[ind]):
                    ind+=1
                if len(ts_unique)<=ind:
                    break

                # Find all the object labels at this timestamp
                alllabels_ds.append({'timestamp':ts_unique[ind],
                                     'labels':[ x['Label'] for x in obj['alllabels'] if x['Timestamp']==ts_unique[ind]]})
                listlabels.extend([x['Name'] for x in alllabels_ds[-1]['labels']])
                frametime += TR
                ind+=1

            # Get unique objects
            listlabels=list(set(listlabels))

            # Create summary dict
            summary = {'alllabels_ds': alllabels_ds, 'listlabels':listlabels,
                       'vid': obj['vid'], 'TR':TR}

            # Write coding file
            s3bucket = boto3.resource('s3').Bucket(obj['vid']['S3Bucket'])
            with open(fn_labels_ds, 'wb') as f:
                pickle.dump(summary, f)

            # Write annotated video and coding files to s3

            s3bucket.upload_file(fn_labels_ds, fn_labels_ds)


'''
def annotate_movie(bucket,prefix):
    # List all objects in the bucket
    s3 = boto3.client('s3')
    allfiles = s3.list_objects(Bucket=bucket, Prefix=prefix)


    fhdf = h5py.File('/home/cusackrh/Dropbox/projects/brainhack2018/movie_resources/WordNetFeatures.hdf5', 'r')

    # List all groups
    file_keys = list(fhdf.keys())
    data_synsets=list(fhdf['synsets'])

    # Create rekognition object
    rekognition=boto3.client('rekognition')
    for movie in allfiles['Contents']:

        # Get HCP labels
        data_cc = list(fhdf[movie['Key'][-20:-10]])

        key_coding = movie['Key']
        fn_coding = s3tools.getpath({'S3Bucket':bucket, 'S3ObjectName':key_coding})
        if os.path.exists(fn_coding):
            with open(fn_coding, 'rb') as f:
                obj = pickle.load(f)

                # Annotated filename
                outkey_annotated = "annotated/" + os.path.splitext(obj['vid']['S3ObjectName'])[0] + '.mp4'
                outfn_annotated=os.path.join(Path.home(), ".s3cache-out", outkey_annotated)

                v = videotools.Video(obj['vid'])

                if v._pth is None or not os.path.exists(v._pth):
                    print("Video not found")
                    return False
                else:
                    v.open()
                    dur = v.get_dur()
                    fps = v.get_fps()
                    print("Dur %f and FPS %f"%(dur,fps))

                    # Make directory if necessary
                    dirname = os.path.dirname(outfn_annotated)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)

                    writer = skvideo.io.FFmpegWriter(outfn_annotated)

                    # Get frame
                    ind=0
                    while v.isopen:
                        img=v.get_next_frame()
                        
                        if not img:
                            break

                        currtime_ms=np.round(v.currtime*1000)

                        # Label timestamp must be at current time or later
                        while (ind+1)<len(obj['alllabels_ds']) and obj['alllabels_ds'][ind+1]['timestamp']<=currtime_ms:
                            ind+=1

                        # Add space for labels
                        newImage = Image.new('RGB', (1920, 720))
                        newImage.paste(img, (0, 0, 1024, 720))

                        # Is label still current?
                        if ind<len(obj['alllabels_ds']) and currtime_ms-obj['alllabels_ds'][ind]['timestamp']<1000:

                            draw = ImageDraw.Draw(newImage)
                            # font = ImageFont.truetype(<font-file>, <font-size>)
                            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20,
                                                      encoding="unic")
                            for itemind,item in enumerate(obj['alllabels_ds'][ind]['labels']):
                                draw.text((1100, itemind*16),item['Name'] , (128, 128, 255), font=font)
                                

                        #if ind<len(data_cc):
                            hcplabels=[data_synsets[item[0]] for item in enumerate(data_cc[ind]) if item[1]==1]
                            for itemind,item in enumerate(hcplabels):
                                #draw.text((1400, itemind*16),item.decode('ascii'), (255, 128, 128), font=font)

                        # Write annotated frame
                           # writer.writeFrame(newImage)

                    #writer.close()


                    s3bucket = boto3.resource('s3').Bucket(obj['vid']['S3Bucket'])
                    s3bucket.upload_file(outfn_annotated, outkey_annotated)

                    print('Done writing video')
'''

if __name__=='__main__':
    bucket="movie-associations"
    prefix="movies-five"

# This sends to movies to rekognition
#send_movies_to_rekognition_labels(bucket, prefix)

# When they're done, process the responses (only run this after SNS has sent email verifying completion)
process_sqs_responses(bucket,'AmazonRekognition-movie-association-sqs',doevenifdone=False)

#don't need to call process_rekognition_video because process_sqs_responses does this if jm['STATUS']=='SUCCEEDED'

""" Annotating the movie files is not necessary
    # Downsample to TR
    #select_frames(bucket,'labels',2000)

    #annotate_movie(bucket,'labels_ds')
"""