#sensitive variables in config.py file that is on .gitignore
from config import key_, secret_, s3_bucket, kaggle_cookie

import boto3

def get_video_link(video_name, aws_key=key_, aws_secret=secret_, bucket=s3_bucket):
    '''
    ##Intended for use when not using Sagemaker##
    takes a video name as input, and returns a downloaded video from s3 bucket in an array
    '''
    s3 = boto3.client('s3',
                      aws_access_key_id=aws_key, 
                      aws_secret_access_key=aws_secret,
                      region_name='us-east-2', #region is hardcoded - this is not a security risk to keep public
                      config= boto3.session.Config(signature_version='s3v4')) #the sig version needs to be s3v4 or the url will error
    video_url = s3.generate_presigned_url('get_object',
                                        Params={"Bucket": bucket,
                                               'Key': video_name},
                                        ExpiresIn=6000)
    return video_url