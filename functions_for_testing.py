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

def grab_frame(video_link, skipped_frames=5):
    '''
    function that takes a link to a video, and returns the frame after 'skipped_frames' input variable
    temporary function to prevent large amount of bucket queries -- combine with resize and detect image function later
    '''
    video = cv.VideoCapture(video_link)
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    for skipped_frame in np.arange(0, (skipped_frames + 1)):
        _ = video.grab()
    _, frame = video.retrieve()
    video.release()
    return frame
# look into improving this - 701 ms when loading from bucket, 50 ms when loading from file, 5 skipped frames


def resize_and_detect_face(frame, new_max_size=750, padding=(.1, 0.05, 0.05)):
    '''
    temporary function -- combine with grab frame later
    -- want to reduce number of bucket queries--
    inputs:
    frame: a single frame or an image
    new_max_size: the maximum size of the longer of the width/height the frame will be resized to prior
    to looking for faces
    padding: tuple of percentages; will be added to the size of the face to ensure the entire face is captured
    -- the tuple is (top, bottom, horizontal)
    the top param will move the top of the face by this param times the size of the face towards the top of the y axis
    the bottom param will move the bottom of the face by this praram times the size of the face towards the bottom
    the horizontal param will move the left and right edges of the face by this param towards the left and
    right edges of the plane respectively
    returns:
    a list of arrays
    each array is a cropped face with dimensions of 146 by 225 pixels
    '''
    #convert the frame to color
    #unsure if this step is necessary, however cvtColor takes very little time (~200 µs )
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    original_height = frame.shape[0]
    original_width = frame.shape[1]
    #get original shape of frame
    original_height, original_width = frame.shape[0], frame.shape[1]
    #get aspect ratio -- want to maintain this
    img_size_ratio = original_height / original_width
    #if the height is greater than the width, make new height the new_max_size, and
    #make new width the new height divided by the aspect ratio
    if original_height > original_width:
        new_height = new_max_size
        new_width = new_height / img_size_ratio
    #otherwise, make the new width equal to the new max size, and 
    #the new height the new width times the aspect ratio
    else:
        new_width = new_max_size
        new_height = new_width * img_size_ratio
    #new dimensions -- the aspect ratio will not match exactly due to rounding, but will be close
    new_dim = (int(new_width), int(new_height))
    #resize the image while maintaining the aspect ratio, and changing the maximum edge length to new_max_size
    resized_image = cv.resize(img, new_dim, interpolation = cv.INTER_AREA)
    face_dictionaries = face_detector.detect_faces(resized_image)
    faces = []
    for face in range(len(face_dictionaries)):
        #only review faces that have more than a 90% confidence of being a face
        if face_dictionaries[face]['confidence'] > 0.9:
            #the 'box' of the face is a list of pixel values as: '[x, y, width, height]'
            box = face_dictionaries[face]['box']
            #this is the left side of the face. This will look at the x 'box' value, and will move left by the 
            #percentage of the horizontal padding param
            start_x = box[0] - (padding[2] * box[2])
            #right side of the face. Will add the horizontal padding param to the width and add the result to the 
            #original x starting value
            end_x = box[0] + ((1 + padding[2]) * box[2])
            #bottom of face
            start_y = box[1] - (padding[1] * box[3])
            #top of face
            end_y = box[1] + ((1 + padding[0]) * box[3])
            #if the adjusted x starting value is negative, change the starting x value to 0 (the 0 index of the frame array)
            if start_x < 0:
                start_x = 0
            if start_y < 0:
                start_y = 0
            #keep consistant - do additional research on this
            face_ratio = 1.54 # will keep horizontal size the same (can experiment with adjusting the horizontal axis later)
            #calculate the number of pixels the face is on the horizontal axis
            x_size = end_x - start_x
            #calculate the number of pixels the face is on the vertical axis
            y_size = end_y - start_y
            #get what y_size needs to be
            y_size_with_ratio = x_size * face_ratio
            #how much the y_size needs to be adjusted
            y_size_change = y_size_with_ratio - y_size
            start_y_ = start_y - y_size_change
            end_y_ = end_y + y_size_change
            if start_y_ < 0:
                y_adjust = 0 - start_y_
                end_y_ = min((end_y_ + y_adjust), resized_image.shape[0])
                start_y_ = 0
            elif end_y_ > resized_image.shape[0]:
                y_adjust = end_y_ - resized_image.shape[0]
                start_y_ = max(0, (start_y_ - y_adjust))
                end_y_ = resized_image.shape[0]
            start_x, end_x, start_y_, end_y_ = int(start_x), int(end_x), int(start_y_), int(end_y_)
            face_image = resized_image[start_y_:end_y_, start_x:end_x]
            new_dim_ = (146, 225) #hard coded - -will want to change if I update the _face_ratio
            new_face = cv.resize(face_image, new_dim_, interpolation = cv.INTER_AREA)#change new_dim_ to face_dim
            faces.append(new_face)
    return faces #this will eventually need to become an array