from PIL import Image
import cv2
import os
from SOURCE.yolo_files import detect
from SOURCE.gan_files import test
from SOURCE.vgg_finetuned_model import vgg_verify
from helper_fns import gan_utils
import shutil
import glob

MEDIA_ROOT = 'media/documents/'
SIGNATURE_ROOT = 'media/UserSignaturesSquare/'
YOLO_RESULT = 'results/yolov5/'
YOLO_OP = 'crops/DLSignature/'
GAN_IPS = 'results/gan/gan_signdata_kaggle/gan_ips/testB'
GAN_OP = 'results/gan/gan_signdata_kaggle/test_latest/images/'
GAN_OP_RESIZED = 'results/gan/gan_signdata_kaggle/test_latest/images/'

def select_cleaned_image(selection):
    ''' Returns the path of cleaned image corresponding to the document the user selected '''
    return GAN_OP + selection + '_fake.png'

def copy_and_overwrite(from_path, to_path):
    '''
    Copy files from results/yolo_ops/ to results/gan/gan_signdata_kaggle/gan_ips
    CycleGAN model requires ip files to be present in results/gan/gan_signdata_kaggle/gan_ips
    '''
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def signature_verify(selection):
    ''' Performs signature verification and displays the anchor image alongside 
        the detections from all the documents and their corresponding cosine 
        similarity score.
    '''
    anchor_image = SIGNATURE_ROOT + selection + '.png'
    # verify the anchor signature with the detctions on all documents
    feature_set = vgg_verify.verify(anchor_image, GAN_OP_RESIZED)
    for image, score in feature_set:
        print(f"Anchor Image: {anchor_image}")
        print(f"Detected Image: {image}")
        print(f"Cosine Similarity Score: {score}")

def signature_cleaning(selection, yolo_op):
    ''' Performs signature cleaning and displays the cleaned signatures '''
    # copy files from results/yolo_ops/ to results/gan/gan_signdata_kaggle/gan_ips
    copy_and_overwrite(yolo_op, GAN_IPS)
    #test.clean() # performs cleaning

    #cleaned images are selected and displayed
    cleaned_image = select_cleaned_image(selection)
    Image.open(cleaned_image).show()

def signature_detection(selection):
    ''' Performs signature detection and returns the results folder. '''
    # call YOLOv5 detection fn on all images in the document folder.
    detect.detect(MEDIA_ROOT)
    # get the path where last detected results are stored.
    latest_detection = max(glob.glob(os.path.join(YOLO_RESULT, '*/')), key=os.path.getmtime)
    # resize and add top and bottom padding to detected sigantures. 
    # gan model expects ips in that particular format.
    gan_utils.resize_images(os.path.join(latest_detection, YOLO_OP))

    # selects and display the detections of the document which the user selected.
    selection_detection = latest_detection + YOLO_OP + selection + '.jpg'
    return latest_detection + YOLO_OP # return the yolo op folder

# Ejemplo de uso
document_selection = ''
yolo_output = signature_detection(document_selection)
print(yolo_output)