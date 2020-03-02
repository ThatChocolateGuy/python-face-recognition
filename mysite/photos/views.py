import Algorithmia
import time
import os, shutil

import string
import random

import tinify

from . import faceDetection

from shutil import copyfile

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views import View
# from django.urls import reverse

from urllib.parse import urlencode

from .forms import PhotoForm
from .models import Photo

# Get directory for file manipulation
DIR_PATH = os.getcwd()

# Authenticate Algorithmia with API key
apiKey = "simsX9jEbWI8fgSqUz1+vQ+IvGr1"
# Create the Algorithmia client object
client = Algorithmia.client(apiKey)

# Authenticate Tinify with API key
tinify.key = faceDetection.tinify.key


# unused
class BasicUploadView(View):
    def get(self, request):
        photos_list = Photo.objects.all()
        return render(self.request, 'photos/basic_upload/index.html', {
            'photos': photos_list})

    def post(self, request):
        form = PhotoForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            data = {'is_valid': True, 'name': photo.file.name,
                    'url': photo.file.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)


# Only View Being Used
class ProgressBarUploadView(View):
    def get(self, request):
        photos_list = Photo.objects.all()

        # grab upload success status if exists
        upload_status = request.GET.get('upl')

        # renders html from photos/templates
        return render(self.request, 'photos/progress_bar_upload/index.html', {
            'photos': photos_list,
            'uplstatus': upload_status if upload_status == 'success' else ''
        })

    def post(self, request):
        form = PhotoForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            print("\ncurrent directory: " + DIR_PATH)
            local_image_path = DIR_PATH + photo.file.url
            print('local img path: ' + local_image_path)

            # compress and resize image with Tinfy API
            compressed_img = tinify.from_file(local_image_path)
            print('compressed image')
            compressed_img = compressed_img.resize(
                method="thumb",
                width=300,
                height=300
            )
            print('resized image')

            # overwrite photo with compressed image
            compress_path = 'media/' + photo.file.name
            compressed_img.to_file(compress_path)
            print('compressed to file: ' + compress_path)

            data = {'is_valid': True, 'name': photo.file.name.split('/')[1],
                    'url': photo.file.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)


# unused
class DragAndDropUploadView(View):
    def get(self, request):
        photos_list = Photo.objects.all()
        return render(self.request, 'photos/drag_and_drop_upload/index.html', {
            'photos': photos_list})

    def post(self, request):
        form = PhotoForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            data = {'is_valid': True, 'name': photo.file.name,
                    'url': photo.file.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)


# Triggered by UI button
def clear_database(request):
    # clear models
    for photo in Photo.objects.all():
        photo.file.delete()
        photo.delete()

    # purge remaining files
    folder = os.path.join(DIR_PATH, 'media')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return redirect(request.POST.get('train'))


# Triggered by UI button
def train_model(request):
    for photo in Photo.objects.all():
        images = faceDetection.get_images(image=photo)

    print("\nIMAGES:\n\n" + images.__str__())
    success = False

    try:
        faceDetection.train_images(images=images)
        success = True
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

    # build redirect url with query string
    base_url = request.POST.get('train')
    # build query string to pass to redirect url
    query_string =  urlencode({'upl': 'success' if success == True else ''})
    ## url: /photos/train-model/?success=success
    url = '{}?{}'.format(base_url, query_string)
    print(url)

    return redirect(url)


# Handle predictions
def predict(request):
    # Prediction Result (JSON Response)
    prediction = faceDetection.model_predictions(
        image_src=request.POST.get('subject'))

    # Check if valid prediction returned
    if prediction["images"][0]["predictions"]:
        # Subject name & confidence score
        predictionSubject = prediction["images"][0]["predictions"][0]["person"]
        predictionConfidence = prediction["images"][0]["predictions"][0]["confidence"]

        # Prints prediction result
        print('PREDICTION RESULT\n\n' + prediction.__str__() + "\n")
        # Prints image path
        print('OUTPUT IMAGE PATH (CLIENT): ' + prediction["images"][0]["output"])

        # Downloads image file from collection (bb: bounding box)
        bbTempFile = client.file(prediction["images"][0]["output"]).getFile()

        # Check if file exists then copy file to workable local directory
        path = "./media/photos/_bb-image_" + predictionSubject + ".jpg"
        if os.path.exists(path=path):
            # Generate unique 6-digit ID to append to image file name
            chars = string.ascii_uppercase + string.digits
            psId = ''.join(random.choice(chars) for _ in range(6))
            bbTempImg = copyfile(
                bbTempFile.name,
                "./media/photos/_bb-image_" + predictionSubject + psId + ".jpg")
        else:
            bbTempImg = copyfile(bbTempFile.name, path)

        # Final File Name
        bbImageName = bbTempImg.split('.')[1]
        bbImageName += ".jpg"
        print('OUTPUT IMAGE PATH (LOCAL): ' + bbImageName)

        # Sets subject name to pass to UI
        if predictionConfidence < 0.51:
            prediction["result"] = "unsure about subject : confidence low"
            # Sets text-success/text-danger bootstrap attributes
            textResultColour = "text-danger"
        else:
            textResultColour = "text-success"
    # Handles undetected face
    else:
        prediction = {
            'resultCode': 'no_face',
            'result': 'No face detected. Try with another image!'}
        predictionSubject = ''
        predictionConfidence = ''
        bbImageName = request.POST.get('subject')
        textResultColour = "text-danger"

    return render(request, 'home.html', {
        'prediction': prediction,
        'predictionSubject': predictionSubject,
        'predictionConfidence': predictionConfidence,
        'bbPhoto': bbImageName,
        'textResultColour': textResultColour
    })
