import Algorithmia
import time
import os

import string
import random

from . import faceDetection

from shutil import copyfile

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views import View

from .forms import PhotoForm
from .models import Photo

# Authenticate Algorithmia with  API key
apiKey = "simsX9jEbWI8fgSqUz1+vQ+IvGr1"
# Create the Algorithmia client object
client = Algorithmia.client(apiKey)


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
        return render(self.request, 'photos/progress_bar_upload/index.html', {
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


def clear_database(request):
    for photo in Photo.objects.all():
        photo.file.delete()
        photo.delete()
    return redirect(request.POST.get('train'))


def train_model(request):
    for photo in Photo.objects.all():
        images = faceDetection.get_images(image=photo)

    faceDetection.train_images(images=images)
    return redirect(request.POST.get('train'))


# Create new function to handle model predictions
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
        print(prediction.__str__() + "\n")
        # Prints image path
        print(prediction["images"][0]["output"])
        print("\n")

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
        print(bbImageName)
        print("\n")

        if predictionConfidence < 0.75:
            prediction["result"] = "unrecognized subject"
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
        'maxWidth': '900px',
        'textResultColour': textResultColour
    })
