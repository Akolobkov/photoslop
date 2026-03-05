from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

from django.shortcuts import render, redirect
from django.conf import settings
from .forms import UserForm
def index(request):
    return render(request, "index.html")


def postpic(request):
    if request.method == 'POST':
        image_urls = request.session.get('uploaded_images', [])

        if request.FILES.getlist('profile_image'):
            images = request.FILES.getlist('profile_image')
            fs = FileSystemStorage()

            for image in images:
                filename = fs.save(image.name, image)
                image_url = fs.url(filename)
                image_urls.append(image_url)

            request.session['uploaded_images'] = image_urls
        return redirect('/result')

    return redirect('/')


def showpic(request):
    image_urls = request.session.get('uploaded_images', [])

    return render(request, 'result.html', {
        'image_urls': image_urls,
        'media_url': settings.MEDIA_URL,
        'debug': settings.DEBUG
    })

# def sshakalit(request):
#     from PIL import Image
#     import os
#     from django.conf import settings
#     from urllib.parse import unquote
#     image_url = request.session.get('uploaded_image_url')
#
#     if not image_url:
#         return redirect('/pic')
#     decoded_url = unquote(image_url)
#     relative_path = decoded_url.replace('/media/', '', 1)
#     image_path = os.path.join(settings.MEDIA_ROOT, relative_path)
#
#
#     try:
#         img = Image.open(image_path)
#         img = img.resize((800, 600))
#         img = img.rotate(90)
#         img = img.crop((100, 100, 400, 400))
#
#         img.save(image_path)
#
#     except Exception as e:
#         return HttpResponse(400)
#
#     return redirect('/pic')
def vanish(request):
    request.session['uploaded_images'] = []
    return redirect('/')
def delete(request, index):
    image_urls = request.session.get('uploaded_images', [])
    image_urls.pop(index)
    request.session['uploaded_images'] = image_urls
    return redirect('/result')
def up(request, index):
    image_urls = request.session.get('uploaded_images', [])
    buf = image_urls[index]
    if index>0:
        image_urls[index] = image_urls[index-1]
        image_urls[index-1] = buf
    request.session['uploaded_images'] = image_urls
    return redirect('/result')
def down(request, index):
    image_urls = request.session.get('uploaded_images', [])
    buf = image_urls[index]
    if index<len(image_urls)-1:
        image_urls[index] = image_urls[index+1]
        image_urls[index+1] = buf
    request.session['uploaded_images'] = image_urls
    return redirect('/result')
def result(request):
    from PIL import Image
    import numpy as np
    import os
    from urllib.parse import unquote
    from django.shortcuts import render
    image_urls = request.session.get('uploaded_images', [])
    images = []
    for image_url in image_urls:
        decoded_url = unquote(image_url)
        relative_path = decoded_url.replace('/media/', '', 1)
        image_path = os.path.join(settings.MEDIA_ROOT, relative_path)
        img = Image.open(image_path)
        images.append(img)
    img_arrays = list(map(np.array, images))
    shapes = []
    for i in img_arrays:
        shapes.append(i.shape)
    mshape = max(shapes)
    if len(img_arrays[0].shape) == 3:
        result = np.zeros((mshape[0], mshape[1], img_arrays[0].shape[2]), dtype=np.uint8)
    else:
        result = np.zeros(mshape, dtype=np.uint8)
    for img in reversed(img_arrays):
        h, w = img.shape[:2]
        result[:h, :w] = img
    img = Image.fromarray(result)
    img.show()
    result_filename = 'result.png'
    result_path = os.path.join(settings.MEDIA_ROOT, result_filename)
    img.save(result_path)
    fs = FileSystemStorage()
    image_url = fs.url(result_filename)
    request.session['result_image'] = image_url

    context = {
        'image_urls': image_urls,
        'result_image': image_url
    }
    return render(request, 'result.html', context)
