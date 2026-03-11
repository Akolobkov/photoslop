from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

from django.shortcuts import render, redirect
from django.conf import settings
from .forms import UserForm
from PIL import Image
import numpy as np
import os
from urllib.parse import unquote
from django.shortcuts import render
import pickle
import base64

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
                file_path = fs.path(filename)

                img = Image.open(file_path).convert('RGBA')

                data = img.getdata()
                img.putdata([(r, g, b, 255) for r, g, b, a in data])


                base = os.path.splitext(file_path)[0]
                new_path = f"{base}_rgba.png"
                img.save(new_path, 'PNG')

                new_filename = os.path.basename(new_path)
                image_url = fs.url(new_filename)

                os.remove(file_path)

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
    mshape = [max(shape[0] for shape in shapes),
              max(shape[1] for shape in shapes)]
    if len(img_arrays[0].shape) == 3:
        result = np.zeros((mshape[0], mshape[1], img_arrays[0].shape[2]), dtype=np.uint8)
    else:
        result = np.zeros(mshape, dtype=np.uint8)
    for img in reversed(img_arrays):
        h, w = img.shape[:2]
        result[:h, :w] = img
    img = Image.fromarray(result)
    result_filename = 'result.png'
    result_path = os.path.join(settings.MEDIA_ROOT, result_filename)
    img.save(result_path, 'PNG')
    fs = FileSystemStorage()
    image_url = fs.url(result_filename)
    request.session['result_image'] = image_url

    context = {
        'image_urls': image_urls,
        'result_image': image_url
    }
    return render(request, 'result.html', context)


def red_filter(request, index):
    image_urls = request.session.get('uploaded_images', [])
    image_url = image_urls[index]
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    image_path = os.path.join(settings.MEDIA_ROOT, relative_path)

    img = Image.open(image_path)
    img_array = np.array(img)

    # Получаем состояние из сессии
    img_state_pickled = request.session.get(f'{image_url}_pickled')

    if img_state_pickled:
        # Десериализуем numpy массивы
        img_state = pickle.loads(base64.b64decode(img_state_pickled))
    else:
        img_state = [0, 0, 0]

    # Обмениваем красный канал
    buf = img_array[:, :, 0].copy()
    img_array[:, :, 0] = img_state[0]
    img_state[0] = buf if isinstance(buf, int) else buf

    # Сериализуем для сохранения в сессии
    img_state_pickled = base64.b64encode(pickle.dumps(img_state)).decode()
    request.session[f'{image_url}_pickled'] = img_state_pickled

    img = Image.fromarray(img_array)
    img.save(image_path)
    return redirect('/result')
def green_filter(request, index):
    image_urls = request.session.get('uploaded_images', [])
    image_url = image_urls[index]
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    image_path = os.path.join(settings.MEDIA_ROOT, relative_path)

    img = Image.open(image_path)
    img_array = np.array(img)

    img_state_pickled = request.session.get(f'{image_url}_pickled')

    if img_state_pickled:
        img_state = pickle.loads(base64.b64decode(img_state_pickled))
    else:
        img_state = [0, 0, 0]

    buf = img_array[:, :, 1].copy()
    img_array[:, :, 1] = img_state[1]
    img_state[1] = buf if isinstance(buf, int) else buf

    img_state_pickled = base64.b64encode(pickle.dumps(img_state)).decode()
    request.session[f'{image_url}_pickled'] = img_state_pickled

    img = Image.fromarray(img_array)
    img.save(image_path)
    return redirect('/result')
def blue_filter(request, index):
    image_urls = request.session.get('uploaded_images', [])
    image_url = image_urls[index]
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    image_path = os.path.join(settings.MEDIA_ROOT, relative_path)

    img = Image.open(image_path)
    img_array = np.array(img)

    img_state_pickled = request.session.get(f'{image_url}_pickled')

    if img_state_pickled:
        img_state = pickle.loads(base64.b64decode(img_state_pickled))
    else:
        img_state = [0, 0, 0]

    buf = img_array[:, :, 2].copy()
    img_array[:, :, 2] = img_state[2]
    img_state[2] = buf if isinstance(buf, int) else buf

    img_state_pickled = base64.b64encode(pickle.dumps(img_state)).decode()
    request.session[f'{image_url}_pickled'] = img_state_pickled

    img = Image.fromarray(img_array)
    img.save(image_path)
    return redirect('/result')
def change_opacity(request, index):
    opacity = request.GET.get('opacity', 255)
    opacity = int(opacity)
    image_urls = request.session.get('uploaded_images', [])
    opacity_values = request.session.get('opacity_values', {})

    if index < len(image_urls):
        image_url = image_urls[index]
        opacity_values[str(index)] = opacity
        opacity_values[image_url] = opacity

        decoded_url = unquote(image_url)
        relative_path = decoded_url.replace('/media/', '', 1)
        image_path = os.path.join(settings.MEDIA_ROOT, relative_path)

        img = Image.open(image_path).convert('RGBA')
        img_array = np.array(img)
        if img_array.shape[2] == 4:
            img_array[:, :, 3] = opacity
            img = Image.fromarray(img_array, 'RGBA')
            img.save(image_path, 'PNG')

    request.session['opacity_values'] = opacity_values
    return redirect('/result')