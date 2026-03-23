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
import photoslop_v1.layers
from .layers import *


def index(request):
    return render(request, "index.html")
def savepic(request):
    result_url = request.session.get('result_image', '')
    if result_url:
        decoded_url = unquote(result_url)
        relative_path = decoded_url.replace('/media/', '', 1)
        image_path = os.path.join(settings.MEDIA_ROOT, relative_path)
        if not os.path.exists(image_path):
            print(f"Файл {image_path} не найден!")

        img = Image.open(image_path)
        img.show()
    return redirect('/result')
def postpic(request):
    if request.method == 'POST':
        image_urls = request.session.get('uploaded_images', [])
        alphas = request.session.get('alphas', [])
        modes = request.session.get('modes', [1] * len(image_urls))
        if request.FILES.getlist('profile_image'):
            images = request.FILES.getlist('profile_image')
            fs = FileSystemStorage()

            for image in images:
                filename = fs.save(image.name, image)
                image_url = fs.url(filename)
                image_urls.append(image_url)
                alphas.append(255)
                modes.append(0)

            request.session['uploaded_images'] = image_urls
            request.session['alphas'] = alphas
            request.session['modes'] = modes
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
    request.session['alphas'] = []
    request.session['modes'] = []
    return redirect('/')


def delete(request, index):
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas', [])


    image_urls.pop(index)
    if alphas and index < len(alphas):
        alphas.pop(index)


    request.session['uploaded_images'] = image_urls
    request.session['alphas'] = alphas

    return redirect('/result')


def up(request, index):
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas', [])

    if index > 0:

        buf_url = image_urls[index]
        image_urls[index] = image_urls[index - 1]
        image_urls[index - 1] = buf_url

        if alphas and index < len(alphas):
            buf_alpha = alphas[index]
            alphas[index] = alphas[index - 1]
            alphas[index - 1] = buf_alpha

    request.session['uploaded_images'] = image_urls
    request.session['alphas'] = alphas

    return redirect('/result')


def down(request, index):
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas', [])

    if index < len(image_urls) - 1:
        buf_url = image_urls[index]
        image_urls[index] = image_urls[index + 1]
        image_urls[index + 1] = buf_url
        if alphas and index < len(alphas):
            buf_alpha = alphas[index]
            alphas[index] = alphas[index + 1]
            alphas[index + 1] = buf_alpha
    request.session['uploaded_images'] = image_urls
    request.session['alphas'] = alphas

    return redirect('/result')


def result(request):
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas')
    alphas = list(map(float, alphas))
    alphas = [x / 255 for x in alphas]
    alphas.reverse()
    modes = request.session.get('modes')
    images = []

    for image_url in reversed(image_urls):
        decoded_url = unquote(image_url)
        relative_path = decoded_url.replace('/media/', '', 1)
        image_path = os.path.join(settings.MEDIA_ROOT, relative_path)
        if not os.path.exists(image_path):
            print(f"Файл {image_path} не найден!")
            continue

        img = Image.open(image_path)
        images.append(img)

    if not images:
        context = {
            'error': 'Нет загруженных изображений',
            'image_urls': image_urls,
            'alphas': alphas
        }
        return render(request, 'result.html', context)

    result_array = images[0]

    for i in range(len(images) - 1):
        current_mode = modes[i] if i < len(modes) else '0'  # режим по умолчанию
        if current_mode == '1':
            result_array = sum_images([result_array, images[i + 1]], [alphas[i], alphas[i + 1]])
        elif current_mode == '2':
            result_array = sub_images([result_array, images[i + 1]], [alphas[i], alphas[i + 1]])
        elif current_mode == '3':
            result_array = max_images([result_array, images[i + 1]], [alphas[i], alphas[i + 1]])
        elif current_mode == '4':
            result_array = geom_images([result_array, images[i + 1]], [alphas[i], alphas[i + 1]])
        elif current_mode == '5':
            result_array = sr_images([result_array, images[i + 1]], [alphas[i], alphas[i + 1]])
        else:
            result_array = layer_images([result_array, images[i + 1]], [alphas[i], alphas[i + 1]])


    img = result_array
    result_filename = 'result.png'
    result_path = os.path.join(settings.MEDIA_ROOT, result_filename)
    img.save(result_path, 'PNG')

    fs = FileSystemStorage()
    image_url = fs.url(result_filename)
    request.session['result_image'] = image_url
    alphas.reverse()
    alphas = list(map(lambda x: x*255, alphas))
    context = {
        'image_urls': image_urls,
        'result_image': image_url,
        'alphas': alphas,
        'modes': modes
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
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas', [255]*len(image_urls))
    alphas[index] = request.GET.get('opacity', 255)
    request.session['alphas'] = alphas

    return redirect('/result')
def change_mode(request, index):
    image_urls = request.session.get('uploaded_images', [])
    modes = request.session.get('modes', [1]*len(image_urls))
    modes[index] = request.GET.get('mode', 2)
    request.session['modes'] = modes
    return redirect('/result')