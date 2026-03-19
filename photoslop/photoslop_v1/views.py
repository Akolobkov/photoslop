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
        alphas = request.session.get('alphas', [])
        if request.FILES.getlist('profile_image'):
            images = request.FILES.getlist('profile_image')
            fs = FileSystemStorage()

            for image in images:
                filename = fs.save(image.name, image)
                image_url = fs.url(filename)
                image_urls.append(image_url)

            request.session['uploaded_images'] = image_urls
            alphas.append(255)
            request.session['alphas'] = alphas
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
    return redirect('/')


def delete(request, index):
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas', [])

    # Удаляем изображение и соответствующее alpha-значение
    image_urls.pop(index)
    if alphas and index < len(alphas):
        alphas.pop(index)

    # Сохраняем обновленные списки
    request.session['uploaded_images'] = image_urls
    request.session['alphas'] = alphas

    return redirect('/result')


def up(request, index):
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas', [])

    if index > 0:
        # Меняем местами изображения
        buf_url = image_urls[index]
        image_urls[index] = image_urls[index - 1]
        image_urls[index - 1] = buf_url

        # Меняем местами alpha-значения
        if alphas and index < len(alphas):
            buf_alpha = alphas[index]
            alphas[index] = alphas[index - 1]
            alphas[index - 1] = buf_alpha

    # Сохраняем обновленные списки
    request.session['uploaded_images'] = image_urls
    request.session['alphas'] = alphas

    return redirect('/result')


def down(request, index):
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas', [])

    if index < len(image_urls) - 1:
        # Меняем местами изображения
        buf_url = image_urls[index]
        image_urls[index] = image_urls[index + 1]
        image_urls[index + 1] = buf_url

        # Меняем местами alpha-значения
        if alphas and index < len(alphas):
            buf_alpha = alphas[index]
            alphas[index] = alphas[index + 1]
            alphas[index + 1] = buf_alpha

    # Сохраняем обновленные списки
    request.session['uploaded_images'] = image_urls
    request.session['alphas'] = alphas

    return redirect('/result')


def result(request):
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas')
    alphas = list(map(float, alphas))
    alphas = [x / 255 for x in alphas]
    alphas.reverse()

    images = []
    for image_url in reversed(image_urls):
        decoded_url = unquote(image_url)
        relative_path = decoded_url.replace('/media/', '', 1)
        image_path = os.path.join(settings.MEDIA_ROOT, relative_path)

        # Проверяем существование файла
        if not os.path.exists(image_path):
            print(f"Файл {image_path} не найден!")
            continue

        img = Image.open(image_path)
        images.append(img)

    # Конвертируем в RGBA
    img_arrays = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_arrays.append(np.array(img))

    # Определяем максимальные размеры
    shapes = [img.shape for img in img_arrays]
    mshape = [
        max(shape[0] for shape in shapes),
        max(shape[1] for shape in shapes)
    ]

    # Инициализируем фоновые массивы
    alpha_bg = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
    c_bg = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

    for i in range(len(img_arrays)):
        img = img_arrays[i]
        alpha = alphas[i] if i < len(alphas) else 1.0

        h, w = img.shape[:2]

        alpha_src = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
        c_src = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

        alpha_src[:h, :w] = img[:, :, 3] / 255.0
        c_src[:h, :w, :] = img[:, :, :3] / 255.0

        alpha_src = alpha_src * alpha

        alpha_out = alpha_src + alpha_bg * (1 - alpha_src)

        mask = alpha_out > 0
        c_out = np.zeros_like(c_src)

        c_out[mask] = (
                              c_src[mask] * alpha_src[mask, np.newaxis] +
                              c_bg[mask] * alpha_bg[mask, np.newaxis] * (1 - alpha_src[mask, np.newaxis])
                      ) / alpha_out[mask, np.newaxis]

        c_out[~mask] = c_bg[~mask]

        alpha_bg = alpha_out
        c_bg = c_out

    # Создаем итоговое изображение
    result = np.zeros((mshape[0], mshape[1], 4), dtype=np.uint8)
    result[:, :, :3] = (c_bg * 255).astype(np.uint8)
    result[:, :, 3] = (alpha_bg * 255).astype(np.uint8)

    # Сохраняем результат
    img = Image.fromarray(result)
    result_filename = 'result.png'
    result_path = os.path.join(settings.MEDIA_ROOT, result_filename)
    img.save(result_path, 'PNG')

    fs = FileSystemStorage()
    image_url = fs.url(result_filename)
    request.session['result_image'] = image_url
    alphas.reverse()
    context = {
        'image_urls': image_urls,
        'result_image': image_url,
        'alphas': alphas
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
    image_urls = request.session.get('uploaded_images', [])
    alphas = request.session.get('alphas', [255]*len(image_urls))
    alphas[index] = request.GET.get('opacity', 255)
    request.session['alphas'] = alphas

    return redirect('/result')
