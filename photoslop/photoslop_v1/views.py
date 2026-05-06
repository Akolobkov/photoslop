from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from .binarization import *
from .filtration import *
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
from .gradation import *
def opengrad(request, index):
    image_urls = request.session.get('uploaded_images', [])
    image_url = image_urls[index]
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    image_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    interactive = InteractiveInterpolation(image_path=image_path, save_path=image_path)
    interactive.show()
    return redirect('/result')
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


def binopen(request, index):
    image_urls = request.session.get('uploaded_images', [])

    if index >= len(image_urls):
        return redirect('/result')

    image_url = image_urls[index]
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)

    if not os.path.exists(source_path):
        print(f"Файл {source_path} не найден!")
        return redirect('/result')

    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_bincorig.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    img = Image.open(source_path)
    img.save(bin_path, 'PNG')


    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['binorig'] = bin_url
    request.session['binindex'] = index

    context = {
        'bin_img_url': bin_url,
    }
    return render(request, 'binar.html', context)

def bingavr(request):
    image_url = request.session.get('binorig')
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    gavr_img = gavr(source_path)
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_binchanging.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    gavr_img.save(bin_path, 'PNG')

    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['binchanging'] = bin_url
    context = {
        'bin_img_url': bin_url,
    }
    return render(request, 'binar.html', context)
def binotsu(request):
    image_url = request.session.get('binorig')
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    otsu_img = otsu(source_path)
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_binchanging.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    otsu_img.save(bin_path, 'PNG')

    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['binchanging'] = bin_url
    context = {
        'bin_img_url': bin_url,
    }
    return render(request, 'binar.html', context)
def binniblek(request):
    try:
        a = int(request.GET.get('a', 15))
        k = float(request.GET.get('k', -0.2))
    except (ValueError, TypeError):
        a = 15
        k = -0.2
    image_url = request.session.get('binorig')
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    nib_img = niblek(source_path, a=a, k= k)
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_binchanging.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    nib_img.save(bin_path, 'PNG')

    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['binchanging'] = bin_url
    context = {
        'bin_img_url': bin_url,
    }
    return render(request, 'binar.html', context)
def binsauvola(request):
    try:
        a = int(request.GET.get('a', 15))
        k = float(request.GET.get('k', -0.2))
    except (ValueError, TypeError):
        a = 15
        k = 0.2
    image_url = request.session.get('binorig')
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    sau_img = savuola(source_path, a=a, k= k)
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_binchanging.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    sau_img.save(bin_path, 'PNG')
    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['binchanging'] = bin_url
    context = {
        'bin_img_url': bin_url,
    }
    return render(request, 'binar.html', context)
def binwolf(request):
    try:
        a = int(request.GET.get('a', 15))
        k = float(request.GET.get('k', -0.2))
    except (ValueError, TypeError):
        a = 15
        k = 0.2
    image_url = request.session.get('binorig')
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    wolf_img = wolf(source_path, a=a, k= k)
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_binchanging.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    wolf_img.save(bin_path, 'PNG')
    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['binchanging'] = bin_url
    context = {
        'bin_img_url': bin_url,
    }
    return render(request, 'binar.html', context)
def binbr(request):
    try:
        a = int(request.GET.get('a', 15))
        k = float(request.GET.get('k', -0.2))
    except (ValueError, TypeError):
        a = 15
        k = 0.15
    image_url = request.session.get('binorig')
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    br_img = bradleyrot(source_path, a=a, k= k)
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_binchanging.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    br_img.save(bin_path, 'PNG')
    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['binchanging'] = bin_url
    context = {
        'bin_img_url': bin_url,
    }
    return render(request, 'binar.html', context)
def binsave(request):
    image_urls = request.session.get('uploaded_images', [])
    changed_url = request.session.get('binchanging')
    index = request.session.get('binindex', 0)
    image_urls[index] = changed_url
    request.session['image_urls'] = image_urls
    return redirect('/result')
def filopen(request, index):
    image_urls = request.session.get('uploaded_images', [])

    if index >= len(image_urls):
        return redirect('/result')

    image_url = image_urls[index]
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)

    if not os.path.exists(source_path):

        return redirect('/result')

    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_filcorig.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    img = Image.open(source_path)
    img.save(bin_path, 'PNG')


    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['filorig'] = bin_url
    request.session['filindex'] = index

    context = {
        'fil_img_url': bin_url,
    }
    return render(request, 'filter.html', context)
def filgauss(request):
    try:
        sig = int(request.GET.get('sig', 1))
        r = float(request.GET.get('r', 3))
    except (ValueError, TypeError):
        sig = 1
        r = 3
    image_url = request.session.get('filorig')
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)
    gauss_img = linearFromGaussianMatrix(source_path, sig, r)
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_filchanging.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    gauss_img.save(bin_path, 'PNG')

    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['filchanging'] = bin_url
    context = {
        'fil_img_url': bin_url,
    }
    return render(request, 'filter.html', context)
def filmedian(request):
    try:
        h = int(request.GET.get('sig', 1))
        w = float(request.GET.get('r', 3))
    except (ValueError, TypeError):
        h = 3
        w = 3
    image_url = request.session.get('filorig')
    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)

    median_img = median(source_path, [h, w])
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_filchanging.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)

    median_img.save(bin_path, 'PNG')

    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['filchanging'] = bin_url
    context = {
        'fil_img_url': bin_url,
    }
    return render(request, 'filter.html', context)
def parse_matrix_from_text(text):
    rows = text.strip().split('\n')
    matrix = []
    for row in rows:
        values = list(map(float, row.strip().split()))
        matrix.append(values)
    return np.array(matrix)
def filmatrix(request):

    matrix_text = request.POST.get('matrix_values', '')
    matrix = parse_matrix_from_text(matrix_text)
    image_url = request.session.get('filorig')
    if not image_url:
        return render(request, 'filter.html', {'error': 'Нет загруженного изображения'})

    decoded_url = unquote(image_url)
    relative_path = decoded_url.replace('/media/', '', 1)
    source_path = os.path.join(settings.MEDIA_ROOT, relative_path)


    filtered_img = linearFromInputtedMatrix(source_path, matrix)
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    bin_filename = f"{original_name}_matrixfilter.png"
    bin_path = os.path.join(settings.MEDIA_ROOT, bin_filename)
    filtered_img.save(bin_path, 'PNG')
    fs = FileSystemStorage()
    bin_url = fs.url(bin_filename)
    request.session['filchanging'] = bin_url

    context = {
            'fil_img_url': bin_url,
    }
    return render(request, 'filter.html', context)
def filsave(request):
    image_urls = request.session.get('uploaded_images', [])
    changed_url = request.session.get('filchanging')
    index = request.session.get('filindex', 0)
    image_urls[index] = changed_url
    request.session['image_urls'] = image_urls
    return redirect('/result')