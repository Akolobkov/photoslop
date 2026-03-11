"""
URL configuration for photoslop project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include
from photoslop_v1 import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.index),
    path("postpic/", views.postpic),
    path("pic/", views.showpic),
    #path("sshakalit/", views.sshakalit)
    path("del/<int:index>/", views.delete, name='delete_image_by_index'),
    path("up/<int:index>/", views.up, name='moveup_image_by_index'),
    path("down/<int:index>/", views.down, name='movedown_image_by_index'),
    path("result/", views.result),
    path("vanish/", views.vanish),
    path("nored/<int:index>/", views.red_filter, name = 'remove_red_image_by_index'),
    path("nogreen/<int:index>/", views.green_filter, name = 'remove_green_image_by_index'),
    path("noblue/<int:index>/", views.blue_filter, name = 'remove_blue_image_by_index'),
    path('change-opacity/<int:index>/', views.change_opacity, name='change_opacity'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)