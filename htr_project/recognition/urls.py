from django.urls import path
from . import views

app_name = 'recognition'

urlpatterns = [
    path('', views.index, name='index'),
]
