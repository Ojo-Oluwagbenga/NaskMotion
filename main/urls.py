from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("motionimagery", views.motionimagery, name="index"),
    path("storyfolds", views.storyfolds, name="index"),
    path("twinview", views.twinview, name="index"),
    path("donate", views.donate, name="index"),
] 