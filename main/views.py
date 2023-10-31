from django.shortcuts import render
from django.http import HttpResponse

from django.conf import settings
from urllib.parse import urlparse
from django.urls import reverse
import requests
import json
from django.shortcuts import redirect

# from .editsystem.interpolator import *
# from .editsystem.c_interpolator import *

# from .editsystem.foldmaker import *

    
# Create your views here.

def index(response):
    return render(response, "index.html", {})


def twinview(response):
    return render(response, "twinview.html", {})

def storyfolds(response):
    return render(response, "storyfolds.html", {})

def motionimagery(response):
    return render(response, "motionimagery.html", {})

def donate(response):
    return render(response, "donate.html", {})