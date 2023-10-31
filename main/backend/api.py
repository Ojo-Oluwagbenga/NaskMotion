from django.shortcuts import render
from django.http import HttpResponse
from django.http import FileResponse
import json
import requests

import base64
import glob

import cv2

from ..editsystem.c_interpolator import twinview
from ..editsystem.foldmaker import storyfold
from ..editsystem.interpolator import motionimagery


import time
import os

import traceback
import logging


class nasaapi:
    def getrand(self, response):
        try:
            params = {
                "api_key":"Eq6R9tW5ocVyfcKe0Ah6OuWKowpLEJG4CFaV3v9i",
                "count":100,
            }
            r = requests.get('https://api.nasa.gov/planetary/apod', params)
            start =  int(response.GET.get('start', 0))
            resp = {
                'data':r.json()[start:start+10],
                'type':'apod',
            }
        except Exception as e:
            logging.error(traceback.format_exc())
            resp = {
                'error': 'Broken pipe',
                'type':'apod',
            }
        return HttpResponse(json.dumps(resp))

    def getsearch(self, response):
        try:
            params = {
                "media_type":"image",
                "q": response.GET.get('query', 'jovian')
            }
            r = requests.get('https://images-api.nasa.gov/search', params)
            start =  int(response.GET.get('start', 0))
            resp = r.json()['collection']['items'][start:start+20]

            resp = {
                'data':resp,
                'type':'nasaimage',
            }

        except Exception as e:
            logging.error(traceback.format_exc())
            resp = {
                'error': 'Broken pipe',
                'type':'apod',
            }
        return HttpResponse(json.dumps(resp))

class sysapi:
    def twin_interpolate(self, response):

        if (response.method == "POST"):
            data =  json.loads(json.loads(response.body.decode('utf-8')).get('payload'))

            callresponse = {
                'state':False,
            }


            imgarr = []
            for k,v in data.items():
                imgarr.append(v[1])

            if len(imgarr) != 2:
                callresponse = {
                    'state':False,
                    'error': 'Two(2) images parameters are expected',
                    'url':"error",
                }
                return HttpResponse(json.dumps(callresponse))

            try:

                interpolimage = twinview(imgarr, data.get('dist', 10))
                loggedinuser = 'mainuser'
                path = os.getcwd()+"/main/static/interpolimages/"+loggedinuser

                # create user edit folder if it doesnt exist
                if not os.path.isdir(path):
                    os.mkdir(path)

                #Deletes all files in the user edit
                files = glob.glob(path+'/*')
                for f in files:
                    os.remove(f)

                imgname = str(time.time()) + '.png'
                writeurl = path +"/"+ imgname

                # This is the path after the static/interpolate images
                pathfromstatic = loggedinuser+"/"+ imgname

                cv2.imwrite(writeurl, interpolimage)

                callresponse = {
                    'state':True,
                    'error': '',
                    'imgname':pathfromstatic
                }
            except Exception as e:
                logging.error(traceback.format_exc())
                callresponse = {
                    'state':False,
                    'error': 'We are working on succesfully implementing OpenCV on the live server. You may pull our open code from the github link provided in the footer',
                    'url':"null"
                }

            return HttpResponse(json.dumps(callresponse))
        else:
            return HttpResponse("<div style='position: fixed; height: 100vhb ; width: 100vw; text-align:center; display: flex; justify-content: center; flex-direction: column; font-weight:bold>Page Not accessible<div>")

    def motion_interpolate(self, response):

        if (response.method == "POST"):
            data =  json.loads(json.loads(response.body.decode('utf-8')).get('payload'))

            callresponse = {
                'state':False,
            }


            imgarr = []
            for k,v in data.items():
                imgarr.append(v[1])

            if len(imgarr) < 2:
                callresponse = {
                    'state':False,
                    'error': 'Minimum of two(2) images parameters are expected',
                    'url':"error",
                }
                return HttpResponse(json.dumps(callresponse))

            try:

                interpolimages = motionimagery(imgarr, data.get('dist', 10))
                img = interpolimages[0]
                mask = interpolimages[1]
                loggedinuser = 'mainuser'
                path = os.getcwd()+"/main/static/interpolimages/"+loggedinuser

                # create user edit folder if it doesnt exist
                if not os.path.isdir(path):
                    os.mkdir(path)

                #Deletes all files in the user edit
                files = glob.glob(path+'/*')
                for f in files:
                    os.remove(f)

                imgname = str(time.time()) + '.png'
                imgwriteurl = path +"/img_"+ imgname
                maskwriteurl = path +"/mask_"+ imgname

                # This is the path after the static/interpolate images
                imgpathfromstatic = loggedinuser+"/img_"+ imgname
                maskpathfromstatic = loggedinuser+"/mask_"+ imgname

                cv2.imwrite(imgwriteurl, img)
                cv2.imwrite(maskwriteurl, mask)

                callresponse = {
                    'state':True,
                    'error': '',
                    'imgname':imgpathfromstatic,
                    'maskname':maskpathfromstatic
                }
            except Exception as e:
                logging.error(traceback.format_exc())
                callresponse = {
                    'state':False,
                    'error': 'Something broke',
                    'url':"null",
                }

            return HttpResponse(json.dumps(callresponse))
        else:
            return HttpResponse("<div style='position: fixed; height: 100vhb ; width: 100vw; text-align:center; display: flex; justify-content: center; flex-direction: column; font-weight:bold>Page Not accessible<div>")

    def folds_interpolate(self, response):

        if (response.method == "POST"):
            data =  json.loads(json.loads(response.body.decode('utf-8')).get('payload'))

            callresponse = {
                'state':False,
            }


            imgarr = []
            for k,v in data.items():
                imgarr.append(v)

            # print ("kamoru",imgarr)
            # callresponse = {
            #     'state':False,
            #     'error': 'Exactly five(5) images are expected',
            #     'url':"error",
            # }
            # return HttpResponse(json.dumps(callresponse))
            if len(imgarr) != 5:
                callresponse = {
                    'state':False,
                    'error': 'Exactly five(5) images are expected',
                    'url':"error",
                }
                return HttpResponse(json.dumps(callresponse))

            try:

                interpolimages = storyfold(imgarr, width=data.get('width', 3500), gap=data.get('gap', 50))

                front = interpolimages[0]
                back = interpolimages[1]
                loggedinuser = 'mainuser'
                path = os.getcwd()+"/main/static/interpolimages/"+loggedinuser

                # create user edit folder if it doesnt exist
                if not os.path.isdir(path):
                    os.mkdir(path)

                #Deletes all files in the user edit
                files = glob.glob(path+'/*')
                for f in files:
                    os.remove(f)

                imgname = str(time.time()) + '.png'
                frontwriteurl = path +"/front_"+ imgname
                backwriteurl = path +"/back_"+ imgname

                # This is the path after the static/interpolate images
                frontpathfromstatic = loggedinuser+"/front_"+ imgname
                maskpathfromstatic = loggedinuser+"/back_"+ imgname

                front.save(frontwriteurl)
                back.save(backwriteurl)

                callresponse = {
                    'state':True,
                    'error': '',
                    'frontname':frontpathfromstatic,
                    'backname':maskpathfromstatic
                }
            except Exception as e:
                logging.error(traceback.format_exc())
                callresponse = {
                    'state':False,
                    'error': 'Something broke',
                    'url':"null",
                }

            return HttpResponse(json.dumps(callresponse))
        else:
            return HttpResponse("<div style='position: fixed; height: 100vhb ; width: 100vw; text-align:center; display: flex; justify-content: center; flex-direction: column; font-weight:bold>Page Not accessible<div>")
