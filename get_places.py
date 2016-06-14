import urllib2
import json
import sys

#############################
# Basic Setting
#############################
dis   = "1000" # the radius of nearby places searching (unit: meter)
mode  = "walking"
key   = "AIzaSyAIEF_B0BKsmqSosQ4isIW9d0CM7SrXJos"
token = "EAALoPr8WJz4BACuIz1hSKsDdez4DRztUghLXsEzQKZB92ssnv8kOR1YYdUB2ZCTU1i3DLHRSkWZA3eH0ZBYOnZBPZCGKaBX7H17Kv6wtPQYUEhgjkrVb7LtOXo4m5OodgNRZCKVLXZAnZAAmGbvZCaonfz33ZAmW4wsiKQZD"

#############################
# Input Place Name
#############################
input = "Cafe+Kuroshio"

#############################
# Get Place ID
#############################
url = "https://maps.googleapis.com/maps/api/place/queryautocomplete/json?language=zh-TW&input=" + input + "&key=" + key
response = urllib2.urlopen(url)
jsonResponse = json.loads(response.read())
if jsonResponse["status"] != "OK": sys.exit(0)
id = jsonResponse["predictions"][0]["place_id"]

#############################
# Get coordinates
#############################
url = "https://maps.googleapis.com/maps/api/place/details/json?placeid=" + id + "&key=" + key
response = urllib2.urlopen(url)
jsonResponse = json.loads(response.read())
if jsonResponse["status"] != "OK": sys.exit(0)
lat = str(jsonResponse["result"]["geometry"]["location"]["lat"])
lng = str(jsonResponse["result"]["geometry"]["location"]["lng"])

#############################
# Get nearby places
#############################
url = "https://graph.facebook.com/search?type=place&center=" + lat + "," + lng + "&distance=" + dis + "&access_token=" + token
response = urllib2.urlopen(url)
jsonResponse = json.loads(response.read())
nearPlaces   = jsonResponse["data"]

print nearPlaces

#############################
# Calculate routes
#############################
distList = []
for place in nearPlaces:
  pLat = place["location"]["latitude"]
  pLng = place["location"]["longitude"]
  url  = "https://maps.googleapis.com/maps/api/directions/json?origin=" + str(lat) + "," + str(lng) + "&destination=" + str(pLat) + "," + str(pLng) + "&mode=" + mode + "&key=" + key
  response = urllib2.urlopen(url)
  jsonResponse = json.loads(response.read())
  distList.append(jsonResponse["routes"][0]["legs"][0]['duration']['text'])

print distList