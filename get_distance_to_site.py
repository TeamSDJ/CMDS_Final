
def get_distance_to_site(pLat,pLng,lat=121.53,lng=25.03):

    dis   = "1000" # the radius of nearby places searching (unit: meter)
    mode  = "walking"
    key   = "AIzaSyAIEF_B0BKsmqSosQ4isIW9d0CM7SrXJos"
    token = "EAALoPr8WJz4BACuIz1hSKsDdez4DRztUghLXsEzQKZB92ssnv8kOR1YYdUB2ZCTU1i3DLHRSkWZA3eH0ZBYOnZBPZCGKaBX7H17Kv6wtPQYUEhgjkrVb7LtOXo4m5OodgNRZCKVLXZAnZAAmGbvZCaonfz33ZAmW4wsiKQZD"
    pLat = float("{0:.2f}".format(pLat))
    pLng = float("{0:.2f}".format(pLng))
    url  = "https://maps.googleapis.com/maps/api/directions/json?origin=" + str(lat) + "," + str(lng) + "&destination=" + str(pLat) + "," + str(pLng) + "&mode=" + mode + "&key=" + key

    import urllib.request, json
    response = urllib.request.urlopen(url)
    content = response.read()
    jsongeocode = json.loads(content.decode("utf8"))

    return jsongeocode["routes"][0]["legs"][0]['distance']['text']
