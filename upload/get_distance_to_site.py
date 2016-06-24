
def get_distance_to_site(pLat,pLng,lat=121.53,lng=25.03):

    dis   = "1000" # the radius of nearby places searching (unit: meter)
    mode  = "walking"
    key   = "XXX"
    token = "XXX"
    pLat = float("{0:.2f}".format(pLat))
    pLng = float("{0:.2f}".format(pLng))
    url  = "https://maps.googleapis.com/maps/api/directions/json?origin=" + str(lat) + "," + str(lng) + "&destination=" + str(pLat) + "," + str(pLng) + "&mode=" + mode + "&key=" + key

    import urllib.request, json
    response = urllib.request.urlopen(url)
    content = response.read()
    jsongeocode = json.loads(content.decode("utf8"))

    return jsongeocode["routes"][0]["legs"][0]['distance']['text']
