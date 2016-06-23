
def get_places(site_name):
    import json
    import sys
    import urllib
    from urllib.parse import quote


    input = site_name
    input = quote(input)

    key   = "AIzaSyBqJjkdH1kTeEJMfILog4w-Wez9M9URZag"
    token = "EAALoPr8WJz4BACuIz1hSKsDdez4DRztUghLXsEzQKZB92ssnv8kOR1YYdUB2ZCTU1i3DLHRSkWZA3eH0ZBYOnZBPZCGKaBX7H17Kv6wtPQYUEhgjkrVb7LtOXo4m5OodgNRZCKVLXZAnZAAmGbvZCaonfz33ZAmW4wsiKQZD"

    url = "https://maps.googleapis.com/maps/api/place/queryautocomplete/json?input=%s&key=%s&language=zh-TW" % (input, key)
    import urllib.request, json
    response = urllib.request.urlopen(url)
    content = response.read()
    jsongeocode = json.loads(content.decode("utf8"))
    if jsongeocode['status'] != 'OK':
        return None
    try:
        id  = jsongeocode['predictions'][0]['place_id']
        url = "https://maps.googleapis.com/maps/api/place/details/json?placeid=%s&key=%s" % (id, key)
        response = urllib.request.urlopen(url)
        jsongeocode = json.loads(response.read().decode("utf8"))
        if jsongeocode['status'] != 'OK':
            return None
    except KeyError:
        return None

    try:
        lat = str(jsongeocode["result"]["geometry"]["location"]['lat'])
        lng = str(jsongeocode["result"]["geometry"]["location"]['lng'])
        dis = "1000"
        url = 'https://graph.facebook.com/search?type=place&center=' + lat + ',' + lng + '&distance=' + dis + '&access_token=' + token
        response = urllib.request.urlopen(url)
        jsongeocode = json.loads(response.read().decode("utf8"))
        return jsongeocode['data']
    except:
        return None
