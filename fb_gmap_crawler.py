# NOTE:crawling geo infomation and category information of sites from google map and facebook graph api
import time
import pickle
import six.moves.cPickle as pickle
# load data
with open("taiwan_sites.dat", 'rb') as f:
    wiki_pages = pickle.load(f)

site_names = list(wiki_pages.keys())


from get_places import *

geo_infos = dict()
for site in site_names:
    print(site)
    try:
        place=get_places(site)
        geo_infos[site]=place
        print(place)
    except:
        print('Error')

    #time.sleep(10)
import six.moves.cPickle as pickle
# save data
with open("taiwan_sites_geo_infos.dat", "wb") as f:
    pickle.dump(geo_infos, f, protocol=1)
