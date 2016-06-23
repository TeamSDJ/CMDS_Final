# NOTE:crawling geo infomation and category information of sites from google map and facebook graph api
import time
import pickle
import six.moves.cPickle as pickle
# load data
with open("taiwan_sites.dat", 'rb') as f:
    wiki_pages = pickle.load(f)

site_names = list(wiki_pages.keys())

with open("taiwan_sites_geo_infos.dat", 'rb') as f:
    try:
        geo_infos = pickle.load(f)
    except:
        geo_infos = dict()


from get_places import *


for site in site_names[0:10]:
    try:
        if geo_infos[site]!=None:
            print(site,' skipped !')
            continue
    except:
        None
    print(site)
    try:
        place=get_places(site)
        geo_infos[site]=place
        if place==None:
            print('None')
        else:
            print('Get')
    except:
        print('Error')

    #time.sleep(10)

import six.moves.cPickle as pickle
# save data
with open("taiwan_sites_geo_infos.dat", "wb") as f:
    pickle.dump(geo_infos, f, protocol=1)
