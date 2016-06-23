import pickle
import six.moves.cPickle as pickle
# load data
with open("taiwan_sites_urls.dat", 'rb') as f:
    site_urls = pickle.load(f)


from urllib.parse import quote
from extract_text import *
for site in list(site_urls.keys())[0:1]:
    for page in site_urls[site]:
        try:
            page["text"]=extract_text(page['url'])
        except:
            print(page['url'])
            mark_index = page['url'].rfind('/')
            page["text"]=extract_text(page['url'][:mark_index]+"/"+quote(page['url'][mark_index+1:]))
