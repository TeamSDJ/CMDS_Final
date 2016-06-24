import pickle
import six.moves.cPickle as pickle
# load data
with open("taiwan_sites_page.dat", 'rb') as f:
    site_urls = pickle.load(f)


from urllib.parse import quote
from extract_text import *
for site in list(site_urls.keys()):
    for page in site_urls[site]:
        try:
            page["text"]=extract_text(page['url'])
            print(page["text"])
        except:
            mark_index = page['url'].rfind('/')
            page["text"]=extract_text(page['url'][:mark_index]+"/"+quote(page['url'][mark_index+1:]))

import six.moves.cPickle as pickle

# save data
with open("taiwan_sites_page_with_text.dat", "wb") as f:
    pickle.dump(url_dict, f, protocol=1)
