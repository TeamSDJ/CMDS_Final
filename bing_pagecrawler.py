import pickle
import six.moves.cPickle as pickle
# load data
with open("taipei_sites_page.dat", 'rb') as f:
    site_urls = pickle.load(f)

from urllib.parse import quote
from extract_text import *
for site in list(site_urls.keys()):
    print(site)
    for page in site_urls[site]:
        try:
            page["text"]=extract_text(page['url'])
            print(page["text"])
        except:
            mark_index = page['url'].rfind('/')
            page["text"]=extract_text(page['url'][:mark_index]+"/"+quote(page['url'][mark_index+1:]))
            print("with chinese url:",page["text"])

import six.moves.cPickle as pickle
# save data
with open("taipei_sites_page_with_text.dat", "wb") as f:
    pickle.dump(site_urls, f, protocol=1)
