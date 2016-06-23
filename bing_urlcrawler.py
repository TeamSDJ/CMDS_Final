# NOTE: crawling url from bing using site name with " 痞客邦 pixnet" or "隨意窩 Xuite日誌"
import pickle
import six.moves.cPickle as pickle
# load data
with open("taiwan_sites.dat", 'rb') as f:
    wiki_pages = pickle.load(f)

site_names = list(wiki_pages.keys())
len(site_names)

pages_dict = dict()
from py_bing_search import PyBingWebSearch
count = 0
for site_name in site_names:
    count = count + 1
    print(count,site_name)
    # hkxX68ZPyntLT6qdbLnlbk6NqXCv5fQadYm0BGSUER8
    bing_web = PyBingWebSearch(
        'hkxX68ZPyntLT6qdbLnlbk6NqXCv5fQadYm0BGSUER8','"'+site_name+'"'+' 痞客邦 pixnet')
    pages = bing_web.search(limit=50, format='json')
    pages_dict[site_name]=pages

url_dict = dict()
for key in pages_dict.keys():
    pages = []
    for page in pages_dict[key]:
        pages.append({"url":page.url,"title":page.title,"description":page.description})
    url_dict[key]=pages

# REVIEW: store the crawled wiki pages as a serialized object in page.dat
import six.moves.cPickle as pickle

# save data
with open("taiwan_sites_page1.dat", "wb") as f:
    pickle.dump(url_dict, f, protocol=1)


pages_dict = dict()
from py_bing_search import PyBingWebSearch
count = 0
for site_name in site_names:
    count = count + 1
    print(count,site_name)
    # hkxX68ZPyntLT6qdbLnlbk6NqXCv5fQadYm0BGSUER8
    bing_web = PyBingWebSearch(
        'hkxX68ZPyntLT6qdbLnlbk6NqXCv5fQadYm0BGSUER8','"'+site_name+'"'+' 隨意窩 Xuite日誌')
    pages = bing_web.search(limit=50, format='json')
    pages_dict[site_name]=pages

url_dict = dict()
for key in pages_dict.keys():
    pages = []
    for page in pages_dict[key]:
        pages.append({"url":page.url,"title":page.title,"description":page.description})
    url_dict[key]=pages

# REVIEW: store the crawled wiki pages as a serialized object in page.dat
import six.moves.cPickle as pickle

# save data
with open("taiwan_sites_page1.dat", "wb") as f:
    pickle.dump(url_dict, f, protocol=1)
