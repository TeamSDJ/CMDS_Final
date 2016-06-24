# NOTE: crawling url from bing using site name with " 痞客邦 pixnet"
import pickle
import six.moves.cPickle as pickle
# load data
with open("pages.dat", 'rb') as f:
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
        'PtmTrhdWTdbSZTpKw/dfN2dc3TL4DsC1QGOWHsW41yA','"'+site_name+'"'+' 痞客邦 pixnet')
    pages = bing_web.search(limit=5, format='json')
    page_list = []
    for page in pages:
        page_list.append({"url":page.url,"title":page.title,"description":page.description})
    pages_dict[site_name]=page_list



# REVIEW: store the crawled wiki pages as a serialized object in page.dat
import six.moves.cPickle as pickle

# save data
with open("taipei_sites_page.dat", "wb") as f:
    pickle.dump(pages_dict, f, protocol=1)
