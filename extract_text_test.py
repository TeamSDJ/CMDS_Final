import pickle
import six.moves.cPickle as pickle
# load data
with open("pages.dat", 'rb') as f:
    wiki_pages = pickle.load(f)

site_names = list(wiki_pages.keys())
wiki_pages
from py_bing_search import PyBingWebSearch
# hkxX68ZPyntLT6qdbLnlbk6NqXCv5fQadYm0BGSUER8
bing_web = PyBingWebSearch(
    'hkxX68ZPyntLT6qdbLnlbk6NqXCv5fQadYm0BGSUER8', '中正紀念堂 痞客邦 pixnet')

pages = bing_web.search(limit=50, format='json')

for page in pages:
    print(page.title)
    print(page.url)

from extract_text import *
# 痞客邦
# xuite
extract_text("http://imissyousomuch0.pixnet.net/blog/post/42764705")
