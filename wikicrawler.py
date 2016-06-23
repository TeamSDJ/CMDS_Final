

# REVIEW: Input site data from excel site database
import pandas as pd
sites_data = pd.read_excel('sitedata.xlsx', 'Section',
                           index_col='Section_Id', na_values=['NA'])


# transform pandas table values to list
site_names = [e[0] for e in sites_data[['stitle']].values.tolist()]
cat_names = [e[0] for e in sites_data[['CAT2']].values.tolist()]


from crawling_package import *
# generate stopwords for better tokenization
stopwords = load_stop_words()


site_pages = dict()
for site_name in site_names[0:1]:
    pages = []
    wiki_pages = get_pages_from_wiki(site_name,stopwords)
    for key in wiki_pages.keys():
        wiki_pages[key] = convert_to_strings(wiki_pages[key])
    site_pages[site_name] = wiki_pages

wiki_pages

site_names

# REVIEW: store the crawled wiki pages as a serialized object in page.dat
import six.moves.cPickle as pickle

# save data
with open("pages.dat", "wb") as f:
    pickle.dump(site_pages, f, protocol=1)

#import six.moves.cPickle as pickle

# load data
# with open("pages.dat",'rb') as f:
#    site_pages_ = pickle.load(f)

# site_pages_['新北投溫泉區']
