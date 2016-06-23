from crawling_package import *
# 臺北市旅遊景點列表
def craw_site_links_and_child_page(page_name):
    import wikipedia
    taiwan_site_page = convert_to_strings(wikipedia.page(page_name))

    subtitle_links = dict()
    for subtitle in taiwan_site_page['sections'].keys():
        for link in taiwan_site_page['links']:
            if taiwan_site_page['sections'][subtitle].count(link)>0:
                if subtitle in subtitle_links:
                    subtitle_links[subtitle].append(link)
                else:
                    subtitle_links[subtitle] = [link]


    all_sites = []
    for subtitle in subtitle_links.keys():
        if(subtitle.count('參見')==0 | subtitle.count('對外連結')==0):
            all_sites.extend(subtitle_links[subtitle])

    child_pages = []
    try:
        for references in subtitle_links['參見']:
            if references.count('列表')>0 and references.count('景')>0:
                child_pages.append(references)
    except:
        print('no referenecs in this pages:')
        child_pages = None
    print('done crawling page',page_name)
    return all_sites,child_pages

parent_sites,child_pages = craw_site_links_and_child_page('臺灣觀光景點列表')


child_sites = dict()
for child in child_pages:
    child_own_site,child_own_child = craw_site_links_and_child_page(child)
    child_sites[child]={"sites":child_own_site,"page_links":child_own_child}


all_sites = parent_sites
len(set(all_sites))
for key in child_sites.keys():
    all_sites.extend(child_sites[key]['sites'])
all_sites=set(all_sites)


# generate stopwords for better tokenization

site_pages_dict = dict()
for site_name in list(all_sites):
    try:
        wiki_page = wikipedia.page(site_name)
        site_pages_dict[site_name]=convert_to_strings(wiki_page)
    except:
        print('no wiki page for :',site_name)


# REVIEW: store the crawled wiki pages as a serialized object in page.dat
import six.moves.cPickle as pickle

# save data
with open("taiwan_sites.dat", "wb") as f:
    pickle.dump(site_pages_dict, f, protocol=1)


# load data
#import pickle
#import six.moves.cPickle as pickle
# load data
#with open("taiwan_sites.dat", 'rb') as f:
#    wiki_pages = pickle.load(f)
