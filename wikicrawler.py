import pandas as pd
import numpy as np


def generate_combination(tokens):
    combs = []
    for i in range(len(tokens)):
        for j in range(i+1):
            combs.append(''.join(tokens[0+j:len(tokens)-i+j]))
    return combs

def merge_lists(lists):
    return [item for sublist in lists for item in sublist]


sites_data = pd.read_excel('sitedata.xlsx', 'Section',
                           index_col='Section_Id', na_values=['NA'])
id_map = pd.read_excel('sitedata.xlsx', 'file',
                       index_col='Section_Id', na_values=['NA'])
img_url = pd.read_excel('sitedata.xlsx', 'img',
                        index_col='file_Id', na_values=['NA'])
audio_url = pd.read_excel('sitedata.xlsx', 'audio',
                          index_col='file_Id', na_values=['NA'])
video_url = pd.read_excel('sitedata.xlsx', 'video',
                          index_col='file_Id', na_values=['NA'])


sites_data[['stitle','CAT2','longitude', 'latitude']]
sites_data[['xbody']]
sites_data.columns.values
site_names = [e[0] for e in sites_data[['stitle']].values.tolist()]
cat_names = [e[0] for e in sites_data[['CAT2']].values.tolist()]


import wikitextparser as wtp
import wikipedia
import jieba
import jieba.analyse
import jieba.posseg as pseg
jieba.set_dictionary('dict.txt.big.txt')
jieba.enable_parallel(4)
jieba.analyse.set_stop_words('stop_words.txt')
jieba.analyse.set_idf_path('idf.txt.big.txt')
jieba.initialize()
wikipedia.set_lang("zh-tw")

stopwords = set([e.decode('utf8').splitlines()[0] for e in open('stop_words.txt','rb').readlines()])

def tokenize(name):
    tokens = []
    for term in jieba.tokenize(name):
        if term[0] in stopwords:
            None
        else:
            tokens.append(term[0])
    return tokens

# totenize each site names


def get_pages_from_wiki(site_name):
    tokens = tokenize(site_name)
    count = 0
    no_page = True
    pages = dict()
    for term in generate_combination(tokens):
        count = count + 1
        try:
            page = wikipedia.page(term)
            if(page.title==term):
                pages[page.title]=page
                print(term)
                print(page.title)
        except:
            None
    return pages


def convert_to_strings(wikipage):
    from hanziconv import HanziConv
    import wikitextparser as wtp
    import pprint
    try:
        summary = HanziConv.toTraditional(wtp.parse(wikipage.content).sections[0].pprint())
    except:
        summary = None
    try:
        sections = [HanziConv.toTraditional(sec.pprint()) for sec in wtp.parse(wikipage.content).sections[1:]]
        try:
            sub_titles = [HanziConv.toTraditional(sec.title[1:-1]) for sec in wtp.parse(wikipage.content).sections[1:]]
        except:
            sub_titles = None
        try:
            section_content = [s[s.find('\n')+1:] for s in sections]
        except:
            section_content = None
    except:
        sections = None

    try:
        sections = list(zip(sub_titles,section_content))
    except:
        sections = None
    try:
        links = wikipage.links
    except:
        links = None
    return {'title':wikipage.title,'summary':summary,'sections':sections,'links':links}

pageget_pages_from_wiki(site_names[0])

site_pages = dict()
for site_name in site_names:
    pages = []
    wiki_pages = get_pages_from_wiki(site_name)
    for key in wiki_pages.keys():
        wiki_pages[key]=convert_to_strings(wiki_pages[key])
    site_pages[site_name]=wiki_pages



import six.moves.cPickle as pickle

# save data
with open("pages.dat", "wb") as f:
    pickle.dump(site_pages, f, protocol=1)

import six.moves.cPickle as pickle

# load data
with open("pages.dat",'rb') as f:
    site_pages_ = pickle.load(f)

site_pages_['新北投溫泉區']
