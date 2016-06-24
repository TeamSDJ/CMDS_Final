# @REVIEW : Read in data

# REVIEW:Read in the crawed wiki pages
import pickle
import six.moves.cPickle as pickle
# load data
with open("pages.dat", 'rb') as f:
    wiki_pages = pickle.load(f)

# REVIEW:Read in the site names and site data
import pandas as pd
sites_data = pd.read_excel('sitedata.xlsx', 'Section',
                           index_col='Section_Id', na_values=['NA'])


wiki_links = []
for title in sites_data[['stitle']].values:
    try:
        string = HanziConv.toTraditional("。".join(merge_lists(
            [attribute['links'] for attribute in wiki_pages[title[0]].values()])))
        wiki_links.append(string)
    except:
        wiki_links.append("。")

prior_table, condi_table, doc_cov_table, class_cov_table, doc_vec_table, w_doc_vec_table = analysis(
    pd, sites_data[['stitle']], sites_data[['stitle']], pd.DataFrame(wiki_links), 0.,0.)




# NOTE: the best space :
import numpy as np
np.shape(k_nearest_neighbor_with_distances(doc_cov_table.transpose(), 318)[1])
np.shape(k_nearest_neighbor_with_distances(doc_cov_table.transpose(), 318)[0])

# NOTE: the second best space :
ct_wiki_summary = condi_table.transpose().sort_index()

dc_wiki_link = doc_cov_table.transpose().sort_index()
pd.concat([ct_wiki_summary, dc_wiki_link], axis=1)
k = 10

ranking_lists,score_matrix = k_nearest_neighbor_with_distances(pd.concat([ct_wiki_summary, dc_wiki_link], axis=1), k)



input = '中正紀念堂'
index = list(ranking_lists.index).index(input)
np.array(ranking_lists)[index,:]
score_matrix[index,:]
