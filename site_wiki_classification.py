# REVIEW : read in data

import pickle
import six.moves.cPickle as pickle
# load data
with open("pages.dat", 'rb') as f:
    wiki_pages = pickle.load(f)


import pandas as pd

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


# REVIEW: constructing wiki pages string for each documents
from classification_package import *

wiki_summaries = []
for title in sites_data[['stitle']].values:
    wiki_summaries.append("".join([attribute['summary']
                                   for attribute in wiki_pages[title[0]].values()]))

wiki_subtitles = []
for title in sites_data[['stitle']].values:
    wiki_subtitles.append("。".join(merge_lists([list(dict(
        attribute['sections']).keys()) for attribute in wiki_pages[title[0]].values()])))


wiki_whole_page = []
# each document include its title, summary of each title terms, sub-titles
# of each title terms, sub-title content of each title terms
for title in sites_data[['stitle']].values:
    wiki_whole_page.append(
        "。".join([title[0]] + [attribute['summary'] for attribute in wiki_pages[title[0]].values()] + merge_lists(
            [list(dict(attribute['sections']).keys()) for attribute in wiki_pages[title[0]].values()] +
            [list(dict(attribute['sections']).values()) for attribute in wiki_pages[title[0]].values()]))
    )

from hanziconv import HanziConv
wiki_links = []
for title in sites_data[['stitle']].values:
    try:
        string = title[0] + HanziConv.toTraditional("。".join(merge_lists(
            [attribute['links'] for attribute in wiki_pages[title[0]].values()])))
        wiki_links.append(string)
    except:
        wiki_links.append(title[0])


# REVIEW : data analysis

import pandas as pd
import numpy as np

# supervised learning
# title_data,cat_data,text_data=sites_data[['stitle']],sites_data[['CAT2']],sites_data[['stitle']]
#hfc = 0.
#lfc = 0.
prior_table1, condi_table1, doc_cov_table1, class_cov_table1, doc_vec_table1, w_doc_vec_table1 = analysis(
    pd, sites_data[['stitle']], sites_data[['CAT2']], sites_data[['stitle']], 0., 0.)
prior_table2, condi_table2, doc_cov_table2, class_cov_table2, doc_vec_table2, w_doc_vec_table2 = analysis(
    pd, sites_data[['stitle']], sites_data[['CAT2']], sites_data[['xbody']], 0.05, 0.2)
prior_table3, condi_table3, doc_cov_table3, class_cov_table3, doc_vec_table3, w_doc_vec_table3 = analysis(
    pd, sites_data[['stitle']], sites_data[['CAT2']], pd.DataFrame(wiki_summaries), 0.01, 0.2)
prior_table4, condi_table4, doc_cov_table4, class_cov_table4, doc_vec_table4, w_doc_vec_table4 = analysis(
    pd, sites_data[['stitle']], sites_data[['CAT2']], pd.DataFrame(wiki_subtitles), 0.1, 0.0)
prior_table5, condi_table5, doc_cov_table5, class_cov_table5, doc_vec_table5, w_doc_vec_table5 = analysis(
    pd, sites_data[['stitle']], sites_data[['CAT2']], pd.DataFrame(wiki_whole_page), 0.01, 0.2)
prior_table6, condi_table6, doc_cov_table6, class_cov_table6, doc_vec_table6, w_doc_vec_table6 = analysis(
    pd, sites_data[['stitle']], sites_data[['CAT2']], pd.DataFrame(wiki_links), 0.0, 0.)

# unsupervised learning
prior_table11, condi_table11, doc_cov_table11, class_cov_table11, doc_vec_table11, w_doc_vec_table11 = analysis(
    pd, sites_data[['stitle']], sites_data[['stitle']], sites_data[['stitle']], 0., 0.)
prior_table12, condi_table12, doc_cov_table12, class_cov_table12, doc_vec_table12, w_doc_vec_table12 = analysis(
    pd, sites_data[['stitle']], sites_data[['stitle']], sites_data[['xbody']], 0.05, 0.2)
prior_table13, condi_table13, doc_cov_table13, class_cov_table13, doc_vec_table13, w_doc_vec_table13 = analysis(
    pd, sites_data[['stitle']], sites_data[['stitle']], pd.DataFrame(wiki_summaries), 0.01, 0.2)
prior_table14, condi_table14, doc_cov_table14, class_cov_table14, doc_vec_table14, w_doc_vec_table14 = analysis(
    pd, sites_data[['stitle']], sites_data[['stitle']], pd.DataFrame(wiki_subtitles), 0.05, 0.3)
prior_table15, condi_table15, doc_cov_table15, class_cov_table15, doc_vec_table15, w_doc_vec_table15 = analysis(
    pd, sites_data[['stitle']], sites_data[['stitle']], pd.DataFrame(wiki_whole_page), 0.05, 0.5)
prior_table16, condi_table16, doc_cov_table16, class_cov_table16, doc_vec_table16, w_doc_vec_table16 = analysis(
    pd, sites_data[['stitle']], sites_data[['stitle']], pd.DataFrame(wiki_links), 0.00, 0.0)


def plot_word_embedding(plt, table, labels=None, title='', num=1):
    plt.figure(num)
    vectors = np.matrix(table).tolist()
    words = list(table.index)

    import matplotlib
    if(type(labels) == type(None)):
        None
        colors = None
    else:
        label_set = list(set(list(labels.values.transpose().tolist())[0]))

        def get_spaced_colors(n):
            max_value = 16581375  # 255**3
            interval = int(max_value / n)
            colors = [hex(I)[2:].zfill(6)
                      for I in range(0, max_value, interval)]

            return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
        colors = get_spaced_colors(len(label_set))

    for i in range(len(words)):
        point = vectors[i]
        word = words[i]
        # plot points
        plt.scatter(point[0], point[1])
        # plot word annotations
        if(type(labels) == type(None)):

            plt.annotate(
                word,
                xy=(point[0], point[1]),
                size="x-small"
            )
        else:
            label_index = label_set.index(
                list(labels.values.transpose().tolist())[0][i])
            plt.annotate(
                word,
                xy=(point[0], point[1]),
                color='#' +
                "".join(list(map(lambda x: format(x, '#04x')
                                 [2:], colors[label_index]))).upper(),
                size="x-small"
            )

    plt.tight_layout()
    plt.title(title)


def dimension_reduction(table):
    from sklearn.manifold import TSNE
    import pandas as pd
    tsne = TSNE(n_components=int(2), perplexity=30.0, early_exaggeration=10.0,
                learning_rate=1000.0,  n_iter=3000, metric='euclidean', init='pca')
    result = tsne.fit_transform(np.matrix(table))
    return pd.DataFrame(result, index=table.index)


# plot unsupervised document vector embedding
import matplotlib.pylab as plt
site_2d = dimension_reduction(condi_table11.transpose())
plot_word_embedding(plt, site_2d, num=1, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(condi_table12.transpose())
plot_word_embedding(plt, site_2d, num=2, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(condi_table13.transpose())
plot_word_embedding(plt, site_2d, num=3, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(condi_table14.transpose())
plot_word_embedding(plt, site_2d, num=4, labels=sites_data[['CAT2']])
site_2d = dimension_reduction(condi_table15.transpose())
plot_word_embedding(plt, site_2d, num=5, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(condi_table16.transpose())
plot_word_embedding(plt, site_2d, num=6, labels=sites_data[['CAT2']])
plt.show()

# plot unsupervised document vector embedding using raw count

import matplotlib.pylab as plt
site_2d = dimension_reduction(doc_vec_table1.transpose())
plot_word_embedding(plt, site_2d, num=1, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(doc_vec_table2.transpose())
plot_word_embedding(plt, site_2d, num=2, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(doc_vec_table3.transpose())
plot_word_embedding(plt, site_2d, num=3, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(doc_vec_table4.transpose())
plot_word_embedding(plt, site_2d, num=4, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(doc_vec_table5.transpose())
plot_word_embedding(plt, site_2d, num=5, labels=sites_data[['CAT2']])


site_2d = dimension_reduction(doc_vec_table6.transpose())
plot_word_embedding(plt, site_2d, num=6, labels=sites_data[['CAT2']])
plt.show()

# plot supervised weighted doc vec embedding

import matplotlib.pylab as plt
site_2d = dimension_reduction(w_doc_vec_table1.transpose())
plot_word_embedding(plt, site_2d, num=1, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(w_doc_vec_table2.transpose())
plot_word_embedding(plt, site_2d, num=2, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(w_doc_vec_table3.transpose())
plot_word_embedding(plt, site_2d, num=3, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(w_doc_vec_table4.transpose())
plot_word_embedding(plt, site_2d, num=4, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(w_doc_vec_table5.transpose())
plot_word_embedding(plt, site_2d, num=5, labels=sites_data[['CAT2']])


site_2d = dimension_reduction(w_doc_vec_table6.transpose())
plot_word_embedding(plt, site_2d, num=6, labels=sites_data[['CAT2']])
plt.show()

# plot unsupervised weighted doc vec embedding

import matplotlib.pylab as plt
site_2d = dimension_reduction(w_doc_vec_table11.transpose())
plot_word_embedding(plt, site_2d, num=1, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(w_doc_vec_table12.transpose())
plot_word_embedding(plt, site_2d, num=2, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(w_doc_vec_table13.transpose())
plot_word_embedding(plt, site_2d, num=3, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(w_doc_vec_table14.transpose())
plot_word_embedding(plt, site_2d, num=4, labels=sites_data[['CAT2']])

site_2d = dimension_reduction(w_doc_vec_table15.transpose())
plot_word_embedding(plt, site_2d, num=5, labels=sites_data[['CAT2']])


site_2d = dimension_reduction(w_doc_vec_table16.transpose())
plot_word_embedding(plt, site_2d, num=6, labels=sites_data[['CAT2']])
plt.show()


def k_nearest_neighbor(table,k):
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(table)
    distances, indices = nbrs.kneighbors(table)
    indices
    neighbor_list = []
    for i in range(len(sites_data)):
        neighbor = []
        for index in indices[i]:
            neighbor.append(sites_data['stitle'][index])
        neighbor_list.append(neighbor)

    return pd.DataFrame(neighbor_list,index=table.index)

k_nearest_neighbor(w_doc_vec_table5.transpose(),5)


compare_table_values(condi_table11,w_doc_vec_table11)
compare_table_values(doc_cov_table5, doc_cov_table4)

# do k nearest neighbor query to test the good or bad of embedding !
import six.moves.cPickle as cPickle
import nearpy
w_doc_vec_
# an embedding consider all info



import numpy as np


# REVIEW : do ploting and comparison
import matplotlib.pylab as plt

plot_matrix(class_cov_table1 > (class_cov_table1 +
                                class_cov_table2 + class_cov_table3) / 3)

plot_matrix(class_cov_table2 > (class_cov_table1 +
                                class_cov_table2 + class_cov_table3) / 3)

plot_matrix(class_cov_table3 > (class_cov_table1 +
                                class_cov_table2 + class_cov_table3) / 3)

plot_matrix(class_cov_table4)

plot_matrix(class_cov_table5)
