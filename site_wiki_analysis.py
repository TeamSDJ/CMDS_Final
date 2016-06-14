# REVIEW : read in data

import pickle
import six.moves.cPickle as pickle
# load data
with open("pages.dat", 'rb') as f:
    wiki_pages = pickle.load(f)


import pandas as pd

sites_data = pd.read_excel('sitedata.xlsx', 'Section',
                           index_col='Section_Id', na_values=['NA'])

# REVIEW: constructing wiki pages as document string differently
from analysis_package import *


# use the wiki page summary as document for each site terms
wiki_summaries = []
for title in sites_data[['stitle']].values:
    wiki_summaries.append("".join([attribute['summary']
                                   for attribute in wiki_pages[title[0]].values()]))

# use the wiki page subtile as document for each site terms
wiki_subtitles = []
for title in sites_data[['stitle']].values:
    wiki_subtitles.append("。".join(merge_lists([list(dict(
        attribute['sections']).keys()) for attribute in wiki_pages[title[0]].values()])))

# use the whole wiki page as document for each site terms
# including the title, the sub-titles, the summary content and the
# contents of each sub-titles.
wiki_whole_page = []
for title in sites_data[['stitle']].values:
    wiki_whole_page.append(
        "。".join([title[0]] + [attribute['summary'] for attribute in wiki_pages[title[0]].values()] +  # title + summary
                 merge_lists(
            # + subtitles
            [list(dict(attribute['sections']).keys()) for attribute in wiki_pages[title[0]].values()] +
            [list(dict(attribute['sections']).values()) for attribute in wiki_pages[title[0]].values()]))  # + the contents of each sub-titles
    )

# use the terms that are annotated with links as documents for each site terms
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
# supervised learning
# title_data,cat_data,text_data=sites_data[['stitle']],sites_data[['CAT2']],sites_data[['stitle']]
#hfc = 0.
#lfc = 0.

# Below, we generate class prior vector, class-term matrix, covariance
# matrices of documents and classes,  document vectors and weighted
# document vectors.

# In input, we gives the index of each training data, the label of each training data, and the document content of each training data

# Here we supvisedly use classes as labels.
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


# Here we un-supvisedly use data index as labels.
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


from visualize_package import *

# plot unsupervised document vector embedding
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





k_nearest_neighbor(w_doc_vec_table14.transpose(), 5)==k_nearest_neighbor(w_doc_vec_table4.transpose(), 5)

compare_table_values(condi_table11, w_doc_vec_table11)
compare_table_values(doc_cov_table5, doc_cov_table4)



plot_matrix(class_cov_table1 > (class_cov_table1 +
                                class_cov_table2 + class_cov_table3) / 3)

plot_matrix(class_cov_table2 > (class_cov_table1 +
                                class_cov_table2 + class_cov_table3) / 3)

plot_matrix(class_cov_table3 > (class_cov_table1 +
                                class_cov_table2 + class_cov_table3) / 3)

plot_matrix(class_cov_table4)

plot_matrix(class_cov_table5)
