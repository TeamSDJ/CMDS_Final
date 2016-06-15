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

# @REVIEW: Constructing different document string using differnt wiki page content
from analysis_package import *

# NOTE:use the wiki page summary as document for each site terms
wiki_summaries = []
for title in sites_data[['stitle']].values:
    wiki_summaries.append("".join([attribute['summary']
                                   for attribute in wiki_pages[title[0]].values()]))

# NOTE:use the wiki page subtile as document for each site terms
wiki_subtitles = []
for title in sites_data[['stitle']].values:
    wiki_subtitles.append("。".join(merge_lists([list(dict(
        attribute['sections']).keys()) for attribute in wiki_pages[title[0]].values()])))

# NOTE:use the whole wiki page as document for each site terms
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

# NOTE:use the terms that are annotated with links as documents for each site terms
from hanziconv import HanziConv
wiki_links = []
for title in sites_data[['stitle']].values:
    try:
        string = title[0] + HanziConv.toTraditional("。".join(merge_lists(
            [attribute['links'] for attribute in wiki_pages[title[0]].values()])))
        wiki_links.append(string)
    except:
        wiki_links.append(title[0])


# @REVIEW : Data analysis

# REVIEW : Supervised and Unsupervised Learning
# generate class prior vector, class-term matrix, covariance
# matrices of documents and classes,  document vectors and weighted
# document vectors.

# @NOTE:For input, we gives the index of each training data, the label of each
# training data, and the document content of each training data

# NOTE:Supervised Learning :
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

# NOTE:Unsupervised Learning :
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

# XXX:
# 'XXX_table1x' with x from 1~6 meaning unsupvervised method
# 'XXX_tablex' with x from 1~6 meaning supervised method
#  x from 1~6 meaning different strings of document as input :
#  x = 1 : using title
#  x = 2 : using xbody from site database
#  x = 3 : using wiki summary
#  x = 4 : using wiki subtitles
#  x = 5 : using whole wiki page
#  x = 6 : using wiki links


# REVIEW: Visualizing and Evaluating the result for explorative analysis.
# here, we use several ways to visualize or generate result,
# in order to understand if the document vector are reasonable.

from visualize_package import *

# @NOTE: plot the generated matrix :
plot_matrix(class_cov_table1 > (class_cov_table1 +
                                class_cov_table2 + class_cov_table3) / 3)
plot_matrix(class_cov_table2 > (class_cov_table1 +
                                class_cov_table2 + class_cov_table3) / 3)
plot_matrix(class_cov_table3 > (class_cov_table1 +
                                class_cov_table2 + class_cov_table3) / 3)
plot_matrix(class_cov_table4)
plot_matrix(class_cov_table5)

# @NOTE: compare the value of two matrix of same shape,
# and scatter the values in 2D space in order to understand the relationship of two matrix.
compare_table_values(condi_table11, w_doc_vec_table11)
compare_table_values(doc_cov_table5, doc_cov_table4)

# @NOTE: visualize 2D dimension reducted document vectors, in order to see if the embedding of sites are reasonable.
# similar sites should be close in the plot and disimilar sites should be
# far away in the plot

# NOTE:First, we use the conditional probability or importance of terms given document as vector element,
# which is obtain during the unsupvervised NB training phase.

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

# NOTE:Second, we use the term count of each document as vector element.
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

# NOTE:Third, we use the over-classes-summed-conditional-probability-weighted term counts as vector element,
# where each weight on each term is calculated by summing all
# supervised-generated conditional probability of term over all classes.

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

# NOTE:Forth, we use the over-document-summed-conditional-probability-weighted term counts as vector element,
# where each weights on each term is calculated by summing all
# un-supervised-generated conditional probability of term over all documents.

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

# @NOTE:Then, we try to find similar sites for each vector space using K nearset neighborhood,
# in order to check if the vector space can gives reasonable similar sites.
# By the method below, we were able to check the local structure of each vector space.

# NOTE:First, we use the conditional probability or importance of terms given document as vector element,
# which is obtain during the unsupvervised NB training phase.
k_nearest_neighbor(condi_table11.transpose(), 5)
k_nearest_neighbor(condi_table12.transpose(), 5)
k_nearest_neighbor(condi_table13.transpose(), 5)
k_nearest_neighbor(condi_table14.transpose(), 5)
k_nearest_neighbor(condi_table15.transpose(), 5)
k_nearest_neighbor(condi_table16.transpose(), 5)
# NOTE:Second, we use the term count of each document as vector element.
k_nearest_neighbor(doc_vec_table1.transpose(), 5)
k_nearest_neighbor(doc_vec_table2.transpose(), 5)
k_nearest_neighbor(doc_vec_table3.transpose(), 5)
k_nearest_neighbor(doc_vec_table4.transpose(), 5)
k_nearest_neighbor(doc_vec_table5.transpose(), 5)
k_nearest_neighbor(doc_vec_table6.transpose(), 5)
# NOTE:Third, we use the over-classes-summed-conditional-probability-weighted term counts as vector element,
# where each weight on each term is calculated by summing all
# supervised-generated conditional probability of term over all classes.
k_nearest_neighbor(w_doc_vec_table1.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table2.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table3.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table4.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table5.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table6.transpose(), 5)
# NOTE:Forth, we use the over-document-summed-conditional-probability-weighted term counts as vector element,
# where each weights on each term is calculated by summing all
# un-supervised-generated conditional probability of term over all documents.
k_nearest_neighbor(w_doc_vec_table11.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table12.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table13.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table14.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table15.transpose(), 5)
k_nearest_neighbor(w_doc_vec_table16.transpose(), 5)


#TODO: how to compare two ranking list ? 
