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

# NOTE:use the terms that are annotated with links as documents for each
# site terms
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
# training data, and the document content of each training data.
# Also, the cutoff frequecy for filtering the vocaburary words are predefined for each kind of documents.
documents = [sites_data[['stitle']], sites_data[['xbody']],
             pd.DataFrame(wiki_summaries), pd.DataFrame(wiki_subtitles), pd.DataFrame(wiki_whole_page), pd.DataFrame(wiki_links)]
hfcs = [0.,0.05,0.01,0.1,0.01,0.]
lfcs = [0.,0.2,0.2,0.,0.2,0.]
# NOTE:Supervised Learning :
sup_priors = []
sup_condis = []
sup_doc_covs = []
sup_class_covs = []
sup_doc_vecs = []
sup_w_doc_vecs = []
# Here we supvisedly use classes as labels.
for i in range(len(documents)):
    prior_table, condi_table, doc_cov_table, class_cov_table, doc_vec_table, w_doc_vec_table = analysis(
    pd, sites_data[['stitle']], sites_data[['CAT2']], documents[i], hfcs[i],lfcs[i])
    sup_priors.append(prior_table)
    sup_condis.append(condi_table)
    sup_doc_covs.append(doc_cov_table)
    sup_class_covs.append(class_cov_table)
    sup_doc_vecs.append(doc_vec_table)
    sup_w_doc_vecs.append(w_doc_vec_table)
# NOTE:Unsupervised Learning :
unsup_priors = []
unsup_condis = []
unsup_doc_covs = []
unsup_class_covs = []
unsup_doc_vecs = []
unsup_w_doc_vecs = []
# Here we un-supvisedly use data index as labels.
for i in range(len(documents)):
    prior_table, condi_table, doc_cov_table, class_cov_table, doc_vec_table, w_doc_vec_table = analysis(
    pd, sites_data[['stitle']], sites_data[['stitle']], documents[i], hfcs[i],lfcs[i])
    unsup_priors.append(prior_table)
    unsup_condis.append(condi_table)
    unsup_doc_covs.append(doc_cov_table)
    unsup_class_covs.append(class_cov_table)
    unsup_doc_vecs.append(doc_vec_table)
    unsup_w_doc_vecs.append(w_doc_vec_table)

# XXX:
#  index from 1~6 meaning different strings of document as input :
#  index = 1 : using title
#  index = 2 : using xbody from site database
#  index = 3 : using wiki summary
#  index = 4 : using wiki subtitles
#  index = 5 : using whole wiki page
#  index = 6 : using wiki links


# REVIEW: Visualizing and Evaluating the result for explorative analysis.
# here, we use several ways to visualize or generate result,
# in order to understand if the document vector are reasonable.

from visualize_package import *

# @NOTE: plot the generated matrix :
for class_cov_table in sup_class_covs:
    plot_matrix(class_cov_table)
for class_cov_table in unsup_class_covs:
    plot_matrix(class_cov_table)
# @NOTE: compare the value of two matrix of same shape,
# and scatter the values in 2D space in order to understand the
# relationship of two matrix.
compare_table_values(condi_table11, w_doc_vec_table11)
compare_table_values(doc_cov_table5, doc_cov_table4)

# @NOTE: visualize 2D dimension reducted document vectors, in order to see if the embedding of sites are reasonable.
# similar sites should be close in the plot and disimilar sites should be
# far away in the plot

# NOTE:First, we use the conditional probability or importance of terms given document as vector element,
# which is obtain during the unsupvervised NB training phase.
for i in range(len(unsup_condis)):
    site_2d = dimension_reduction(unsup_condis[i].transpose())
    plot_word_embedding(plt, site_2d, num=i, labels=sites_data[['CAT2']])
plt.show()

# NOTE:Second, we use the term count of each document as vector element.
for i in range(len(unsup_doc_vecs)):
    site_2d = dimension_reduction(unsup_doc_vecs[i].transpose())
    plot_word_embedding(plt, site_2d, num=i, labels=sites_data[['CAT2']])
plt.show()

# NOTE:Third, we use the over-classes-summed-conditional-probability-weighted term counts as vector element,
# where each weight on each term is calculated by summing all
# supervised-generated conditional probability of term over all classes.
for i in range(len(sup_w_doc_vecs)):
    site_2d = dimension_reduction(sup_w_doc_vecs[i].transpose())
    plot_word_embedding(plt, site_2d, num=i, labels=sites_data[['CAT2']])
plt.show()

# NOTE:Forth, we use the over-document-summed-conditional-probability-weighted term counts as vector element,
# where each weights on each term is calculated by summing all
# un-supervised-generated conditional probability of term over all documents.
for i in range(len(unsup_w_doc_vecs)):
    site_2d = dimension_reduction(unsup_w_doc_vecs[i].transpose())
    plot_word_embedding(plt, site_2d, num=i, labels=sites_data[['CAT2']])
plt.show()
# NOTE:Fifth, we use the rows of document covariance matrix generated from term count as document vector,
for i in range(len(unsup_doc_covs)):
    site_2d = dimension_reduction(unsup_doc_covs[i].transpose())
    plot_word_embedding(plt, site_2d, num=i, labels=sites_data[['CAT2']])
plt.show()

# @NOTE:Then, we try to find similar sites for each vector space using K nearset neighborhood,
# in order to check if the vector space can gives reasonable similar sites.
# By the method below, we were able to check the local structure of each
# vector space.
k = 5
# NOTE:First, we use the conditional probability or importance of terms given document as vector element,
# which is obtain during the unsupvervised NB training phase.
unsup_condis_neighbors = []
for table in unsup_condis:
    unsup_condis_neighbors.append(k_nearest_neighbor(table.transpose(),k))
# NOTE:Second, we use the term count of each document as vector element.
unsup_doc_vecs_neighbors = []
for table in unsup_doc_vecs:
    unsup_doc_vecs_neighbors.append(k_nearest_neighbor(table.transpose(),k))
# NOTE:Third, we use the over-classes-summed-conditional-probability-weighted term counts as vector element,
# where each weight on each term is calculated by summing all
# supervised-generated conditional probability of term over all classes.
sup_w_doc_vecs_neighbors = []
for table in sup_w_doc_vecs:
    sup_w_doc_vecs_neighbors.append(k_nearest_neighbor(table.transpose(),k))
# NOTE:Forth, we use the over-document-summed-conditional-probability-weighted term counts as vector element,
# where each weights on each term is calculated by summing all
# un-supervised-generated conditional probability of term over all documents.
unsup_w_doc_vecs_neighbors = []
for table in unsup_w_doc_vecs:
    unsup_doc_vecs_neighbors.append(k_nearest_neighbor(table.transpose(),k))
unsup_doc_covs_neighbors = []
for table in unsup_doc_covs:
    unsup_doc_covs_neighbors.append(k_nearest_neighbor(table.transpose(),k))

# TODO: how to compare two ranking list ?
