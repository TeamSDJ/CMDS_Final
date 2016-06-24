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
from hanziconv import HanziConv


# NOTE:use the wiki page summary as document for each site terms
wiki_summaries = []
for title in sites_data[['stitle']].values:
    wiki_summaries.append(HanziConv.toTraditional("".join([attribute['summary']
                                   for attribute in wiki_pages[title[0]].values()])))

# NOTE:use the wiki page subtile as document for each site terms
wiki_subtitles = []
for title in sites_data[['stitle']].values:
    wiki_subtitles.append(HanziConv.toTraditional("。".join(merge_lists([list(dict(
        attribute['sections']).keys()) for attribute in wiki_pages[title[0]].values()]))))

# NOTE:use the whole wiki page as document for each site terms
# including the title, the sub-titles, the summary content and the
# contents of each sub-titles.
wiki_segs = []
for title in sites_data[['stitle']].values:
    wiki_segs.append(
        HanziConv.toTraditional("。".join(merge_lists(
            # + subtitles
            [list(dict(attribute['sections']).values()) for attribute in wiki_pages[title[0]].values()]))  # + the contents of each sub-titles
        )
    )

# NOTE:use the whole wiki page as document for each site terms
# including the title, the sub-titles, the summary content and the
# contents of each sub-titles.
wiki_whole_page = []
for title in sites_data[['stitle']].values:
    wiki_whole_page.append(
        HanziConv.toTraditional("。".join([attribute['summary'] for attribute in wiki_pages[title[0]].values()] +  # title + summary
                 merge_lists(
            # + subtitles
            [list(dict(attribute['sections']).keys()) for attribute in wiki_pages[title[0]].values()] +
            [list(dict(attribute['sections']).values()) for attribute in wiki_pages[title[0]].values()]))  # + the contents of each sub-titles
        )
    )

# NOTE:use the terms that are annotated with links as documents for each
# site terms
wiki_links = []
for title in sites_data[['stitle']].values:
    try:
        string = HanziConv.toTraditional("。".join(merge_lists(
            [attribute['links'] for attribute in wiki_pages[title[0]].values()])))
        wiki_links.append(string)
    except:
        wiki_links.append("。")


# @REVIEW : Data analysis

# REVIEW : Supervised and Unsupervised Learning
# generate class prior vector, class-term matrix, covariance
# matrices of documents and classes,  document vectors and weighted
# document vectors.

# @NOTE:For input, we gives the index of each training data, the label of each
# training data, and the document content of each training data.
# Also, the cutoff frequecy for filtering the vocaburary words are
# predefined for each kind of documents.
documents = [sites_data[['stitle']], sites_data[['xbody']],
             pd.DataFrame(wiki_summaries), pd.DataFrame(wiki_subtitles), pd.DataFrame(wiki_segs), pd.DataFrame(wiki_links)]

#hfcs = [0., 0.05, 0.01, 0.1, 0.01, 0.]
#lfcs = [0., 0.2, 0.2, 0., 0.2, 0.]
hfcs = [0.]*len(documents)
lfcs = [0.]*len(documents)
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
        pd, sites_data[['stitle']], sites_data[['CAT2']], documents[i], hfcs[i], lfcs[i])
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
        pd, sites_data[['stitle']], sites_data[['stitle']], documents[i], hfcs[i], lfcs[i])
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
#  index = 5 : using wiki segtions
#  index = 6 : using wiki links


# REVIEW: Visualizing and Evaluating the result for explorative analysis.
# here, we use several ways to visualize or generate result,
# in order to understand if the document vector are reasonable.

from visualize_package import *

# @NOTE: plot the generated matrix :
for class_cov_table in sup_class_covs:
    plot_matrix(class_cov_table)

# @NOTE: plot the class 2d embedding
for i in range(len(sup_condis)):
    cat_2d = dimension_reduction(sup_condis[i].transpose())
    plot_word_embedding(plt, cat_2d, num=i, labels=pd.DataFrame(sup_class_covs[0].columns))
plt.show()

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
# DEBUG: Third and Forth are the same
#for i in range(len(sup_w_doc_vecs)):
#    site_2d = dimension_reduction(sup_w_doc_vecs[i].transpose())
#    plot_word_embedding(plt, site_2d, num=i, labels=sites_data[['CAT2']])
#plt.show()

# NOTE:Forth, we use the over-document-summed-conditional-probability-weighted term counts as vector element,
# where each weights on each term is calculated by summing all
# un-supervised-generated conditional probability of term over all documents.
for i in range(len(unsup_w_doc_vecs)):
    site_2d = dimension_reduction(unsup_w_doc_vecs[i].transpose())
    plot_word_embedding(plt, site_2d, num=i, labels=sites_data[['CAT2']])
plt.show()

# NOTE:Fifth, we use the rows of document covariance matrix generated from
# term count as document vector,
for i in range(len(unsup_doc_covs)):
    site_2d = dimension_reduction(unsup_doc_covs[i].transpose())
    plot_word_embedding(plt, site_2d, num=i, labels=sites_data[['CAT2']])
plt.show()

# @NOTE:Then, we try to find similar sites for each vector space using K nearset neighborhood,
# in order to check if the vector space can gives reasonable similar sites.
# By the method below, we were able to check the local structure of each
# vector space.
k = 100
# NOTE:First, we use the conditional probability or importance of terms given document as vector element,
# which is obtain during the unsupvervised NB training phase.
unsup_condis_neighbors = []
for table in unsup_condis:
    unsup_condis_neighbors.append(k_nearest_neighbor(table.transpose(), k))
# NOTE:Second, we use the term count of each document as vector element.
unsup_doc_vecs_neighbors = []
for table in unsup_doc_vecs:
    unsup_doc_vecs_neighbors.append(k_nearest_neighbor(table.transpose(), k))
# NOTE:Third, we use the over-classes-summed-conditional-probability-weighted term counts as vector element,
# where each weight on each term is calculated by summing all
# supervised-generated conditional probability of term over all classes.
sup_w_doc_vecs_neighbors = []
for table in sup_w_doc_vecs:
    sup_w_doc_vecs_neighbors.append(k_nearest_neighbor(table.transpose(), k))
# NOTE:Forth, we use the over-document-summed-conditional-probability-weighted term counts as vector element,
# where each weights on each term is calculated by summing all
# un-supervised-generated conditional probability of term over all documents.
unsup_w_doc_vecs_neighbors = []
for table in unsup_w_doc_vecs:
    unsup_w_doc_vecs_neighbors.append(k_nearest_neighbor(table.transpose(), k))
unsup_doc_covs_neighbors = []
for table in unsup_doc_covs:
    unsup_doc_covs_neighbors.append(k_nearest_neighbor(table.transpose(), k))

# TODO: how to compare two ranking list ?
from measures.rankedlist import *
def compare_two_ranking_lists(table1,table2):
    import numpy as np
    scores = []
    for i in range(len(np.array(table1[0]))):
        scores.append(RBO.score(np.array(table1)[i].tolist(), np.array(table2)[i].tolist()))
    return sum(scores)/len(np.array(table1[0]))


def local_similarity(embeddings):
    import numpy as np
    k = len(embeddings)
    scores_matrix = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            scores_matrix[i,j]=compare_two_ranking_lists(embeddings[i],embeddings[j])
    return scores_matrix

unsup_condis_sim_M = local_similarity(unsup_condis_neighbors)
unsup_doc_vecs_sim_M = local_similarity(unsup_doc_vecs_neighbors)
unsup_w_doc_vecs_sim_M = local_similarity(unsup_w_doc_vecs_neighbors)
sup_w_doc_vecs_sim_M = local_similarity(sup_w_doc_vecs_neighbors)
unsup_doc_covs_sim_M = local_similarity(unsup_doc_covs_neighbors)

# REVIEW:visualize local similarity matrix of each kind of embeddings
def plot_np_matrix(M):
    import matplotlib.pylab as plt
    import numpy as np
    np.fill_diagonal(M, 0.)
    plt.imshow(M,interpolation='nearest')
    plt.colorbar()

unsup_condis_sim_table = pd.DataFrame(unsup_condis_sim_M)
unsup_doc_vecs_sim_table = pd.DataFrame(unsup_doc_vecs_sim_M)
unsup_w_doc_vecs_sim_table = pd.DataFrame(unsup_w_doc_vecs_sim_M)
sup_w_doc_vecs_sim_table = pd.DataFrame(sup_w_doc_vecs_sim_M)
unsup_doc_covs_sim_table = pd.DataFrame(unsup_doc_covs_sim_M)

indexes = ['title','xbody','wiki summary','wiki subtitles','wiki segtions','wiki links']
unsup_condis_sim_table.index=indexes
unsup_condis_sim_table.columns = indexes
unsup_doc_vecs_sim_table.index=indexes
unsup_doc_vecs_sim_table.columns = indexes
unsup_w_doc_vecs_sim_table.index=indexes
unsup_w_doc_vecs_sim_table.columns = indexes
sup_w_doc_vecs_sim_table.index=indexes
sup_w_doc_vecs_sim_table.columns = indexes
unsup_doc_covs_sim_table.index=indexes
unsup_doc_covs_sim_table.columns = indexes

plot_matrix(unsup_condis_sim_table)
plot_matrix(unsup_w_doc_vecs_sim_table)
plot_matrix(unsup_doc_vecs_sim_table)
plot_matrix(unsup_doc_covs_sim_table)

plot_matrix((unsup_doc_vecs_sim_table-unsup_doc_covs_sim_table)/unsup_doc_vecs_sim_table)

# using unsupervised weighted doc vec as embedding is the same as supervised weighted doc vec
# => the weighted are the same no matter supevising label or not.
plt.show()

# NOTE: dimension reduction

info_2d = dimension_reduction(unsup_condis_sim_table.transpose())
plot_word_embedding(plt, info_2d, num=1, labels=pd.DataFrame(indexes),color_on=False)
info_2d = dimension_reduction(unsup_w_doc_vecs_sim_table.transpose())
plot_word_embedding(plt, info_2d, num=2, labels=pd.DataFrame(indexes),color_on=False)
info_2d = dimension_reduction(unsup_doc_vecs_sim_table.transpose())
plot_word_embedding(plt, info_2d, num=3, labels=pd.DataFrame(indexes),color_on=False)
info_2d = dimension_reduction(unsup_doc_covs_sim_table.transpose())
plot_word_embedding(plt, info_2d, num=4, labels=pd.DataFrame(indexes),color_on=False)
plt.show()


matrix_name = ['conditional weighted','term count weighted','term count','term count covariance']
for k in range(6):
    plt.figure(k)
    all_doc_em_sim_M = local_similarity([unsup_condis_neighbors[k],unsup_w_doc_vecs_neighbors[k],unsup_doc_vecs_neighbors[k],unsup_doc_covs_neighbors[k]])
    all_doc_em_sim_table = pd.DataFrame(all_doc_em_sim_M)
    all_doc_em_sim_table.index = matrix_name
    all_doc_em_sim_table.columns = matrix_name
    plot_matrix(all_doc_em_sim_table)

plt.show()


for k in range(6):
    plt.figure(k)
    all_doc_em_sim_M = local_similarity([unsup_condis_neighbors[k],unsup_w_doc_vecs_neighbors[k],unsup_doc_vecs_neighbors[k],unsup_doc_covs_neighbors[k]])
    all_doc_em_sim_table = pd.DataFrame(all_doc_em_sim_M)
    all_doc_em_sim_table.index = matrix_name
    all_doc_em_sim_table.columns = matrix_name
    matrix_2d = dimension_reduction(all_doc_em_sim_table.transpose())
    plot_word_embedding(plt, matrix_2d, num=k, labels=pd.DataFrame(matrix_name),color_on=False,size_='x-large')
plt.show()

# NOTE: analysis of the result:
# 1. using title is similar to using wiki link as documents, since they have similar ranking score in term count unweighted vector space,
# including using basic term counts as vector and using the by-generated document covariance matrix as vectors.
# Both of them have ranking score outlier using conditional matrix as vector space.
# 2. In contrast to above, using body, wiki-summary, wiki-subtitles, and wiki-segtions are more similar in their vector space of term count and wighted term count then simple term count generated vector space.
# especially using wiki-subtitles and wiki-segtions.
# from 1 and 2 we conclude that, wiki link and title have a similar functionality, the weights on term count will effect the vector space a lot.
# 3. the information loss by using covariance document vector from simply term count vector have a ranking :
# Information loss: wiki-segtions > wiki-summary = wiki-subtitles > wiki-link > title > body
# TODO: Since, its reasonable that with larger document, information loss is greater, however,
# body have less information loss, which is strange.
# Maybe one reason is that the vocaburary set of body is smaller then title,
# another reason is that the document matrix of body is already have some low rank property.
# 4. the effect adding wight to term count document vector rank from wiki-link to body:
#    wiki-link > title > wiki-summary = wiki-segtions > wiki-subtitles > body
# 5. the effect of conditional weighted vector space v.s. term count :
#    wiki-link > title = wiki-subtitles = wiki-segtions > wiki-summary > body
# from 4.5.
# maybe terms in body and wiki-summary better present the site then the others
# wiki-link and title have much more noisy term then the others
# wiki-subtitles have better weighted effect on codtional weighted vector space
# wiki-summary have better weighted effect on term count weighted vector space

# also,


#TODO: using EM algorithm to cluster sites, and show the unsupervised learned class importance terms in order to understand each topic
#TODO: using the semi-supervised method as IR HW by EM algorithm, we will use small amount of human-labeled tour related categories from (FB) or (us) as supervised training data
