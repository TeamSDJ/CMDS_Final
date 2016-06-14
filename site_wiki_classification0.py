# REVIEW : read in data

import pickle
import six.moves.cPickle as pickle
# load data
with open("pages.dat",'rb') as f:
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
video_url = pd.read_excel('sitedata.xlsx', 'video',index_col='file_Id', na_values=['NA'])


# REVIEW: constructing wiki pages string for each documents
from classification_package import *

wiki_summaries = []
for title in sites_data[['stitle']].values:
    wiki_summaries.append("".join([attribute['summary'] for attribute in wiki_pages[title[0]].values()]))

wiki_subtitles = []
for title in sites_data[['stitle']].values:
    wiki_subtitles.append("。".join(merge_lists([list(dict(attribute['sections']).keys()) for attribute in wiki_pages[title[0]].values()])))


wiki_whole_page = []
for title in sites_data[['stitle']].values:
    wiki_whole_page.append("".join([attribute['summary'] for attribute in wiki_pages[title[0]].values()]))


# REVIEW : data analysis

import pandas as pd
import numpy as np
#title_data,cat_data,text_data = sites_data[['stitle']],sites_data[['CAT2']],sites_data[['stitle']]
#hfc = 0.
#lfc = 0.

# supervised learning
prior_table1,condi_table1,doc_cov_table1,class_cov_table1,doc_vec_table1 = analysis(pd,sites_data[['stitle']],sites_data[['CAT2']],sites_data[['stitle']],0.,0.)
prior_table2,condi_table2,doc_cov_table2,class_cov_table2,doc_vec_table2 = analysis(pd,sites_data[['stitle']],sites_data[['CAT2']],sites_data[['xbody']],0.01,0.2)
prior_table3,condi_table3,doc_cov_table3,class_cov_table3,doc_vec_table3 = analysis(pd,sites_data[['stitle']],sites_data[['CAT2']],pd.DataFrame(wiki_summaries),0.01,0.2)
prior_table4,condi_table4,doc_cov_table4,class_cov_table4,doc_vec_table4 = analysis(pd,sites_data[['stitle']],sites_data[['CAT2']],pd.DataFrame(wiki_subtitles),0.0,0.0)

# unsupervised learning
prior_table5,condi_table5,doc_cov_table5,class_cov_table5,doc_vec_table5 = analysis(pd,sites_data[['stitle']],sites_data[['stitle']],sites_data[['stitle']],0.,0.)
prior_table6,condi_table6,doc_cov_table6,class_cov_table6,doc_vec_table6 = analysis(pd,sites_data[['stitle']],sites_data[['stitle']],sites_data[['xbody']],0.01,0.2)
prior_table7,condi_table7,doc_cov_table7,class_cov_table7,doc_vec_table7 = analysis(pd,sites_data[['stitle']],sites_data[['stitle']],pd.DataFrame(wiki_summaries),0.01,0.2)
prior_table8,condi_table8,doc_cov_table8,class_cov_table8,doc_vec_table8 = analysis(pd,sites_data[['stitle']],sites_data[['stitle']],pd.DataFrame(wiki_subtitles),0.0,0.0)


def plot_word_embedding(plt,table,labels=None,title='',num=1):
    plt.figure(num)
    vectors = np.matrix(table).tolist()
    words = list(table.index)

    import matplotlib
    if(type(labels)==type(None)):
        None
        colors = None
    else:
        label_set = list(set(list(labels.values.transpose().tolist())[0]))
        def get_spaced_colors(n):
            max_value = 16581375 #255**3
            interval = int(max_value / n)
            colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

            return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
        colors = get_spaced_colors(len(label_set))

    for i in range(len(words)):
        point = vectors[i]
        word = words[i]
            # plot points
        plt.scatter(point[0], point[1])
        # plot word annotations
        if(type(labels)==type(None)):

            plt.annotate(
                word,
                xy = (point[0], point[1]),
                size = "x-small"
            )
        else:
            label_index = label_set.index(list(labels.values.transpose().tolist())[0][i])
            plt.annotate(
                word,
                xy = (point[0], point[1]),
                color = '#'+"".join(list(map(lambda x:format(x, '#04x')[2:],colors[label_index]))).upper(),
                size = "x-small"
            )

    plt.tight_layout()
    plt.title(title)




def dimension_reduction(table):
    from sklearn.manifold import TSNE
    import pandas as pd
    tsne = TSNE(n_components=int(2), perplexity=30.0, early_exaggeration=10.0, learning_rate=1000.0,  n_iter=3000, metric='euclidean', init='pca')
    result = tsne.fit_transform(np.matrix(table))
    return pd.DataFrame(result,index=table.index)

import matplotlib.pylab as plt
class_2d = dimension_reduction(condi_table1.transpose())
plot_word_embedding(plt,class_2d,num=1)


class_2d = dimension_reduction(condi_table2.transpose())
plot_word_embedding(plt,class_2d,num=2)
class_2d = dimension_reduction(condi_table3.transpose())
plot_word_embedding(plt,class_2d,num=3)
class_2d = dimension_reduction(condi_table4.transpose())
plot_word_embedding(plt,class_2d,num=4)
plt.show()


import matplotlib.pylab as plt
site_2d = dimension_reduction(condi_table5.transpose())
plot_word_embedding(plt,site_2d,num=1,labels=sites_data[['CAT2']])

site_2d = dimension_reduction(condi_table6.transpose())
plot_word_embedding(plt,site_2d,num=2,labels=sites_data[['CAT2']])

site_2d = dimension_reduction(condi_table7.transpose())
plot_word_embedding(plt,site_2d,num=3,labels=sites_data[['CAT2']])

site_2d = dimension_reduction(condi_table8.transpose())
plot_word_embedding(plt,site_2d,num=4,labels=sites_data[['CAT2']])
plt.show()


import matplotlib.pylab as plt
site_2d = dimension_reduction(doc_vec_table1.transpose())
plot_word_embedding(plt,site_2d,num=1,labels=sites_data[['CAT2']])

site_2d = dimension_reduction(doc_vec_table2.transpose())
plot_word_embedding(plt,site_2d,num=2,labels=sites_data[['CAT2']])

site_2d = dimension_reduction(doc_vec_table3.transpose())
plot_word_embedding(plt,site_2d,num=3,labels=sites_data[['CAT2']])

site_2d = dimension_reduction(doc_vec_table4.transpose())
plot_word_embedding(plt,site_2d,num=4,labels=sites_data[['CAT2']])
plt.show()


# an embedding consider all info
doc_vec_all = pd.concat([doc_vec_table1,doc_vec_table2,doc_vec_table3,doc_vec_table4])
site_2d = dimension_reduction(doc_vec_all.transpose())
plot_word_embedding(plt,site_2d,num=1,labels=sites_data[['CAT2']])
plt.show()


import numpy as np


# REVIEW : do ploting and comparison
import matplotlib.pylab as plt

plot_matrix(class_cov_table1>(class_cov_table1+class_cov_table2+class_cov_table3)/3)

plot_matrix(class_cov_table2>(class_cov_table1+class_cov_table2+class_cov_table3)/3)

plot_matrix(class_cov_table3>(class_cov_table1+class_cov_table2+class_cov_table3)/3)

plot_matrix(class_cov_table4)
