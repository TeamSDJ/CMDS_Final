# @REVIEW : Read in data
from get_distance_to_site import *
from analysis_package import *
def similar_site_lists(input,k):
    #k = 10
    #input = '中正紀念堂'

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

    prior_table, condi_table, doc_cov_table, class_cov_table, doc_vec_table,w_doc_vec_table = analysis(
        pd, sites_data[['stitle']], sites_data[['stitle']], pd.DataFrame(wiki_links), 0.,0.)



    import numpy as np

    # NOTE: the best space :
    dc_wiki_link = doc_cov_table.transpose().sort_index()

    # NOTE: the second best space :
    ct_wiki_summary = condi_table.transpose().sort_index()



    ranking_lists,score_matrix = k_nearest_neighbor_with_distances(pd.concat([ct_wiki_summary, dc_wiki_link], axis=1), k)

    index = list(ranking_lists.index).index(input)

    def coordinate(name):
        index=np.array(sites_data[['stitle']]).tolist().index([name])
        return np.array(sites_data[['latitude']]).tolist()[index][0],np.array(sites_data[['longitude']]).tolist()[index][0]

    def gettime(name):
        coor = coordinate(name)
        return get_distance_to_site(coor[1],coor[0])
    distances = []
    for site in np.array(ranking_lists)[index,:].tolist():
        try:
            distances.append(gettime(site))
        except:
            distances.append(None)

    return np.array(ranking_lists)[index,:],score_matrix[index,:],distances



def ranking_lists(name,k):

    sites,score,distances = similar_site_lists(name,k)

    dist = distances[0]
    dist_num = []
    for dist in distances:
        dis = 1
        try:
            if('km' in dist):
                # print(float((dist.split(' ')[0]).replace(',',''))*1000)
                dis = float((dist.split(' ')[0]).replace(',',''))*1000
            else:
                # print(float((dist.split(' ')[0]).replace(',','')))
                dis = float((dist.split(' ')[0]).replace(',',''))
        except:
            None
        dist_num.append(dis)
    X = sites.tolist()
    Y = score*np.array(dist_num).tolist()
    keydict = dict(zip(X, Y))
    X.sort(key=keydict.get)
    return X



sites_data[['stitle']]


original_ranking_list,_,_ = similar_site_lists('國立台北教育大學',30)
original_ranking_list.tolist()
ranking_lists('國立臺北教育大學',20)
