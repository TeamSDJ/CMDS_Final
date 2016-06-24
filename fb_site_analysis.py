# REVIEW : re-craw all google map sites that are in facebook !
import pickle
import six.moves.cPickle as pickle
with open("taiwan_sites_geo_infos.dat", 'rb') as f:
    geo_infos = pickle.load(f)


all_sites = []
for key in geo_infos.keys():
    if(geo_infos[key]!=None):
        all_sites.extend(geo_infos[key])


for site in all_sites_dict.values():
    print(site['name'])
    print(site['category_list'])


fb_site_names = []
for site in all_sites:
    fb_site_names.append(site['name'])


from analysis_package import *
from hanziconv import HanziConv

fb_site_categories = []
for site in all_sites:
    categories = []
    #print(site['name'])
    for cat in site['category_list']:
        categories.append(cat['name'])
    fb_site_categories.append((" ".join(categories)).replace("&amp;"," ").replace("/"," "))
    #print((" ".join(categories)).replace("&amp;"," "))


documents = [pd.DataFrame(fb_site_categories)]

import pandas as pd
fb_site_table = pd.DataFrame(fb_site_names)
# NOTE:Unsupervised Learning :
unsup_priors = []
unsup_condis = []
unsup_doc_covs = []
unsup_class_covs = []
unsup_doc_vecs = []
unsup_w_doc_vecs = []

for i in range(len(documents)):
    prior_table, condi_table, doc_cov_table, class_cov_table, doc_vec_table, w_doc_vec_table = analysis(
        pd, fb_site_table, fb_site_table, documents[i],0.,0.)
    unsup_priors.append(prior_table)
    unsup_condis.append(condi_table)
    unsup_doc_covs.append(doc_cov_table)
    unsup_class_covs.append(class_cov_table)
    unsup_doc_vecs.append(doc_vec_table)
    unsup_w_doc_vecs.append(w_doc_vec_table)


unsup_doc_vecs[0]
