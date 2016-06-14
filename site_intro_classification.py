


# start training using NB classifier
import classification_package



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


prior_table1,condi_table1,doc_cov_table1,class_cov_table1 = analysis(sites_data[['stitle']],sites_data[['CAT2']],sites_data[['stitle']],0.,0.)
prior_table2,condi_table2,doc_cov_table2,class_cov_table2 = analysis(sites_data[['stitle']],sites_data[['CAT2']],sites_data[['xbody']],0.01,0.3)
plot_matrix(prior_table2)
plot_matrix(condi_table2)
plot_matrix(doc_cov_table2)
def compare_table_values(table1,table2):
    import matplotlib.pylab as plt
    import numpy as np
    (W,L) = np.shape(np.matrix(table1))
    plt.scatter(np.reshape(np.matrix(table1),(W*L,1)),np.reshape(np.matrix(table2),(W*L,1)))
    plt.show()

compare_table_values(class_cov_table1,class_cov_table2)
compare_table_values(doc_cov_table1,doc_cov_table2)
compare_table_values(prior_table1,prior_table2)


prior_table3,condi_table3,doc_cov_table3,class_cov_table3 = analysis(sites_data[['stitle']],sites_data[['stitle']],sites_data[['xbody']],0.2,0.7)


prior_table3,condi_table3,doc_cov_table3,class_cov_table3 = analysis(sites_data[['stitle']],sites_data[['stitle']],sites_data[['xbody']],0.0,0.0)
compare_table_values(doc_cov_table3,class_cov_table3)
