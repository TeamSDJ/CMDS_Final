import pandas as pd
import numpy as np
def edition_distance(str1,str2):
    import numpy as np
    M = np.zeros((len(str1)+1,len(str2)+1),int)
    #initialize the first rol and col
    for i in range(len(str1)+1):
        M[i,0]=i
    for i in range(len(str2)+1):
        M[0,i]=i
    #generate dynamic programming matrix
    for i in range(len(str1)):
        for j in range(len(str2)):
            M[i+1,j+1]=min(M[i,j]+(0 if str1[i]==str2[j] else 1), M[i,j+1]+1,M[i+1,j]+1)

    result = M[-1,-1]
    return result

def generate_combination(tokens):
    combs = []
    for i in range(len(tokens)):
        for j in range(i+1):
            combs.append(''.join(tokens[0+j:len(tokens)-i+j]))
    return combs

def merge_lists(lists):
    return [item for sublist in lists for item in sublist]

def vectorize_docs(token_list, V):
    V_dict = dict(zip(V, range(len(V))))  # use dictionary for faster indexing
    from scipy.sparse import lil_matrix
    from scipy.sparse import csr_matrix
    doc_vec_M = lil_matrix((len(V) + 1, len(token_list)), dtype=np.int16)

    def find_index(key):
        try:
            return V_dict[key]
        except:
            return -1
    find_index_map = np.vectorize(find_index)
    key_list = []
    doc_index_list = []
    count_list = []
    for i in range(len(token_list)):
        try:
            key, count = np.unique(token_list[i], return_counts=True)
            # remove keys that are not contrain in voc list
            key_list.extend(find_index_map(key).tolist())
            count_list.extend(count)
            doc_index_list.extend([i] * len(key))
        except:
            None  # print('doc ', i, ' ,: term num = ', len(key))
    doc_vec_M[key_list, doc_index_list] = np.matrix(count_list)
    return csr_matrix(doc_vec_M)[:-1, :]


def maximum_likelihood_estimation(doc_vecs, posterior, smooth_const):
    prior = np.sum(posterior, 0)
    prior = prior + smooth_const
    prior = prior / np.sum(prior)
    condprob = doc_vecs * posterior
    condprob = condprob + smooth_const
    condprob = condprob / np.sum(condprob, 0)
    return prior, condprob


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


sites_data[['stitle','CAT2','longitude', 'latitude']]
sites_data.columns.values
site_names = [e[0] for e in sites_data[['stitle']].values.tolist()]
cat_names = [e[0] for e in sites_data[['CAT2']].values.tolist()]


cat_set = set(cat_names)
import collections
collections.Counter(cat_names)
import google

import tldextract

domains = []
urls = []
count = 0
for url in google.search(site_names[0], stop=100, pause=0.1):
    try:
        urls.append(url)
        domains.append(tldextract.extract(url)[1])
        count = count + 1
    except:
        count = count + 1
        None
count

len(urls)
len(domains)

##############################################################
# using Bing Web Search API
# https://github.com/tristantao/py-bing-search
# Primary Account Key	   hkxX68ZPyntLT6qdbLnlbk6NqXCv5fQadYm0BGSUER8
# Customer ID	           27000f77-3c8b-4f96-8640-d010ed81577e
# Account Keys             hkxX68ZPyntLT6qdbLnlbk6NqXCv5fQadYm0BGSUER8
from py_bing_search import PyBingWebSearch
search_term = "Python Software Foundation"
# web_only is optional, but should be true to use your web only quota
# instead of your all purpose quota
bing_web = PyBingWebSearch('hkxX68ZPyntLT6qdbLnlbk6NqXCv5fQadYm0BGSUER8', site_names[0], web_only=False)
first_fifty_result= bing_web.search(limit=50, format='json') #1-50
#second_fifty_result= bing_web.search(limit=50, format='json') #51-100


first_fifty_result
for result in first_fifty_result:
    print(result.title)
    print(result.url)
    print(result.description)

################################################################
# using goole search of taiwan tourism buera to predict category
# URL : http://eventaiwan.tw/tw/cms/siteSearchAction.do?method=doSearchForSite&isCMS=true&keyword=中正紀念堂

#TODO : using wiki pages information to predict the category of sites
#https://wikipedia.readthedocs.io/en/latest/code.html
import wikipedia
wikipedia.set_lang("zh-tw")
wikipedia.summary(site_names[1])
print(wikipedia.page(site_names[1]).content)
wikipedia.page(site_names[1]).title
wikipedia.page(site_names[1]).links

#TODO:Chinese segmentation
#结巴中文分词
#作者：Fooying
#链接：https://www.zhihu.com/question/19651613/answer/15607598
#来源：知乎
#著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
import jieba
jieba.set_dictionary('dict.txt.big.txt')
for site_name in site_names:
    print(", ".join(jieba.cut(site_name, cut_all=False)))

for cat in cat2_names:
    print(", ".join(jieba.cut(cat, cut_all=False)))

#TODO : using the searched web description, title, content and ranking to predict the category of sites
#TODO : using the wiki page title, content related to “景點”、'遊'、‘觀光’ and ranking to predict the category of sites
wikipedia.summary('computer')
wikipedia.page('台北 歷史').content
wikipedia.page('台北 歷史').title
wikipedia.page('台北 歷史').links
wikipedia.page('台北 歷史').images
wikipedia.suggest("computer").images
wikipedia.donate()
#TODO: structurlize wiki page content
# package for parsing the wiki pages : https://github.com/5j9/wikitextparser
import wikitextparser as wtp
import wikipedia
import jieba
import jieba.analyse
import jieba.posseg as pseg
jieba.set_dictionary('dict.txt.big.txt')
jieba.enable_parallel(4)
jieba.analyse.set_stop_words('stop_words.txt')
jieba.analyse.set_idf_path('idf.txt.big.txt')


jieba.analyse.extract_tags(pages[0][1].content, topK=50, allowPOS=('ns', 'n', 'vn','v'))

jieba.analyse.textrank(pages[0][1].content, topK=50, withWeight=False, allowPOS=('ns', 'n', 'vn'))



wikipedia.set_lang("zh-tw")
site_name = site_names[0]
generate_combination(tokens)
def get_pages_from_wiki(site_name):
    tokens = [l[0] for l in list(jieba.tokenize(site_name))]
    count = 0
    no_page = True
    pages = []
    for term in generate_combination(tokens):
        count = count + 1
        try:
            page = wikipedia.page(term)
            if(page.title==term):
                pages.append(page)
                print(term)
                print(page.title)
        except:
            None
    return pages



page_test = get_pages_from_wiki(site_names[-1])
page_test[0].content

wikipedia.search('李臨秋 故居')
page_test.sections
site_name=site_names[0]

pages = []
for site_name in site_names:
    pages.append(get_pages_from_wiki(site_name))


for i in range(len(site_names)):
    print(site_names[i])
    print(pages[i])

pages[0][0].title
wtp.parse(pages[0][0].content).sections[3]
wtp.parse(pages[0][0]).wikilinks



#TODO: adding stop words removing steps to text pre-processing

#TODO:找出所有子標題的字詞與其所對應的內文字詞，建立成一個表格。
titles = []
sub_titles = []
sub_titles_terms = []
for page in merge_lists(pages):
    print(page.title)
    titles.append(page.title)
    for sec in wtp.parse(page.content).sections:
        #print()
        sub_titles.append(jieba.analyse.extract_tags(sec.title, topK=100))
        #print()
        sub_titles_terms.append(jieba.analyse.textrank(" ".join(str(sec).splitlines()[1:]), topK=1000, withWeight=True, allowPOS=('ns', 'n', 'vn')))
        #print('=================================')
        #sub_titles.append(" ".join(jieba.cut(sec.title, cut_all=False)).split())
        #sub_titles_tokens.append(" ".join(jieba.cut(sec, cut_all=False)).split())


def merge_two_dict(x,y):
    return {k: x.get(k,0) + y.get(k,0) for k in x.keys() | y.keys()}


# analysis on term similarity of differet sub_titles
table = dict()
for sub_title,sub_title_terms in zip(sub_titles,sub_titles_terms):
    for term in sub_title:
        if term in table.keys():
            table[term]=merge_two_dict(table[term],dict(sub_title_terms))
        else:
            table[term] = dict(sub_title_terms)




table['香港']
# TODO: construct voc for each document
V=list(set(merge_lists([list(values.keys()) for values in list(table.values())])))


# TODO: construct vector for each document
doc_vec = []
doc_title = []
for title,terms in table.items():
    doc_title.append(title)
    vec = [0.]*len(V)
    for key,value in terms.items():
        vec[V.index(key)] = value
    doc_vec.append(vec)

# TODO: do pca for dimensional reduction
from scipy.sparse import csr_matrix
doc_vec_M = csr_matrix(doc_vec)
np.shape(doc_vec_M)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(doc_vec_M.toarray(), y=None)

import matplotlib.pylab as plt

plt.scatter(pca_result[:,0],pca_result[:,1])
plt.show()




get_pages_from_wiki(site_name)
site_name
len(pages)

#TODO:找出所有景點名稱中的字詞及其所對應到的維基頁中，對應的子標題字詞

#TODO:找出所有景點名稱中的字詞及其所對應到的維基頁中，對應的總結字詞





#TODO: combine site tags from different site or database and do analysis on their documents to estimate the similarity of tags
# EX:Website



#similarity on tokenized site names

site_tokens = []
for site_name in site_names:
    tokens = " ".join(jieba.cut(site_name, cut_all=False)).split()
    site_tokens.append(tokens)


import collections
vocs = collections.Counter(merge_lists(site_tokens))
V = list(vocs)
site_tokens

title_vecs_M = vectorize_docs(site_tokens,V)

np.shape(title_vecs_M)


cat_set
cat_list = list(cat_set)
posterior_train = []
for cat in cat_names:
    class_posterior = [0.] * len(cat_list)
    class_posterior[cat_list.index(cat)] = 1.
    posterior_train.append(class_posterior)
posterior_train_M = np.matrix(posterior_train)

prior, condprob = maximum_likelihood_estimation(title_vecs_M, posterior_train_M, 0.1)

def common_words(c,k,V,condprob):
    return collections.Counter(dict(zip(V,condprob[:,c].transpose().tolist()[0]))).most_common(k)


for i in range(len(cat_list)):
    print(cat_list[i])
    print(common_words(i,10,V,condprob))


# TODO : method
# 1. 在所挖掘出的景點維基百科中，所有的景點都能夠得到一個頁面，描述了該景點的介紹。其中，每個景點介紹中包含了許多大標，如：歷史、人文、觀光等等。
# 若能比較每個景點中的子標題是否接近，即可得到每個景點的相似程度，也就是說，若能夠得到子標題的意涵，就能夠依據為得到景點的意涵。
# 比如說，一個具有溫泉性質的景點，其大多的子標題，極為溫泉相關的名詞，若為一個具有古蹟性質的景點，其子標題則大多為歷史介紹、文化意涵等等。
# 然而，要得到每個子標題的意涵，即是一個未知的課題，在這裡，我們假設，子標題所含有的詞若相似，其內文應該也會相似，以其作為新的標題去維基百科搜尋所得到的
# 頁面，也應該會有相似的子標題。相似度以其所出現的內文做為依據。屏除
#
