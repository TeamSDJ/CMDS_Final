
#REVIEW functions for NB classifier
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
def filter_words(voc_count, fc_h, fc_l):
    import collections
    N = len(voc_count)
    voc_count.subtract(collections.Counter(
        dict(voc_count.most_common(int(N * fc_h)))))
    voc_count = collections.Counter(
        dict(voc_count.most_common(int(N * (1. - fc_h - fc_l)))))
    return collections.Counter({k: v for k, v in voc_count.items() if v > 0})
def vectorize_docs(token_list, V):
    import numpy as np
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
    import numpy as np
    prior = np.sum(posterior, 0)
    prior = prior + smooth_const
    prior = prior / np.sum(prior)
    condprob = doc_vecs * posterior
    condprob = condprob + smooth_const
    condprob = condprob / np.sum(condprob, 0)
    return prior, condprob


# tokenize texts


def initialize_tokenizer():
    import jieba
    import jieba.analyse
    import jieba.posseg as pseg
    jieba.set_dictionary('dict.txt.big.txt')
    jieba.enable_parallel(4)
    jieba.analyse.set_stop_words('stop_words.txt')
    jieba.analyse.set_idf_path('idf.txt.big.txt')
    jieba.initialize()
    stopwords = set([e.decode('utf8').splitlines()[0] for e in open('stop_words.txt','rb').readlines()])
    return (stopwords,jieba)

def tokenize(name,tokenizer):
    (stopwords,jieba) = tokenizer
    #jieba.enable_parallel(4)
    try:
        original_tokens = jieba.tokenize(name)
    except ValueError:
        print(name,'not a uni-code')
        return
    tokens = []
    for term in original_tokens:
        if term[0] in stopwords:
            None
        else:
            tokens.append(term[0])

    return tokens
def common_words(c,k,V,condprob):
    import collections
    return collections.Counter(dict(zip(V,condprob[:,c].transpose().tolist()[0]))).most_common(k)

def analysis(pd,title_data,cat_data,text_data,hfc=0.,lfc=0.):
    import collections
    import numpy as np

    # input : document classes array, documents texts
    # calculate conditional probability of each term in given a class
    # return
    # 1. the converiance matrix of each class calculated by their content
    # 2. the converiance matrix of each document calculated by their content
    # 3. the conditional probability of each term given a class
    # 4. list of the vocaburary V
    # 5. list of the classes
    # 6. prior of each classes
    # 7. show the first 10 common words for each class

    #convert pandas to numpy
    site_names = [e[0] for e in title_data.values.tolist()]
    cat_names = [e[0] for e in cat_data.values.tolist()]
    site_body = [e[0] for e in text_data.values.tolist()]

    tokenizer = initialize_tokenizer()
    site_body_tokens = [tokenize(sb,tokenizer) for sb in site_body]

    vocs = filter_words(collections.Counter(merge_lists(site_body_tokens)),hfc,lfc)
    V = list(vocs)

    body_vecs_M = vectorize_docs(site_body_tokens,V)

    # setup the posterior
    cat_list = list(set(cat_names))

    posterior_train = []
    for cat in cat_names:
        class_posterior = [0.] * len(cat_list)
        class_posterior[cat_list.index(cat)] = 1.
        posterior_train.append(class_posterior)
    posterior_train_M = np.matrix(posterior_train)

    prior, condprob = maximum_likelihood_estimation(body_vecs_M, posterior_train_M, 0.1)

    # generate weighted document vectors
    weights = np.sum(condprob,1)

    weighted_doc_vec_M=body_vecs_M.toarray()*np.array(weights.tolist())


    for i in range(len(cat_list)):
        print(cat_list[i])
        print(common_words(i,50,V,condprob))

    class_cov_M = np.cov(condprob.transpose())
    doc_cov_M = np.cov(body_vecs_M.toarray().transpose())
    prior_table = pd.DataFrame(prior,columns=cat_list,index=['prior'])
    condi_table = pd.DataFrame(condprob,columns=cat_list,index=V)
    doc_cov_table = pd.DataFrame(doc_cov_M,columns=site_names,index=site_names)
    class_cov_table = pd.DataFrame(class_cov_M,columns=cat_list,index=cat_list)
    weighted_doc_vec_table = pd.DataFrame(np.matrix(body_vecs_M.toarray()),columns=site_names,index=V)
    doc_vec_table = pd.DataFrame(np.matrix(weighted_doc_vec_M),columns=site_names,index=V)
    return prior_table,condi_table,doc_cov_table,class_cov_table,doc_vec_table,weighted_doc_vec_table



def plot_matrix(table):
    import matplotlib.pyplot as plt
    import numpy as np
    M = np.matrix(table)
    (W,H) = np.shape(M)
    if(W==H):
        np.fill_diagonal(M, 0.)
    plt.imshow(M,interpolation='nearest')
    plt.colorbar()
    tick_marks_x = np.arange(len(list(table.columns)))
    tick_marks_y = np.arange(len(list(table.index)))
    plt.xticks(tick_marks_x, list(table.columns), rotation=45)
    plt.yticks(tick_marks_y, list(table.index))
    plt.show()

def compare_table_values(table1,table2):
    import matplotlib.pylab as plt
    import numpy as np
    (W,L) = np.shape(np.matrix(table1))
    plt.scatter(np.reshape(np.matrix(table1),(W*L,1)),np.reshape(np.matrix(table2),(W*L,1)))
    plt.show()
