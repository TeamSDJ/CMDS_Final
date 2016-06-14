*facebook graph api access token EAACEdEose0cBAG7aS3xZAlQpKdUHySK67lskzS6FURnTuTkrT4bM9aqhm04leXOXFMpmufDG1RMoU8YWDooySDaaqSbTGmGudmgKYr5KkpmSBzyRc3xT1irZBwV9nDKkTZA2DZAX4ncj0QXWa99DZCoLMOWD7TdKaZBxAzlH3pqAZDZD

# 1. in this analysis work, we come up with using wiki pages and other infos of the site as a guide of similarity between
# sites and further categories sites. The definition of the similarity should not include the closeness of their location!
# 2. I train the model using supervised NB (11 classes) and un-supervised NB (all documents as classes)
# the input data includes names of the site,
  summary, sub-titles, sub title contents of wiki pages of each site terms.

# 3. We tried variance mixing of each kind of inputs and observed

```
    the posterior probability of each class (doc),
    the conditional probability of each term given a class (or doc)
    the document vector with term count as feature
    the document vector with each term count weighted by the sum of each term-conditional-probability over all classes.
    the covariance matrix of each class calculated by the vector with their corresponding conditional term probability as the feature
    the covariance matrix of document using term count as feature
```

# 4. Using the covariance matrix of classes, we can observed the relationship between classes
# 5. Using the dimension reduction (tf-idf), we can observed if all distance between each document are of reasonable similarity
# 6. Using the k_nearest_neighborhood method, we retrieved k nearest sites allowing us the judge their similarity pair by pair.
**The result shows that :

# 1. similarity of location are taken into account due to the repeat of term site name, which should not be included in our similarity definition. For example, "北投圖書館", "北投溫泉區", etc

# TODO:
1. Try facebook graph API for better site tourism description.
2. Try using facebook graph API to get for sites.
3. Try first analysis the term meanings to get the similarity between term pairs by its sub-content or summary on wiki,
then develop a site similarity metric that add them into consideration.
4. Try to use Bing as the document generation engine, find the suitable query for each site-name and analysis using previous methods.


# TODO : method
1. 在所挖掘出的景點維基百科中，所有的景點都能夠得到一個頁面，描述了該景點的介紹。其中，每個景點介紹中包含了許多大標，如：歷史、人文、觀光等等。
若能比較每個景點中的子標題是否接近，即可得到每個景點的相似程度，也就是說，若能夠得到子標題的意涵，就能夠依據為得到景點的意涵。
比如說，一個具有溫泉性質的景點，其大多的子標題，極為溫泉相關的名詞，若為一個具有古蹟性質的景點，其子標題則大多為歷史介紹、文化意涵等等。
然而，要得到每個子標題的意涵，即是一個未知的課題，在這裡，我們假設，子標題所含有的詞若相似，其內文應該也會相似，以其作為新的標題去維基百科搜尋所得到的頁面，也應該會有相似的子標題。相似度以其所出現的內文做為依據。
