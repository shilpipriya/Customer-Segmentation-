#DATA PREPARATION 

#importing the dataset 
import pandas as pd
import numpy as np

#for data visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the dataset 
data=pd.read_csv('data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})

#printing the dimension of the dataset 
data.shape
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

#To gives some infos on columns types and number of null values
tab_info=pd.DataFrame(data.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(data.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(data.isnull().sum()/data.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
display(tab_info)

#show first five lines in dataset
data.head()

#by looking at the table info, i know that approx 25% of data is not assigned
#to customer ID. With the data available it is not possible to predict the
#data of this missing values. So,I'll drop the customer data which 
#contains missing customer id values

data.dropna(axis = 0, subset = ['CustomerID'], inplace = True)
data.shape

#Again, To gives some infos on columns types and number of null values remaining 
#after dropping all missing values w.r.t Customer id
tab_info=pd.DataFrame(data.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(data.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(data.isnull().sum()/data.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
display(tab_info)

#Now, our dataset is 100% filled!
#Looking for any duplicate entry in our dataset!
format(data.duplicated().sum())
#Now, I can see that our data contains some duplicate entry
#so, i'll also delete them as it is of no use
data.drop_duplicates(inplace = True)

#Exploring the content of dataset 
#This dataframe contains 8 variables that correspond to:

#InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
#StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
#Description: Product (item) name. Nominal.
#Quantity: The quantities of each product (item) per transaction. Numeric.
#InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
#UnitPrice: Unit price. Numeric, Product price per unit in sterling.
#CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
#Country: Country name. Nominal, the name of the country where each customer resides.


#Now, I'll look for the country from which orders were made and 
#the number of orders made from each country
countries = data['Country'].value_counts()
print("No. of Countries from which orders were made: ",len(countries))

print("No of orders per country!!")
countries

# Fixing random state for reproducibility
#np.random.seed(1961)
plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(countries.index,countries, align='center')
ax.set_yticks(countries.index)
ax.set_yticklabels(countries.index)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Orders')
ax.set_ylabel('Countries')
ax.set_title('Orders per Country')
plt.show()


#PIE chart
fig1, ax1 = plt.subplots()
ax1.pie(countries,labels=countries.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

#We see that the datset is largely populated by the orders from UK
#approximately 88.8% of our data contains orders from UK

#The dataframe contains 400,000 entries. 
#What are the number of users and products in these entries ?

pd.DataFrame([{'products': len(data['StockCode'].value_counts()),    
               'transactions': len(data['InvoiceNo'].value_counts()),
               'customers': len(data['CustomerID'].value_counts()),  
              }], columns = ['products', 'transactions', 'customers'], index = ['quantity'])

#Now, determine the number of products purchased in every transaction

temp = data.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
nb_products_per_basket = temp.rename(columns = {'InvoiceDate':'Number of products'})
nb_products_per_basket[:10].sort_values('CustomerID')

#Here C in invoice number indicates the order that has been cancelled
#We can see the canceled order 12346 has similar order which was made except the invoice number
#This dataset also contains users like 12347 which shopped several times
#and bought multiple products at a time
# and the users like 12346 who shopped only a single time and bought just one product

#I'll calculate the number of transactions corresponding to cancelled orders
nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x:int('C' in x))
display(nb_products_per_basket[:5])

n1 = nb_products_per_basket['order_canceled'].sum()
n2 = nb_products_per_basket.shape[0]
print('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))

#Number of cancelled orders is 16% of the total orders. Quite large though!
#As mentioned above there are transaction like 12346 which made similar order to the cancelled order
#So, i'll check how many of the cancelled orders follows this trend or not!

df_check = data[data['Quantity'] < 0][['CustomerID','Quantity','StockCode','Description','UnitPrice']]
#if we find any cancelled order for which our dataset doesn't contain similar
#order, then our hypothesi is wrong

for index, col in  df_check.iterrows():
    if data[(data['CustomerID'] == col[0]) & (data['Quantity'] == -col[1]) 
                & (data['Description'] == col[2])].shape[0] == 0: 
        print(df_check.loc[index])
        print(15*'-'+'>'+' HYPOTHESIS NOT FULFILLED')
        break
    
#We see that the initial hypothesis is not fulfilled because of the existence of a 'Discount' entry. I check again the hypothesis but this time discarding the 'Discount' entries:
    
df_check = data[(data['Quantity'] < 0) & (data['Description'] != 'Discount')][['CustomerID','Quantity','StockCode','Description','UnitPrice']]

for index, col in  df_check.iterrows():
    if data[(data['CustomerID'] == col[0]) & (data['Quantity'] == -col[1]) & (data['Description'] == col[2])].shape[0] == 0: 
        print(index, df_check.loc[index])
        print(15*'-'+'>'+' HYPOTHESIS NOT FULFILLED')
        break

#Once more, we find that the initial hypothesis is not verified. 
#Hence, cancellations do not necessarily correspond to orders that would have been made beforehand.

#Now, I'll check the number of cancelled entry that exist with counterpart and 
#the number of doubtfull entry thata exists without counterpart
        
df_cleaned = data.copy(deep = True)
df_cleaned['QuantityCanceled'] = 0

entry_to_remove = [] ; doubtfull_entry = []

for index, col in  data.iterrows():
    if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
    df_test = data[(data['CustomerID'] == col['CustomerID']) &
                         (data['StockCode']  == col['StockCode']) & 
                         (data['InvoiceDate'] < col['InvoiceDate']) & 
                         (data['Quantity']   > 0)].copy()
    #_________________________________
    # Cancelation WITHOUT counterpart
    if (df_test.shape[0] == 0): 
        doubtfull_entry.append(index)
    #________________________________
    # Cancelation WITH a counterpart
    elif (df_test.shape[0] == 1): 
        index_order = df_test.index[0]
        df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
        entry_to_remove.append(index)        
    #______________________________________________________________
    # Various counterparts exist in orders: we delete the last one
    elif (df_test.shape[0] > 1): 
        df_test.sort_index(axis=0 ,ascending=False, inplace = True)        
        for ind, val in df_test.iterrows():
            if val['Quantity'] < -col['Quantity']: continue
            df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index) 
            break               

print("entry_to_remove: ",len(entry_to_remove))
print("doubtfull_entry: ",len(doubtfull_entry))

#delete these entries
df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)
df_cleaned.drop(doubtfull_entry, axis = 0, inplace = True)

#Number of cancelled orders that still exist in the dataset
remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
print("nb of entries to delete: ",remaining_entries.shape[0])
remaining_entries[:5]

#we see that the quantity cancelled is greater than the sum of the previous purchases
#Exploring stockCode

list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
list_special_codes

for code in list_special_codes:
    print("{:<15} -> {:<30}".format(code, df_cleaned[df_cleaned['StockCode'] == code]['Description'].unique()[0]))

#add a column of total price of every purchase
df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
df_cleaned.sort_values('CustomerID')[:5]

#In the dataset every entry specifies the quatity and amount of single order
#hence our order is split over several lines. Now,  I collect all the purchases made during a single order to recover the total order price
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice':'Basket Price'})

df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])

basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID')[:6]

#Calculate the number of basket which lies in the given price range
price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
count_price = []
for i, price in enumerate(price_range):
    if i == 0: continue
    val = basket_price[(basket_price['Basket Price'] < price) &
                       (basket_price['Basket Price'] > price_range[i-1])]['Basket Price'].count()
    count_price.append(val)

#pie chart to view the number of basket price that lies in the given range
       
plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(11, 6))
colors = ['yellowgreen', 'gold', 'wheat', 'c', 'violet', 'royalblue','firebrick']
labels = [ '{}<.<{}'.format(price_range[i-1], s) for i,s in enumerate(price_range) if i != 0]
sizes  = count_price
explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
ax.pie(sizes, explode = explode, labels=labels, colors = colors,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow = False, startangle=0)
ax.axis('equal')
f.text(0.5, 1.01, "number of orders in given price range", ha='center', fontsize = 18);

#It is clear that approx 66% of our order has price greater than 200

#Now, I'll explore description variable to sort our products in different categories
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

is_noun = lambda pos: pos[:2] == 'NN'

def keywords_inventory(dataframe, colonne = 'Description'):
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys   = []
    count_keywords  = dict()
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        
        for t in nouns:
            t = t.lower() ; racine = stemmer.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1                
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    print("No of keywords in variable '{}': {}".format(colonne,len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords

#keyword_inventory takes description as input and find the root node 
#ans maintains the number of times the root occurs in the description of different orders
 
#list of products with roots and its ocuurence
list_of_products = pd.DataFrame(data['Description'].unique()).rename(columns = {0:'Description'})

#Analysis of description of various products

keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(data)

#The execution of this function returns three variables:

#keywords: the list of extracted keywords
#keywords_roots: a dictionary where the keys are the keywords roots and the values are the lists of words associated with those roots
#count_keywords: dictionary listing the number of times every word is used
#At this point, I convert the count_keywords dictionary into a list, to sort the keywords according to their occurences:

list_products = []
for k,v in count_keywords.items():
    list_products.append([keywords_select[k],v])
list_products.sort(key = lambda x:x[1], reverse = True)

#plotting the list_products w.r.t. occurence
lists = sorted(list_products, key = lambda x:x[1], reverse = True)

plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(7, 25))
y_axis = [i[1] for i in lists[:125]]
x_axis = [k for k,i in enumerate(lists[:125])]
x_label = [i[0] for i in lists[:125]]
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 13)
plt.yticks(x_axis, x_label)
plt.xlabel("Nb. of occurences", fontsize = 18, labelpad = 10)
ax.barh(x_axis, y_axis, align = 'center')
ax = plt.gca()
ax.invert_yaxis()

plt.title("Words occurence",bbox={'facecolor':'k', 'pad':5}, color='w',fontsize = 25)
plt.show()

#Now, from list_of_products, i'll discard the anaysis of the word that occur less than 13 times
#and, the colors as it carries less information 

list_products = []
for k,v in count_keywords.items():
    word = keywords_select[k]
    if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
    if len(word) < 3 or v < 13: continue
    if ('+' in word) or ('/' in word): continue
    list_products.append([word, v])
 
list_products.sort(key = lambda x:x[1], reverse = True)
len(list_products)

lists_products = data['Description'].unique()
X = pd.DataFrame()
for key, occurence in list_products:
    X.loc[:, key] = list(map(lambda x:int(key.upper() in x), lists_products))


#matrix indicates the words contained in the description of the 
#products using the one-hot-encoding principle. In practice, 
#I have found that introducing the price range results in more 
#balanced groups in terms of element numbers. Hence, I add 6 extra 
#columns to this matrix, where I indicate the price range of the products:
threshold = [0, 1, 2, 3, 5, 10]
label_col = []
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])
    label_col.append(col)
    X.loc[:, col] = 0

for i, prod in enumerate(lists_products):
    price = df_cleaned[ df_cleaned['Description'] == prod]['UnitPrice'].mean()
    j = 0
    while price > threshold[j]:
        j+=1
        if j == len(threshold): break
    X.loc[i, label_col[j-1]] = 1


print("{:<8} {:<20} \n".format('gamme', 'number of products') + 20*'-')
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])    
    print("{:<10}  {:<20}".format(col, X.loc[:, col].sum()))
    
#creating cluster of products 

#using kmeans method of sklearn(uses Euclidean Distance, can also use kmode(which uses hamming distance, best method in this case))
#I'll use silhoutee score to determine the number of clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection

matrix = X.as_matrix()
for n_clusters in range(3,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    
#I'll choose to seperate the dataset into 5 clusters 
# In order to ensure a good classification at every run of the 
#notebook, I iterate untill we obtain the best possible silhouette 
#score, which is, in the present case, around 0.15:

n_clusters = 5
silhouette_avg = -1
while silhouette_avg < 0.0774:
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    
    #km = kmodes.KModes(n_clusters = n_clusters, init='Huang', n_init=2, verbose=0)
    #clusters = km.fit_predict(matrix)
    #silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    
#Counting the number of elements in every cluster
pd.Series(clusters).value_counts()    

#Plot to represent the silhoutee score of different clusters

def graph_component_silhouette(n_clusters, lim_x, mat_size, sample_silhouette_values, clusters):
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax1.set_xlim([lim_x[0], lim_x[1]])
    ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
    y_lower = 10
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / n_clusters)        
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.8)
        
        ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color = 'red', fontweight = 'bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))
        
        y_lower = y_upper + 10  
        
#word cloud    

#Now, i'll plot word cloud of all 5 clusters to determine which 
#keyword is frequent in them
#This will give us the overall view of all clusters and the type of
#products that these clusters represent

list_prod = pd.DataFrame(lists_products)
lists_words = [word for (word, occurence) in list_products]

occurence = [dict() for _ in range(n_clusters)]

for i in range(n_clusters):
    lists_cluster = list_prod.loc[clusters == i]
    for word in lists_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurence[i][word] = sum(lists_cluster.loc[:, 0].str.contains(word.upper()))      


#Plotting word cloud for all 5 clusters
 
from wordcloud import WordCloud, STOPWORDS       
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)

def make_wordcloud(liste, increment):
    ax1 = fig.add_subplot(4,2,increment)
    words = dict()
    trunc_occurences = liste[0:150]
    for s in trunc_occurences:
        words[s[0]] = s[1]

    wordcloud = WordCloud(width=1000,height=400, background_color='lightgrey', 
                          max_words=1628,relative_scaling=1,
                          color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    plt.title('cluster nº{}'.format(increment-1))

fig = plt.figure(1, figsize=(14,14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
for i in range(n_clusters):
    list_cluster_occurences = occurence[i]

    tone = color[i] 
    liste = []
    for key, value in list_cluster_occurences.items():
        liste.append([key, value])
    liste.sort(key = lambda x:x[1], reverse = True)
    make_wordcloud(liste, i+1)  

#Now, we can see cluster 3 is related to gifts and cluster 4 is related to jewellery
#Similarly, other clusters also represents a group of similar items
#Nevertheless, it can also be observed that many words appear in 
#various clusters and it is therefore difficult to clearly distinguish them.    
    
#Now, to be sure that these clusters are different we'll look at their composition
#Let's first perform the dimensionality reduction (PCA) for easy visualisation
#As the initial matrix contains alot of variables.
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(matrix)
pca_samples = pca.transform(matrix)        

#Amount of variance explained by each component
fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='cumulative explained variance')
sns.barplot(np.arange(1,matrix.shape[1]+1), pca.explained_variance_ratio_, alpha=0.5, color = 'g',
            label='individual explained variance')
plt.xlim(0, 100)

ax.set_xticklabels([s if int(s.get_text())%2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel('Explained variance', fontsize = 14)
plt.xlabel('Principal components', fontsize = 14)
plt.legend(loc='upper left', fontsize = 13)

#We see that the number of components required to explain the data 
#is extremely important: we need more than 100 components to explain
#90% of the variance of the data. In practice, I decide to keep only
#a limited number of components since this decomposition is only 
#performed to visualize the data

#taking the number of component to be 50 
pca = PCA(n_components=50)
matrix_9D = pca.fit_transform(matrix)
mat = pd.DataFrame(matrix_9D)
mat['cluster'] = pd.Series(clusters)

#add another column in initial dataset where cluster of each product is indicated
corresp = dict()
for key, val in zip (lists_products, clusters):
    corresp[key] = val 
df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)

#In a second step, I decide to create the categ_N variables 
#that contains the amount spent in each product category
for i in range(5):
    col = 'categ_{}'.format(i)        
    df_temp = df_cleaned[df_cleaned['categ_product'] == i]
    price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
    price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
    df_cleaned.loc[:, col] = price_temp
    df_cleaned[col].fillna(0, inplace = True)

df_cleaned[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3','categ_4']][:5]

#Up to now, the information related to a single order was split over
#several lines of the dataframe (one line per product). 
#I decide to collect the information related to a particular 
#order and put in in a single entry. I therefore create a new 
#dataframe that contains, for each order, the amount of the basket,
# as well as the way it is distributed over the 5 categories of 
#products

temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice':'Basket Price'})

for i in range(5):
    col = 'categ_{}'.format(i) 
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
    basket_price.loc[:, col] = temp 

df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])

basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID', ascending = True)[:5]

basket_price.head()

#The dataframe basket price contains the data of 12 months 
#Later, one of the objectives will be to develop a model capable of 
#characterizing and anticipating the habits of the customers 
#visiting the site and this, from their first visit. In order to be 
#able to test the model in a realistic way, I split the data set by 
#retaining the first 10 months to develop the model and the 
#following two months to test it

#To print minimum and maximum value of invoice date to know the range of data
print(basket_price['InvoiceDate'].min(), '->',  basket_price['InvoiceDate'].max())

#splitting into training and test data 
#All data with invoice date less than 2011,10,1 is kept under training data
#and remaining data is kept under test data
import datetime

set_train = basket_price[basket_price['InvoiceDate'] < datetime.date(2011,10,1)]
set_test         = basket_price[basket_price['InvoiceDate'] >= datetime.date(2011,10,1)]
basket_price = set_train.copy(deep = True)

#In a second step, I group together the different entries that 
#correspond to the same user. I thus determine the number of 
#purchases made by the user, as well as the minimum, maximum, 
#average amounts and the total amount spent during all the visits

transactions_per_user=basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(['count','min','max','mean','sum'])
for i in range(5):
    col = 'categ_{}'.format(i)
    transactions_per_user.loc[:,col] = basket_price.groupby(by=['CustomerID'])[col].sum() /\
                                            transactions_per_user['sum']*100

transactions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending = True)[:5]

#Finally, I define two additional variables that give the number 
#of days elapsed since the first purchase ( FirstPurchase ) and 
#the number of days since the last purchase ( LastPurchase )

last_date = basket_price['InvoiceDate'].max().date()

first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
last_purchase      = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

test  = first_registration.applymap(lambda x:(last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x:(last_date - x.date()).days)

transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop = False)['InvoiceDate']
transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop = False)['InvoiceDate']

transactions_per_user[:5]

#A customer category of particular interest is that of customers 
#who make only one purchase. One of the objectives may be, 
#for example, to target these customers in order to retain them. 
#In part, I find that this type of customer represents 1/3 of the 
#customers listed

n1 = transactions_per_user[transactions_per_user['count'] == 1].shape[0]
n2 = transactions_per_user.shape[0]
print("% of client with only one transaction: {:<2}/{:<5} ({:<2.2f}%)".format(n1,n2,n1/n2*100))

#The dataframe transactions_per_user contains a summary of all 
#the commands that were made. Each entry in this dataframe 
#corresponds to a particular client. I use this information to 
#characterize the different types of customers and only keep a 
#subset of variables

list_cols = ['count','min','max','mean','categ_0','categ_1','categ_2','categ_3','categ_4']

selected_customers = transactions_per_user.copy(deep = True)
matrix = selected_customers[list_cols].as_matrix()

#To scale and standardize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(matrix)
print('variables mean values: \n' + 90*'-' + '\n' , scaler.mean_)
scaled_matrix = scaler.transform(matrix)

# I will use this base in order to create a representation of the different clusters and thus verify the quality of the separation of the different groups. I therefore perform a PCA beforehand
pca = PCA()
pca.fit(scaled_matrix)
pca_samples = pca.transform(scaled_matrix)

#Plot to represent the amount of variance explained by each of the components
fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='cumulative explained variance')
sns.barplot(np.arange(1,matrix.shape[1]+1), pca.explained_variance_ratio_, alpha=0.5, color = 'g',
            label='individual explained variance')
plt.xlim(0, 10)

ax.set_xticklabels([s if int(s.get_text())%2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel('Explained variance', fontsize = 14)
plt.xlabel('Principal components', fontsize = 14)
plt.legend(loc='best', fontsize = 13)

#At this point, I define clusters of clients from the standardized matrix that was defined earlier and using the k-means algorithm fromscikit-learn. I choose the number of clusters based on the silhouette score and I find that the best score is obtained with 11 clusters

n_clusters = 11
kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=100)
kmeans.fit(scaled_matrix)
clusters_clients = kmeans.predict(scaled_matrix)
silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
print('score de silhouette: {:<.3f}'.format(silhouette_avg))

#Now, I look at the number of customers in each cluster
pd.DataFrame(pd.Series(clusters_clients).value_counts(), columns = ['no. of clients']).T

#There is a certain disparity in the sizes of different groups that have been created. Hence I will now try to understand the content of these clusters in order to validate (or not) this particular separation. At first, I use the result of the PCA
pca = PCA(n_components=6)
matrix_3D = pca.fit_transform(scaled_matrix)
mat = pd.DataFrame(matrix_3D)
mat['cluster'] = pd.Series(clusters_clients)

#Plot to create the representation of various cluster
import matplotlib.patches as mpatches

sns.set_style("white")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

LABEL_COLOR_MAP = {0:'r', 1:'tan', 2:'b', 3:'k', 4:'c', 5:'g', 6:'deeppink', 7:'skyblue', 8:'darkcyan', 9:'orange',
                   10:'yellow', 11:'tomato', 12:'seagreen'}
label_color = [LABEL_COLOR_MAP[l] for l in mat['cluster']]

fig = plt.figure(figsize = (12,10))
increment = 0
for ix in range(6):
    for iy in range(ix+1, 6):   
        increment += 1
        ax = fig.add_subplot(4,3,increment)
        ax.scatter(mat[ix], mat[iy], c= label_color, alpha=0.5) 
        plt.ylabel('PCA {}'.format(iy+1), fontsize = 12)
        plt.xlabel('PCA {}'.format(ix+1), fontsize = 12)
        ax.yaxis.grid(color='lightgray', linestyle=':')
        ax.xaxis.grid(color='lightgray', linestyle=':')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if increment == 12: break
    if increment == 12: break
        

comp_handler = []
for i in range(n_clusters):
    comp_handler.append(mpatches.Patch(color = LABEL_COLOR_MAP[i], label = i))

plt.legend(handles=comp_handler, bbox_to_anchor=(1.1, 0.9), 
           title='Cluster', facecolor = 'lightgrey',
           shadow = True, frameon = True, framealpha = 1,
           fontsize = 13, bbox_transform = plt.gcf().transFigure)

plt.tight_layout()

#From this representation, it can be seen, for example, 
#that the first principal component allow to separate the tiniest 
#clusters from the rest. More generally, we see that there 
#is always a representation in which two clusters will appear to be 
#distinct.

#At this stage, I have verified that the different clusters are 
#indeed disjoint (at least, in a global way). It remains to 
#understand the habits of the customers in each cluster. 
#To do so, I start by adding to the selected_customers dataframe a 
#variable that defines the cluster to which each client belongs
selected_customers.loc[:, 'cluster'] = clusters_clients

#Then, I average the contents of this dataframe by first selecting 
#the different groups of clients. This gives access to, for example,
#the average baskets price, the number of visits or the total sums 
#spent by the clients of the different clusters. I also determine 
#the number of clients in each group (variable size )

merged_df = pd.DataFrame()
for i in range(n_clusters):
    test = pd.DataFrame(selected_customers[selected_customers['cluster'] == i].mean())
    test = test.T.set_index('cluster', drop = True)
    test['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
    merged_df = pd.concat([merged_df, test])

merged_df.drop('CustomerID', axis = 1, inplace = True)
print('number of customers:', merged_df['size'].sum())

merged_df = merged_df.sort_values('sum')

#Finally, I re-organize the content of the dataframe by ordering 
#the different clusters: first, in relation to the amount wpsent 
#in each product category and then, according to the total amount spent

list_index = []
for i in range(5):
    column = 'categ_{}'.format(i)
    list_index.append(merged_df[merged_df[column] > 45].index.values==0)

list_index_reordered = list_index
list_index_reordered += [ s for s in merged_df.index if s not in list_index]

merged_df = merged_df.reindex(index = list_index_reordered)
merged_df = merged_df.reset_index(drop = False)
display(merged_df[['cluster', 'count', 'min', 'max', 'mean', 'sum', 'categ_0',
                   'categ_1', 'categ_2', 'categ_3', 'categ_4', 'size']])
        
#Classification of Customers
#Since the goal is to define the class to which a client belongs and this, as soon as its first visit, I only keep the variables that describe the content of the basket, and do not take into account the variables related to the frequency of visits or variations of the basket price over time
columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4' ]
X = selected_customers[columns]
Y = selected_customers['cluster']

#Training different classification model 
precision={}
accuracy={}
#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)

#Support Vector Classifier

X_train1=X_train
y_train1=Y_train
X_test1=X_test
# Training the SVM model on the Training set
from sklearn.svm import SVC
csvm = SVC(kernel = 'linear', random_state = 0)
csvm.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred1 = csvm.predict(X_test1)

print("Training Accuracy :", csvm.score(X_train1, y_train1))
print("Testing Accuracy :", csvm.score(X_test1, Y_test))

cm1 = confusion_matrix(Y_test, y_pred1)
print(cm1)

print("Precision ",100*metrics.accuracy_score(Y_test,y_pred1))
accuracy['SVM']=csvm.score(X_test1, Y_test)
precision['SVM']=100*metrics.accuracy_score(Y_test,y_pred1)

#FITTING LOGISTIC REGRESSION CLASSIFIER TO OUR TRAINING SET
X_train2=X_train
y_train2=Y_train
X_test2=X_test

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train2, y_train2)

# Predicting the Test set results
y_pred2 = classifier_lr.predict(X_test2)

print("Training Accuracy :", classifier_lr.score(X_train2, y_train2))
print("Testing Accuracy :", classifier_lr.score(X_test2, Y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(Y_test, y_pred2)
print(cm2)


print("Precision ",100*metrics.accuracy_score(Y_test,y_pred2))

accuracy['Logistic regression']=classifier_lr.score(X_test2, Y_test)
precision['Logistic regression']=100*metrics.accuracy_score(Y_test,y_pred2)

#FITTING K_NEAREST_NEIGHBORS MODEL TO OUR TRAINING SET
X_train3=X_train
y_train3=Y_train
X_test3=X_test

from sklearn.neighbors import KNeighborsClassifier
# try to find best k value
scoreList = []
for i in range(1,20):
    knn3 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn3.fit(X_train3, y_train3)
    scoreList.append(knn3.score(X_test3, Y_test))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

# Training the K-NN model on the Training set
cknn = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
cknn.fit(X_train3, y_train3)

# Predicting the Test set results
y_pred3 = cknn.predict(X_test3)

print("Training Accuracy :", cknn.score(X_train3, y_train3))
print("Testing Accuracy :", cknn.score(X_test3, Y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(Y_test, y_pred3)
print(cm3)

print("Precision ",100*metrics.accuracy_score(Y_test,y_pred3))

accuracy['K_NN']=cknn.score(X_test3, Y_test)
precision['K_NN']=100*metrics.accuracy_score(Y_test,y_pred3)

#FITTING DECISION TREE CLASSIFIER 

X_train4=X_train
y_train4=Y_train
X_test4=X_test

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
cdt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
cdt.fit(X_train4, y_train4)

# Predicting the Test set results
y_pred4 = cdt.predict(X_test4)

print("Training Accuracy :", cdt.score(X_train4, y_train4))
print("Testing Accuracy :", cdt.score(X_test4, Y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(Y_test, y_pred4)
print(cm4)

print("Precision ",100*metrics.accuracy_score(Y_test,y_pred4))

accuracy['Decision Trees']=cdt.score(X_test4, Y_test)
precision['Decision Trees']=100*metrics.accuracy_score(Y_test,y_pred4)

# Fitting Random Forest classifier with 100 trees to the Training set
X_train5=X_train
y_train5=Y_train
X_test5=X_test

from sklearn.ensemble import RandomForestClassifier
cl5 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
cl5.fit(X_train5, y_train5)

y_pred5 = classifier.predict(X_test5)

print("Training Accuracy :", cl5.score(X_train5, y_train5))
print("Testing Accuracy :", cl5.score(X_test5, Y_test))

# Making the Confusion Matrix
cm5 = confusion_matrix(Y_test, y_pred5)
print(cm5)

print("Precision ",100*metrics.accuracy_score(Y_test,y_pred5))

accuracy['Random forest']=cl5.score(X_test5, Y_test)
precision['Random Forest']=100*metrics.accuracy_score(Y_test,y_pred5)

#comparing precision of different models
colors = ["purple", "green", "orange", "magenta","yellow"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,4))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Precision %")
plt.xlabel("Algorithms")
sns.barplot(x=list(precision.keys()), y=list(precision.values()), palette=colors)
plt.show()

#comparing accuracy of different models

colors = ["purple", "green", "orange", "magenta","yellow"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,4))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()), palette=colors)
plt.show()

#SVM performs better than other models with an accuracy of 91.55%!!
