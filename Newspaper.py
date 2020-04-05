import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
import re 

df=pd.read_csv('Articles.csv')

def formattext(df):
#formats data removes junk words
    lis=df.Article.to_list()    
    article_list=list(map(lambda x:x.lower(),lis))
    
    pattern = '[0-9]+'  
    article_list = [re.sub(pattern, '', i) for i in article_list] 
    
    remove=['.',',',')','(',':','!','$','"','-']
    for i in range(len(article_list)):
        for unwanted in remove:
            article_list[i]=article_list[i].replace(unwanted,'')
        
    return article_list

#get articles and title as a list
articles=formattext(df)
title=df.Heading

#getting a csr matrix for articles/Sparse matrix
tfidf = TfidfVectorizer() 
csr_mat = tfidf.fit_transform(articles)

#nmf modells the data into 6 components
model = NMF(n_components=6)
nmf_features=model.fit_transform(csr_mat)

#normalizing the data
norm_features = normalize(nmf_features)

#creating a dataframe with an index
df = pd.DataFrame(norm_features,index=title)
df.drop_duplicates(inplace=True)

#testing the model
article=df.loc[df.index[123]]
#article=df.loc[df.index[21]]

#checking cosine simmilarity
similarities = df.dot(article)
sim=similarities.nlargest(n=7).reset_index()

#output formatting
print('For the article\n',sim['Heading'][0])
print('\n\nRecomendations are:'.upper())
for n in range(1,len(sim)):
    
    print(n,':',sim['Heading'][n])


