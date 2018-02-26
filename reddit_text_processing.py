import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import os
import datetime as dt
import string
from nltk import pos_tag
from nltk.corpus import wordnet
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix #need this if you want to save tfidf_matrix




data_input_path = 'C:\\Users\\wayne.wong\\PycharmProjects\\red\\data_output'

column_names = ['extract_date', 'subreddit_id', 'subreddit_display_name',
       'submission_id', 'submission_title', 'submission_score',
       'submission_url', 'submission_author', 'submission_date_utc',
       'comment_id', 'comment_body', 'comment_author', 'comment_date_utc']

files = [file for file in os.listdir() if file.endswith(".csv")]

os.chdir(data_input_path) 
df = pd.DataFrame(columns = column_names)
i = 0
for file in files:
    df_temp = pd.read_csv(file, names = column_names, skiprows = 1, header=None, encoding='utf-8')
    df = df.append(df_temp)
    i = i + 1
    print("Completed files %d of %d" %(i, len(files)))  
del df_temp,file,i

df['extract_date'] = pd.to_datetime(df['extract_date'])
df['submission_date_utc'] = pd.to_datetime(df['submission_date_utc'])
df['comment_date_utc'] = pd.to_datetime(df['comment_date_utc'])

''' 
Note:
    - NLTK lemmatization seems to be taking a long time to process
    - NLTK lemmatization seems limited (cars -> car, but not does convert reading -> read)    
'''

subreddit_display_name = 'ethereum'

#Filter for specific subreddit
ethereum_subreddit = df[df['subreddit_display_name'] == subreddit_display_name].reset_index() 
del ethereum_subreddit['index']

#Filter for time window
time_window_length = 9 #days
time_window_max = ethereum_subreddit['submission_date_utc'].max()
time_window_min = time_window_max - dt.timedelta(days = time_window_length)

ethereum_subreddit_windowed = ethereum_subreddit[(ethereum_subreddit['submission_date_utc'] > time_window_min) &
                                                  (ethereum_subreddit['submission_date_utc'] < time_window_max)].reset_index() 
del ethereum_subreddit_windowed['index']

#Derive date-only column for groupby
ethereum_subreddit_windowed['submission_date_only'] = pd.to_datetime(ethereum_subreddit_windowed['submission_date_utc']).dt.date
ethereum_subreddit_windowed['comment_date_only'] = pd.to_datetime(ethereum_subreddit_windowed['comment_date_utc']).dt.date

#Get total number of days 
#time_diff = ethereum_subreddit_windowed['submission_date_utc'].max() - ethereum_subreddit_windowed['submission_date_utc'].min()
#ndays = time_diff.days + (((time_diff.seconds / 60) / 60)/ 24)

#Get mean/median number of submissions per day 
submissions_per_day = ethereum_subreddit_windowed.groupby(['submission_date_only'], as_index=False).agg({'submission_id':'nunique'})
submissions_per_day = submissions_per_day.drop(submissions_per_day.index[0])
submissions_per_day = submissions_per_day.drop(submissions_per_day.index[-1])
mean_submissions_per_day = submissions_per_day['submission_id'].mean()
median_submissions_per_day = submissions_per_day['submission_id'].median()

#Get mean/median number of comments per submission
#Note that this is for the full window period, unlike above where a day before/after the window is removed to ensure fair per-day-mean
submission_comment_count = ethereum_subreddit_windowed.groupby(['submission_id', 'submission_title'], as_index=False).agg({'comment_id':'nunique'})
mean_submission_comment = submission_comment_count['comment_id'].mean()
median_submission_comment = submission_comment_count['comment_id'].median()

#Get mean/median number of engaged-users per day (i.e. make submission/comment at least once) 
submission_authors = ethereum_subreddit_windowed[['submission_date_only', 'submission_author']]
submission_authors.rename(columns={'submission_date_only': 'date', 'submission_author': 'author'}, inplace=True)
comment_authors = ethereum_subreddit_windowed[['comment_date_only', 'comment_author']]
comment_authors.rename(columns={'comment_date_only': 'date', 'comment_author': 'author'}, inplace=True)
all_authors = submission_authors.append(comment_authors, ignore_index=True).drop_duplicates().reset_index(drop=True)
engaged_users_per_day = all_authors.groupby(['date'], as_index=False).agg({'author':'nunique'})
engaged_users_per_day = engaged_users_per_day.drop(engaged_users_per_day.index[0])
engaged_users_per_day = engaged_users_per_day.drop(engaged_users_per_day.index[-1]) 
mean_engaged_users_per_day = engaged_users_per_day['author'].mean()
median_engaged_users_per_day =  engaged_users_per_day['author'].median()

#Number of posts per user (submissions and comments), over the window period
#Note that this is for the full window period, unlike above where a day before/after the window is removed to ensure fair per-day-mean
submissions_per_user = ethereum_subreddit_windowed[['submission_id', 'submission_author']].groupby(['submission_author'], as_index=False).agg({'submission_id':'nunique'})
submissions_per_user.rename(columns={'submission_id': 'post', 'submission_author': 'author'}, inplace=True)
comments_per_user = ethereum_subreddit_windowed[['comment_id', 'comment_author']].groupby(['comment_author'], as_index=False).agg({'comment_id':'nunique'})
comments_per_user.rename(columns={'comment_id': 'post', 'comment_author': 'author'}, inplace=True)
posts_per_user = submissions_per_user.append(comments_per_user, ignore_index=True)
posts_per_user2 = posts_per_user.groupby(['author'], as_index=False).agg({'post':'sum'}) #Adds up the count of submissions and comments
mean_posts_per_user = posts_per_user2['post'].mean()
median_posts_per_user = posts_per_user2['post'].median()

#Get activity of key people (submissions/ comments)
key_people_list = ['vbuterin', 'nickjohnson', 'Souptacular', 'heliumcraft', '5chdn', 'avsa', 'Souptacular']
posts_per_key_user = posts_per_user2[posts_per_user2['author'].isin(key_people_list)].reset_index(drop=True)
mean_posts_per_key_user = posts_per_key_user['post'].mean()
median_posts_per_key_user = posts_per_key_user['post'].median()



engagement_stats_columns = ['subreddit_display_name','time_window_max','time_window_min',
                            'mean_submissions_per_day','median_submissions_per_day',
                            'mean_submission_comment','median_submission_comment', 
                            'mean_engaged_users_per_day', 'median_engaged_users_per_day', 
                            'mean_posts_per_user', 'median_posts_per_user',
                            'mean_posts_per_key_user', 'median_posts_per_key_user']

all_engagement_stats_holding = pd.DataFrame(columns=engagement_stats_columns)

subreddit_engagement_stats = pd.DataFrame({'subreddit_display_name':[subreddit_display_name], 
                                           'time_window_max':[time_window_max],
                                           'time_window_min':[time_window_min],
                                           'mean_submissions_per_day':[mean_submissions_per_day],
                                           'median_submissions_per_day':[median_submissions_per_day],
                                           'mean_submission_comment':[mean_submission_comment],
                                           'median_submission_comment':[median_submission_comment],
                                           'mean_engaged_users_per_day':[mean_engaged_users_per_day],
                                           'median_engaged_users_per_day':[median_engaged_users_per_day],
                                           'mean_posts_per_user':[mean_posts_per_user],
                                           'median_posts_per_user':[median_posts_per_user],
                                           'mean_posts_per_key_user':[mean_posts_per_key_user],
                                           'median_posts_per_key_user':[median_posts_per_key_user]})

    
all_engagement_stats_holding = all_engagement_stats_holding.append(subreddit_engagement_stats, ignore_index=True) 

all_engagement_stats = all_engagement_stats_holding[engagement_stats_columns]


# ------------------- NLP -------------------

#Process submissions
submissions_list = ethereum_subreddit_windowed[['submission_id','submission_title']].drop_duplicates().reset_index(drop=True)
submission_document_list = submissions_list['submission_title'].dropna() #Remove NaN because NLTK needs a sentence

#Process comments
comments_list = ethereum_subreddit_windowed[['comment_id','comment_body']].drop_duplicates().reset_index(drop=True)
comment_document_list = comments_list['comment_body'].dropna() #Remove NaN because NLTK needs a sentence


#help functions for apply

def _tokenize(text):
    tokens = nltk.word_tokenize(text) #convert to tokens
    return(tokens)
    
def _lemmatize(tokens):
    wnl = nltk.WordNetLemmatizer()
    tokens_pos = pos_tag(tokens) ##we need pos info to make lemmatizer work, which requires the original tokens with case information and stopwords
    tokens_lem = [wnl.lemmatize(t[0], pos=t[1].lower()[0]) if t[1].lower().startswith('v') else wnl.lemmatize(t[0]) for t in tokens_pos ] #lemmatize
    return(tokens_lem)

def _lowercase(tokens):
     tokens_low = [x.lower() for x in tokens] #lowercase
     return(tokens_low)
    
def _remove_stopwords(tokens):
    all_stopwords = stopwords.words('english')
    tokens_nostop = [x for x in tokens if x not in all_stopwords] #remove stopwords
    return(tokens_nostop)

def _remove_punc(tokens):
    tokens_nopunc = [x for x in tokens if x not in string.punctuation] #remove punctuation
    return(tokens_nopunc)

def _process_text_all(text):
    tokens = _tokenize(text) #convert to tokens
    tokens = _lemmatize(tokens) #pos tagging and lemmatization
    tokens = _lowercase(tokens) #lowercase
    tokens = _remove_stopwords(tokens) #remove stopwords
    tokens = _remove_punc(tokens) #remove punctuation
    return(tokens)

#Need to be careful about removing stopwords before generating bigrams - might not make sense!
def _process_text_noStopWordRemoval(text):
    tokens = _tokenize(text) #convert to tokens
    tokens = _lemmatize(tokens) #pos tagging and lemmatization
    tokens = _lowercase(tokens) #lowercase
    #tokens = _remove_stopwords(tokens) #remove stopwords
    tokens = _remove_punc(tokens) #remove punctuation
    return(tokens)
    

#Generate unigrams    
submissions_unigrams = list(chain.from_iterable(submission_document_list.apply(_process_text_all).tolist())) #apply _process_all function, and unlist
submissions_unigrams_dist = nltk.FreqDist(submissions_unigrams) #Get distribution of unigrams
submissions_unigrams_dist_top = pd.DataFrame(submissions_unigrams_dist.most_common(20), columns=['ngram', 'frequency'])
#Get the frequency of specific words
submissions_unigrams_dist['price']  

#Generate bigrams
submissions_bigrams = list(nltk.bigrams(submissions_unigrams))
submissions_bigrams_dist = nltk.FreqDist(submissions_bigrams)
submissions_bigrams_dist_top = pd.DataFrame(submissions_bigrams_dist.most_common(20), columns=['raw_ngram', 'frequency'])
submissions_bigrams_dist_top['ngram'] = [' '.join(words) for words in submissions_bigrams_dist_top['raw_ngram']]
submissions_bigrams_dist_top2 = submissions_bigrams_dist_top[['ngram', 'frequency']]

#Generate trigrams
submissions_trigrams = list(nltk.trigrams(submissions_unigrams))
submissions_trigrams_dist = nltk.FreqDist(submissions_trigrams)
submissions_trigrams_dist_top = pd.DataFrame(submissions_trigrams_dist.most_common(20), columns=['raw_ngram', 'frequency'])
submissions_trigrams_dist_top['ngram'] = [' '.join(words) for words in submissions_trigrams_dist_top['raw_ngram']]
submissions_trigrams_dist_top2 = submissions_trigrams_dist_top[['ngram', 'frequency']]


#TF-IDF: Allows us to weight terms based on how important they are to a document. 
#   - Features with High TFIDF: Appear often in particular document, but rarely used across documents
#   - Features with Low TFIDF: Commonly used across all documents OR rarely used and only occur in long documents
corpus = submission_document_list #Feed in raw text
vectorizer = TfidfVectorizer(tokenizer=_process_text_all, min_df=1, ngram_range=(1,3)) #Tokenize, lemmatize etc
vectorized = vectorizer.fit_transform(submission_document_list)
tfidf_raw = vectorized.max(0).toarray()[0]
tfidf_features = np.array(vectorizer.get_feature_names())
tfidf_results = pd.Series(tfidf_raw, index = tfidf_features).sort_values(ascending=False)
tfidf_results2 = pd.DataFrame(tfidf_results).reset_index()
tfidf_results2.rename(columns={'index': 'ngram', 0: 'ti_idf'}, inplace=True)



#TODO: Repeat above for comments



'''
THOUGHTS
- Tried to not remove stopwords, but result was too noisy
- iExec came up top, from bigram results even though I don't recall seeing it 
- Punctuation and "'s" exist in unigram results - should remove
- Is TF-IDF appropriate? i.e. if 'price' is a common term across all documents, we would want to know
- Better way to define document? i.e. a document = submission + comments
'''


#Unsupervised topic modeling





    