#Given a subreddit
#- Get all submissions (date and user)
#- Get all comments (date and user)

#Outstanding Issues
#- Are we scraping all submissions? Or is there a limit? What role does sorting (e.g. by hot) play?
#- Are we scraping all comments? What are the limits?
#- How much data is needed (i.e. how far back?); Iterating through submissions of one subreddit with limit=None yielded 379 submissions and 7114 comments, over the past 10 days; with multiple subreddits, ...
#   - Would need to assess a snapshot, before major development (e.g. closure of bitconnect)
#   - Track how the community has changed over time / how founder engagement has changed overtime (compared with 2014-2016)
#   - Next Steps:
#       - Understand what the true limits are
#       - Figure out how to work with the limits (i.e. use time delay and run overnight)
#       - Figure out how to set up on AWS to run 24/7 if needed
#- What if price talk was in a separate subreddit you are unaware of? Assumption: Only considering the main/official subreddit


import pandas as pd
import praw
import datetime
import os

data_output_path = 'C:\\Users\\wayne.wong\\PycharmProjects\\red\\data_output'

extract_date = datetime.date.today()

#Obtain Reddit instance (Read-only)
reddit = praw.Reddit(client_id='DouuN9u4FsJLQw',
                     client_secret='lJrG9BsFxIDnJNSAmyb3pjzxRJ0',
                     user_agent='python:local:v0.1 (by /u/waynewz90)')

#list: ethereum, ethtrader, bitcoin, litecoin, bitconnect, davorcoin, populous_platform
subreddit_list = ['bitconnect', 'davorcoin', 'populous_platform']





#Get the subreddit object given subreddit id
#list(reddit.info(['t5_2zf9m'])) #Need to prefix with t5_ 


#Obtain Submissions

df = pd.DataFrame()

for subreddit_name in subreddit_list:

    for submission in reddit.subreddit(subreddit_name).hot(limit=None):
        
        #In each submission, get the comments
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            subreddit_id = reddit.subreddit(subreddit_name).id
            subreddit_display_name = reddit.subreddit(subreddit_name).display_name
            submission_id = submission.id
            submission_title = submission.title
            submission_score = submission.score
            submission_url = submission.url
            submission_author = submission.author.name if submission.author is not None else None
            submission_date_utc = datetime.datetime.fromtimestamp(submission.created_utc)
            comment_id = comment.id
            comment_body = comment.body
            comment_author = comment.author.name if comment.author is not None else None
            commend_date_utc = datetime.datetime.fromtimestamp(comment.created_utc)
            
            _row = [extract_date, subreddit_id, subreddit_display_name, submission_id, submission_title, submission_score, submission_url, submission_author, submission_date_utc, comment_id, comment_body, comment_author, commend_date_utc]
            
            df = df.append(pd.Series(_row), ignore_index=True) 

df.rename(columns={0: 'extract_date', 1: 'subreddit_id', 2: 'subreddit_display_name', 
                   3: 'submission_id', 4: 'submission_title', 5: 'submission_score',
                   6: 'submission_url', 7: 'submission_author', 8: 'submission_date_utc', 9: 'comment_id',
                   10: 'comment_body', 11: 'comment_author', 12: 'comment_date_utc'}, inplace=True)     
    
    
os.chdir(data_output_path) 
df.to_csv("reddit_data_ethtrade_bitcoin-half_18Feb2018.csv", sep=',', index=False, encoding = 'utf-8')     
      