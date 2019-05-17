#!/usr/bin/env python
# coding: utf-8

# This project focuses on the FIFA 2019 Dataset collected from Kaggle.com. The purpose here is use Data Science and improve myself in EDA/MLA for data visualization and creating learning models on Jupyter. 

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


datafile = pd.read_csv('fifa19.csv')


# In[3]:


#data.drop('Unnamed: 0',axis=1,inplace=True)
#With some assistance from my brother who has experience with the FIFA games, some of these columns are stats for players
#that indicate how well they can play the game overall in terms of mechanics and skills. The acronyms are all possible 
#positions in soccer and the scores are a respective range for how well each player could play each position. From personal
#experience, my brother states that most managers don't deviate their players from their original roles so for the purposes 
#of this project, we can drop those columns that won't be useful in our ML models.
data.head()


# In[4]:


part1 = data[['Position','Age',
       'Height','Weight','Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes','Overall','Value']]
part1 = part1.dropna(axis=0)


# In[5]:


def w_convert(weight):
    num = weight[0:3]
    out = float(num)
    return out 
def inches(height):
    h = height.split("'")
    feet = float(h[0])*12
    inches = float(h[1])
    return feet + inches 
def dollars(value):
    val = value[1:len(value)-1]
    if value[-1] == 'M':
        res = float(val)*(10**6)
        return res 
    elif value[-1] == 'K':
        res = float(val)*(10**3)
        return res 


# In[6]:


part1['Weight'] = part1['Weight'].apply(w_convert)
part1['Height'] = part1['Height'].apply(inches)
part1['Value'] = part1['Value'].apply(dollars)
part1['Value'] = part1['Value'].apply(np.log10)
df1 = part1
df1.dropna(axis=0,inplace=True)
#Here we will modify the string columns so that their data is in float form 


# In[7]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
#Import the respective libraries from sklearn


# In[8]:


scaler = StandardScaler()
scaler.fit(df1.drop(['Position'],axis=1))
scaled_data = scaler.transform(df1.drop(['Position'],axis=1))


# In[9]:


pca = PCA(n_components=2)
pca.fit(scaled_data)
le = LabelEncoder()
le.fit(df1['Position'])
pos_codes = le.transform(df1['Position'])
#Here we labelencode the data so that we can assign each soccer position to a class for the algorithm 


# In[10]:


x_pca = pca.transform(scaled_data)


# In[11]:


x_pca.shape


# In[12]:


plt.figure(figsize=(12,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=pos_codes,cmap='rainbow')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')


# From the PCA analysis involving 2 principal components, there is a clear distinct position on the right. This class is the Goalkeeper who have the highest stats in all GK positions. Based on the position codes, there are 27 possible positions that include alternative names for the original 11 positions! This already explains the vast amount of uncertainty that arises from the left of the PCA plot!

# In[13]:


df_comp = pd.DataFrame(pca.components_,columns=part1.drop(['Position'],axis=1).columns)
plt.figure(figsize=(20,12))
sns.heatmap(df_comp,cmap='coolwarm')


# To perhaps simplify the algorithm, it might be advised to change some of the positional names to match those of the 
# original 11. I would also like to incorporate a classification that indicates whether the player is a forward, center
# or defender with as few parameters as possible! So given a few vital stats, we can indicate which of the main 3 positions
# a player might actually be!!
# Just playing around with the minimums, we are already seeing that those all belong to Goalkeepers! 

# In[14]:


gk = part1[part1['Position'] == 'GK'][['Height','Weight','Age','GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes','Overall','Value']]
#Let's have some fun with the goalkeepers!


# In[15]:


sns.pairplot(gk)
#From just the pairplots, the most important factor towards a successful goalie is their reflexes which again makes the role
#the most stressful when it comes to being the last line of defense for your team!


# In[16]:


plt.figure(figsize=(12,6))
sns.heatmap(gk.corr(),annot=True)
#Important to note is that even though weight appears to have a weak correlation to the attributes of the player, we must
#keep note that is also the moderate correlation between the height/weight of player that is accounting for the appearance 
#of this info.


# In[17]:


X = gk[['Height','Age','Overall','Weight']].apply(np.log10)
y= gk['Value']
gk.shape
#Here we can run a linear regression model through a log-log transformation


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[19]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print (lm.coef_)


# In[20]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y test')
plt.ylabel('Predicted Y')


# In[21]:


from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test,predictions)
MSE = metrics.mean_squared_error(y_test,predictions)
r2 = metrics.r2_score(y_test,predictions)
RMSE = np.sqrt(MSE)
print('MAE: {}'.format(MAE))
print('MSE: {}'.format(MSE))
print('rsq: {}'.format(r2))
print('RMSE: {}'.format(RMSE))


# In[22]:


residuals = y_test-predictions 
sns.distplot(residuals,bins=30)


# Now that we have ran some regression on the goalie position, we can focus on the other position namely the forwards, centers,
# and defenders. The goal here is to use the attributes to determine which class a player would most probably belong to, figure 
# out the more important parameters for each position!

# In[23]:


df1.shape


# In[24]:


#We will create a function to alter the positions so that we are only involved with the original 11.
df1['Position'].value_counts()


# In[25]:


def original_11(pos):
    if pos in ['CF','LS','RS']:
        return 'ST'
    elif pos in ['LCB','RCB']:
        return 'CB'
    elif pos in ['LCM','RCM']:
        return 'CM'
    elif pos == 'LWB':
        return 'LB'
    elif pos == 'RWB':
        return 'RB'
    elif pos in ['LM','LF']:
        return 'LW'
    elif pos in ['RM','RF']:
        return 'RW'
    elif pos in ['LAM','RAM']:
        return 'CAM'
    elif pos in ['LDM','RDM']:
        return 'CDM'
    else:
        return pos


# In[26]:


df1['Position'] = df1['Position'].apply(original_11)


# In[27]:


df1['Position'].value_counts()
#Since the goalkeeper has separate stats, we can drop this position and those columns and look at the rest of the players!
part2 = df1[df1['Position'] != 'GK']
part2.drop(['GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes'],axis=1,inplace=True)


# In[28]:


from sklearn.svm import SVC
X = part2.drop('Position',axis=1)
y = part2['Position']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[29]:


svc_model = SVC(kernel='linear',C=1,decision_function_shape='ovo')
svc_model.fit(X_train,y_train)


# In[30]:


predictions = svc_model.predict(X_test)


# In[31]:


from sklearn.metrics import confusion_matrix,classification_report


# In[32]:


print(confusion_matrix(y_test,predictions))


# In[33]:


print(classification_report(y_test,predictions))


# In[34]:


error_rate = np.mean(y_test != predictions)
print(error_rate)
#The fact that the error rate for classification is 33% is not the best of news. What we can do from here is two options,
#either reduce the numbers of classes to a more general role in soccer (offense,defense or mid) or running a gridsearch in 
#hope of obtaining much better results. 


# In[35]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)


# In[36]:


grid.best_params_


# In[37]:


grid.best_estimator_


# In[38]:


grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))


# In[39]:


print(classification_report(y_test,grid_predictions))


# Unfortunately, it would be best to run a VIF analysis to solve the issue of multicollinearity and appropriately select 
# the key features. Another option would be to further generalize the positions so that we only have 3 classes to worry about
# however the problem is that some positions are required to be hybrid players; for example a central attacking mid would be
# shifting between offense and mid. Another issue that can be seen is that some positions will be almost the same the 
# difference is left and right which is minimal. So here we will sacrifice accuracy to simplify the noise in the dataset! 

# In[40]:


def general_3(pos):
    if pos in ['LW','RW','ST','CAM']:
        return 'Forward'
    elif pos in ['CDM','CM']:
        return 'Mid'
    else:
        return 'Defender'


# In[41]:


part3 = part2
part3['Role'] = part2['Position'].apply(general_3)


# In[42]:


part3.shape


# In[43]:


X = part3.drop(['Position','Role'],axis=1)
y = part3['Role']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[44]:


svc_model = SVC(kernel='linear',C=1,decision_function_shape='ovo')
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[45]:


scaler = StandardScaler()
scaler.fit(part3.drop(['Position','Role'],axis=1))
scaled_data = scaler.transform(part3.drop(['Position','Role'],axis=1))
pca = PCA(n_components=2)
pca.fit(scaled_data)
le = LabelEncoder()
le.fit(part3['Role'])
pos_codes = le.transform(part3['Role'])
x_pca = pca.transform(scaled_data)
plt.figure(figsize=(12,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=pos_codes,cmap='rainbow')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')


# In[46]:


pca.components_


# In[47]:


df_comp = pd.DataFrame(pca.components_,columns=part3.drop(['Position','Role'],axis=1).columns)
plt.figure(figsize=(20,12))
sns.heatmap(df_comp,cmap='rainbow')


# With the adjustment to 3 general positions, the SVC classifier did a much better job in separating the players in the dataset.
# We can use the heatmap and pick out the more important attributes for feature engineering in hopes of better separation of 
# classes

# In[48]:


feat_eng = part3.drop(['Position','Role','Overall'],axis=1)
X = feat_eng
y = part3['Role']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[49]:


svc_model = SVC(kernel='linear',C=1,decision_function_shape='ovo')
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[50]:


sns.scatterplot('Overall','Value',data=part3)


# In[51]:


#One factor that I did not take into account is the how the distribution of the roles was handled. The formation I chose 
#focused on a 4-2-4 formation however from looking at a coaching manual, there are many formations to take into 
#consideration.This accounted for the extra positions available for each player. Just to ensure that this is the situation, 
#I will modify the classes to be based on the formation of preference! 
def modified_role(pos):
    if pos in ['LB','RB']:
        return 'SB'
    elif pos in ['LW','RW']:
        return 'SW'
    else:
        return pos


# In[52]:


part4 = part2
part4.shape


# In[53]:


part4['Role'] = part4['Position'].apply(modified_role)


# In[54]:


X = part4.drop(['Position','Role'],axis=1)
y = part4['Role']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
svc_model = SVC(kernel='linear',C=1,decision_function_shape='ovo')
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
#Again due to the complexity of the formations, we improved our model slightly but there is still a 20% error rate.


# In[55]:


#This time around I will use the RFC model to see if the issue the entire time was selecting the wrong model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
X = part4.drop(['Position','Role'],axis=1)
y = part4['Role']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print('\n')
print(confusion_matrix(y_test,rfc_pred))


# In[56]:


rfc = RandomForestClassifier(n_estimators=600)
X = part3.drop(['Position','Role'],axis=1)
y = part3['Role']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print('\n')
print(confusion_matrix(y_test,rfc_pred))


# In[57]:


from sklearn.model_selection import cross_val_score
svcscore = SVC(kernel='linear',C=1)
rfcscore = RandomForestClassifier(n_estimators=600)
X = part4.drop(['Position','Role'],axis=1)
y = part4['Role']
print(cross_val_score(svcscore,X,y,cv=5,verbose=2))
print(cross_val_score(rfcscore,X,y,cv=5,verbose=2))


# Creating a classfication system to determine soccer player positions can prove to be very difficult due to similiarities from neighboring classes as well as the number of formations a soccer team is expected to play so the functions created were for the 
# assumption of a 4-2-4 formation that is typically played. This is apparently in how the original 11 function was defined when I assigned the other roles to fit one of the positions. The best overall classifier would have to be the SVC model even though it takes much longer than the RFC model to distinguish between the two models!

# In[58]:


def hist(nat):
    his = {}
    for ele in nat:
        his[ele] = his.get(ele,0) + 1
    return his


# In[59]:


def UK(nat):
    if nat == 'England':
        return 'United Kingdom'
    else:
        return nat


# In[60]:


data['Nationality'] = data['Nationality'].apply(UK)


# In[61]:


nat_list = list(data['Nationality'])
nat_hist = hist(nat_list)
keys = nat_hist.keys()
country = []
count_co = []
for key in keys:
    country.append(key)
    count_co.append(nat_hist[key])


# In[62]:


import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)


# In[63]:


data = dict(type='choropleth',
            colorscale = 'Rainbow',
            locations=country,
            locationmode = 'country names',
           z = count_co,
           text=country,
           colorbar = {'title':'Player Counts'})

layout = dict(
    title = 'Players by Nationality',
    geo = dict(showframe = False, projection = {'type':'mercator'}))


# In[64]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# In[65]:


def topnum_stat(stat,no_players):
    topnum_stat = part3.sort_values(by=stat,ascending=False).head(no_players)
    return sns.countplot(x='Role',data=topnum_stat)


# In[66]:


overall = topnum_stat('Overall',100)


# In[67]:


value = topnum_stat('Value',100)


# In[68]:


Marking = topnum_stat('Finishing',500)


# In[69]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X = part4.drop(['Position','Role'],axis=1)
y = part4['Role']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[70]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[71]:


predictions = dtree.predict(X_test)


# In[72]:


from sklearn.metrics import classification_report,confusion_matrix


# In[73]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[74]:


sns.distplot(part1['Value'],bins=50)


# In[75]:


sns.distplot(part1['Overall'],bins=50)


# In[76]:


sns.jointplot('Overall','Value',part1,kind='kde',color='r')


# In[79]:


top_50 = datafile.sort_values(by='Overall',ascending=False).head(50)
club_50 = hist(list(top_50['Club']))
keys = club_50.keys()
club = []
club_co = []
for key in keys:
    club.append(key)
    club_co.append(club_50[key])
    
labels = club
sizes = club_co
colors = None
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=90,shadow=True)
ax1.axis('equal')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




