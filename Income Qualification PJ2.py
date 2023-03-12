#!/usr/bin/env python
# coding: utf-8

# In[151]:


import pandas as pd
import numpy as np


# In[152]:


tr = pd.read_csv('train1.csv')
te = pd.read_csv('test1.csv')


# In[153]:


tr.shape


# In[154]:


te.shape


# In[155]:


tr.head()


# In[156]:


te.head()


# In[157]:


tr.info()


# In[158]:


tr.dtypes.value_counts()


# In[159]:


te.dtypes.value_counts()


# In[160]:


tr.drop(['Id', 'idhogar'], axis=1, inplace=True)


# In[161]:


tr['dependency'].value_counts()


# In[162]:


#converting obj into numerical
def map(i):
    
    if i=='yes':
        return(float(1))
    elif i=='no':
        return(float(0))
    else:
        return(float(i))


# In[163]:


tr['dependency']=tr['dependency'].apply(map)


# In[164]:


for i in tr.columns:
    a=tr[i].dtype
    if a == 'object':
        print(i)


# In[165]:


tr['edjefe']=tr['edjefe'].apply(map)
tr['edjefa']=tr['edjefa'].apply(map)


# In[166]:


tr.info()


# In[167]:


#variable with zero variance
variable=pd.DataFrame(np.var(tr,0),columns=['variance'])
variable.sort_values(by='variance').head(15)
col=list((variable[variable['variance']==0]).index)


# In[168]:


col


# In[169]:


#from above all values of elimbasu5 is same so there is no variance


# In[170]:


#Check if there are any biases in your dataset

contingency_tab=pd.crosstab(tr['r4t3'],tr['hogar_total'])
Observed_Values=contingency_tab.values


# In[171]:


import scipy.stats


# In[172]:


b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]


# In[173]:


rows=len(contingency_tab.iloc[0:2,0])
columns=len(contingency_tab.iloc[0,0:2])


# In[174]:


df =(rows-1)*(columns-1)


# In[175]:


from scipy.stats import chi2


# In[176]:


chi_sq=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_sq_stat=chi_sq[0]+chi_sq[1]


# In[177]:


alpha=0.05
critical_value=chi2.ppf(q=1-alpha, df=df)


# In[178]:


p_value=1-chi2.cdf(x=chi_sq_stat,df=df)


# In[179]:


alpha


# In[180]:


df


# In[181]:


chi_sq_stat


# In[182]:


critical_value


# In[183]:


p_value


# In[184]:


if chi_sq_stat >= critical_value:
    print('Reject H0')
else:
    print('Accept H0')


# In[185]:


if p_value <= alpha:
    print('Reject H0')
else:
    print('Accept H0')


# In[186]:


contingency_tab=pd.crosstab(tr['tipovivi3'],tr['v2a1'])
Observed_Values=contingency_tab.values

b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]
rows=len(contingency_tab.iloc[0:2,0])
columns=len(contingency_tab.iloc[0,0:2])
df=(rows-1)*(columns-1)
#print("Degree of Freedom:-",df)

chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
#print("chi-square statistic:-",chi_square_statistic)

alpha=0.05
critical_value=chi2.ppf(q=1-alpha,df=df)
#print('critical_value:',critical_value)

p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
#print('p-value:',p_value)


# In[187]:


print('Significance level: ',alpha)
print('Degree of Freedom: ',df)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)


# In[188]:


if chi_sq_stat >= critical_value:
    print('Reject H0')
else:
    print('Accept H0')
    
if p_value <= alpha:
    print('Reject H0')
else:
    print('Accept H0')


# In[ ]:





# In[189]:


contingency_tab=pd.crosstab(tr['v18q'],tr['v18q1'])
Observed_Values=contingency_tab.values

b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]
rows=len(contingency_tab.iloc[0:2,0])
columns=len(contingency_tab.iloc[0,0:2])
df=(rows-1)*(columns-1)
#print("Degree of Freedom:-",df)

chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
#print("chi-square statistic:-",chi_square_statistic)

alpha=0.05
critical_value=chi2.ppf(q=1-alpha,df=df)
#print('critical_value:',critical_value)

p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
#print('p-value:',p_value)


# In[190]:


print('Significance level: ',alpha)
print('Degree of Freedom: ',df)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)


# In[191]:


if chi_sq_stat >= critical_value:
    print('Reject H0')
else:
    print('Accept H0')
    
if p_value <= alpha:
    print('Reject H0')
else:
    print('Accept H0')


# In[192]:


# There is bias in our dataset
tr.drop('r4t3', axis=1, inplace=True)


# In[193]:


##Check if there is a house without a family head
tr.parentesco1.value_counts()


# In[194]:


pd.crosstab(tr['edjefa'],tr['edjefe'])


# In[195]:


##Count how many null values are existing in columns
tr.isna().sum().value_counts()


# In[196]:


tr['Target'].isnull().any()


# In[197]:


float_col=[]
for i in tr.columns:
    a=tr[i].dtype
    if a == 'float64':
        float_col.append(i)
print(float_col)


# In[198]:


tr['v18q1'].value_counts()


# In[199]:


pd.crosstab(tr['tipovivi1'],tr['v2a1'])


# In[200]:


pd.crosstab(tr['v18q1'],tr['v18q'])


# In[201]:


tr['v2a1'].fillna(0,inplace=True)
tr['v18q1'].fillna(0,inplace=True)


# In[202]:


tr.drop(['tipovivi3', 'v18q','rez_esc','elimbasu5'],axis=1,inplace=True)


# In[203]:


tr['meaneduc'].fillna(np.mean(tr['meaneduc']),inplace=True)
tr['SQBmeaned'].fillna(np.mean(tr['SQBmeaned']),inplace=True)


# In[204]:


tr.isna().sum().value_counts()


# In[205]:


int_col=[]
for i in tr.columns:
    a=tr[i].dtype
    if a == 'int64':
        int_col.append(i)
print(int_col)


# In[206]:


tr[int_col].isna().sum().value_counts()


# In[207]:


##Set poverty level of the members and the head of the house within a family.
Poverty_level=tr[tr['v2a1'] !=0]


# In[208]:


Poverty_level.shape


# In[209]:


poverty_level=Poverty_level.groupby('area1')['v2a1'].apply(np.median)
poverty_level


# In[210]:


def povert(x):
    if x<8000:
        return('Below level')
    
    elif x>140000:
        return('Above level')
    elif x<140000:
        return('Below poverty level: Ur-ban ; Above poverty level : Rural ')
   


# In[211]:


c=Poverty_level['v2a1'].apply(povert)


# In[212]:


c.shape


# In[213]:


pd.crosstab(c,Poverty_level['area1'])


# In[214]:


features=tr.drop('Target',axis=1)
target=tr.Target


# In[215]:


features_col = features.columns


# In[216]:


from sklearn.preprocessing import StandardScaler


# In[217]:


SS = StandardScaler()


# In[218]:


features1 = SS.fit_transform(features)
features1 = pd.DataFrame(features1,columns=features_col)


# In[219]:


from sklearn.model_selection import train_test_split


# In[220]:


X_train, X_test, y_train, y_test = train_test_split(features1, target)


# In[221]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[222]:


RF=RandomForestClassifier(random_state=0)
parameters={'n_estimators':[10,50,100,300],'max_depth':[3,5,10,15]}
grid=zip([RF],[parameters])


# In[223]:


best = None

for i, j in grid:
    a=GridSearchCV(i,param_grid=j,cv=3,n_jobs=1)
    a.fit(X_train,y_train)
    if best is None:
        best=a
    elif a.best_score_>best.best_score_:
        best=a
        


# In[224]:


print ("Best CV Score",best.best_score_)
print ("Model Parameters",best.best_params_)
print("Best Estimator",best.best_estimator_)


# In[225]:


RF=best.best_estimator_
Model=RF.fit(X_train,y_train)
pred=Model.predict(X_test)


# In[226]:


print('Score of train data : {}'.format(Model.score(X_train,y_train)))
print('Score of test data : {}'.format(Model.score(X_test,y_test)))


# In[227]:


Important_features=pd.DataFrame(Model.feature_importances_,features_col,columns=['feature_importance'])


# In[228]:


Top50=Important_features.sort_values(by='feature_importance',ascending=False).head(50).index


# In[229]:


Top50


# In[230]:


for i in Top50:
    if i not in features_col:
        print(i)


# In[231]:


Features_Top50=features[Top50]


# In[232]:


X_train,X_test,y_train,y_test=train_test_split(Features_Top50,target,test_size=0.25,stratify=target,random_state=0)


# In[233]:


Model_1=RF.fit(X_train,y_train)
y_pred=Model_1.predict(X_test)


# In[234]:


from sklearn.metrics import confusion_matrix,f1_score,accuracy_score


# In[235]:


confusion_matrix(y_test,y_pred)


# In[236]:


f1_score(y_test,y_pred,average='weighted')


# In[237]:


accuracy_score(y_test,y_pred)


# In[ ]:




