import pandas as pd
import quandl,math,time,datetime,pickle
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")   #style to make our graphs look decent

# matplotlib is used for plotting graph
# preprocessing helps in processing before calculation
# cross_validation help in shuffling data especially for statics

df=quandl.get('WIKI/GOOGL')    #'WIKI/GOOGL' is code of dataset in quandl.com
df=df[['Adj. Close','Adj. Open','Adj. High','Adj. Low','Adj. Volume']] #features

df['hl_pct']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100
df['pct_chg']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
df=df[['Adj. Close','hl_pct','pct_chg','Adj. Volume']]

forecast_col='Adj. Close'
df.fillna(-99999,inplace=True)  #this we have done we don't want to lose data
forecast_out=int(math.ceil(0.01*len(df)))   #math.ceil function return float value
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1)) #this is our feature & drop drops label
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test=cross_validation.train_test_split(X, y, test_size=0.2)

#clf= LinearRegression(n_jobs=-1)  #n_jobs is the no. of processor we want to use. -1 means we are using all the processors
##clf = svm.SVR(kernel="linear")    #svm=support vector machine,kernel is the type we want to fit it may be linear,poly,etc.
#clf.fit(X_train, y_train)
#accuracy=clf.score(X_test,y_test)   #accuracy is not always equal to confidence
##print accuracy

#with open('linearregression.pickle','wb') as f:  #saving our trained data
#    pickle.dump(clf, f)

#we are doing pickling because we don't want to train data every time we want 
#to predict as this will be slow


pickle_in = open('linearregression.pickle','rb')  #using the trained data
clf = pickle.load(pickle_in)

forecast_set=clf.predict(X_lately)

#print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
#last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
 next_date = datetime.datetime.fromtimestamp(next_unix)
 next_unix += one_day
 df.loc[next_date]= [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
