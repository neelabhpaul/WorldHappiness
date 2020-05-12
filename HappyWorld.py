# World Happyness report analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from sklearn import preprocessing as prep

data19 = pd.read_csv("2019.csv")
data18 = pd.read_csv("2018.csv")
data17 = pd.read_csv("2017.csv")
data16 = pd.read_csv("2016.csv")
data15 = pd.read_csv("2015.csv")

# Top 10 Happiest countries (yearwise)

data = [data15, data16, data17, data18, data19]


def filter_columns(df):
    df = df.filter(['Country','Happiness Score','GDP per Capita','Life expectancy','Freedom'or'Freedom to make life choices','Generosity','Perceptions of corruption'])
    return df    

year = 2015 # to label yearwise plots
for i in range(len(data)):
    print(data[i].isna().sum())
    plt.figure(figsize=(12,5))
    plt.grid(alpha=0.3)
    data[i] = filter_columns(data[i])
    print(data[i].info())
    x = data[i]['Country'].values[0:10]
    y = data[i]['Happiness Score'].values[0:10]
    plt.ylim(5,10)
    plt.bar(x,y)
    
    print('-------------------- Year:',year,'-------------------------')
    plt.title('Happiness Score Top 10')
    plt.xlabel('Countries')
    plt.ylabel('Score')
    plt.show()
    year = year + 1
    

# Model training 
# To predict Happiness scores for the year 2019
linear_reg = LinearRegression()    
for df in data:
    model_df = df.drop(['Happiness Score','Country'], axis = 1)
    train_x, train_y = model_df, df.loc[:, ('Happiness Score')]
    #linear_reg.fit(train_x, train_y)

atrain_x = train_x.to_numpy()
atrain_y = train_y.to_numpy()

test_y = data[4]['Happiness Score']
test_x = data[4].drop(['Happiness Score','Country'], axis = 1)
linear_reg.fit(atrain_x, atrain_y)
pred_y = linear_reg.predict(test_x)

acc_list = [] 
mse_list = []
for i in range(len(test_y)):
    prsnt = abs((test_y[i] - pred_y[i])/test_y[i])
    prsnt1 = (test_y[i] - pred_y[i])**2
    acc_list.append(prsnt)
    mse_list.append(prsnt1)

pred_y = pd.DataFrame(pred_y)
pred_y = pred_y.rename(columns={0:'Prediction'})
print(data19.filter(['Country','Happiness Score']).merge(pred_y, left_index = True, right_index = True ).head(5))

mse_score = (sum(mse_list))/156
mae_score = (sum(acc_list))/156
print('\n Mean Absolute Percentage Error: ', mae_score*100)
print('\n Mean Squared Error: ', mse_score)
print('\n R squared error: ', r2_score(test_y, pred_y))

plt.figure(figsize=(10,6))
plt.scatter(test_x['Freedom'],test_y)
plt.scatter(test_x['Generosity'],test_y)
plt.scatter(test_x['GDP per Capita'],test_y)
plt.scatter(test_x['Life expectancy'],test_y)
plt.scatter(test_x['Perceptions of corruption'],test_y)

# Feature scaling to applied here...
plt.figure(figsize=(10,6))
sc = prep.StandardScaler()
x_scaler = sc.fit_transform(atrain_x)
x_scaler = pd.DataFrame(x_scaler)
atrain_y = np.reshape(atrain_y,(-1, 1))
y_scaler = sc.fit_transform(atrain_y)


linear_reg.fit(x_scaler, y_scaler.flatten())
spred_y = linear_reg.predict(x_scaler)


spred_y = np.reshape(spred_y, (-1,1))
spred_y = sc.fit_transform(spred_y)
for i in range(len(x_scaler)):
    column = []
    for j in range(len(x_scaler)):
        column.append(x_scaler[i][j])
    plt.scatter(column, test_y)
    if (i == 4): break

#plt.plot(x_scaler,linear_reg)
acc_list = [] 
mse_list = []
for i in range(len(spred_y)):
    prsnt = abs((y_scaler[i] - spred_y[i])/y_scaler[i])
    prsnt1 = (y_scaler[i] - spred_y[i])**2
    acc_list.append(prsnt)
    mse_list.append(prsnt1)
    
mse_score = (sum(mse_list))/156
mae_score = (sum(acc_list))/156
print('\n Mean Absolute Percentage Error: ', mae_score*100)
print('\n Mean Squared Error: ', mse_score)
print('\n R squared error: ', r2_score(y_scaler, spred_y))

coeff = linear_reg.coef_

print('intercept', linear_reg.intercept_)
print('coeff.', coeff)
#plt.plot(x_scaler,np.polyval([coeff, linear_reg.intercept_],x_scaler))
c = linear_reg.intercept_
