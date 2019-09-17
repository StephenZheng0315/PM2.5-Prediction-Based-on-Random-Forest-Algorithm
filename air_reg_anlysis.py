# 4510002988676@hgcxyka
#0xpxtupj

#-*- coding: utf-8 -*-
##导入包
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


'''对数据进行统计分析，查看数据的分布情况'''
data=pd.read_csv('datasets/air_train&test.csv',index_col=0,encoding='gb2312')
print (data.head())
print (data.shape)
index=data.index
col=data.columns
class_names=np.unique(data.iloc[:,-1])
#print (type(data))
print (class_names)
#print (data.describe())


'''划分训练集和验证集'''
data_train, data_test= train_test_split(data,test_size=0.1, random_state=0)
print ("训练集统计描述：\n",data_train.describe().round(2))
print ("验证集统计描述：\n",data_test.describe().round(2))
print ("训练集信息：\n",data_train.iloc[:,-1].value_counts())
print ("验证集信息：\n",data_test.iloc[:,-1].value_counts())


'''查看各变量间的相关系数'''
data.drop([u'质量等级'],axis = 1).corr()

##绘制散点图矩阵
import seaborn as sns
sns.set(style="ticks", color_codes=True);
# 创建自定义颜色调色板
palette = sns.xkcd_palette(['dark blue', 'dark green', 'gold', 'orange'])
# 画散点图矩阵
sns.pairplot(data.drop([u'质量等级'],axis = 1), diag_kind = 'kde', plot_kws=dict(alpha = 0.7))
plt.show()


'''构建随机森林回归模型预测AQI'''
#获取训练集和验证集
X_train=data_train.iloc[:,0:-2]
X_test=data_test.iloc[:,0:-2]
feature=data_train.iloc[:,0:-2].columns
print (feature)
y_train=data_train.iloc[:,-2]
y_test=data_test.iloc[:,-2]
#print (y_test_reg)


'''模型调参'''
##参数选择
from sklearn.model_selection import RandomizedSearchCV
criterion=['mse','mae']
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'criterion':criterion,
                'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#构建模型
clf= RandomForestRegressor()
clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                              n_iter = 10,  
                              cv = 3, verbose=2, random_state=42, n_jobs=1)
#回归
clf_random.fit(X_train, y_train)
print (clf_random.best_params_)


'''模型训练、验证、评估'''
from pyecharts import Bar
rf=RandomForestRegressor(criterion='mse',bootstrap=False,max_features='sqrt', max_depth=20,min_samples_split=10, n_estimators=1200,min_samples_leaf=2)

rf.fit(X_train, y_train) 
y_train_pred=rf.predict(X_train)
y_test_pred=rf.predict(X_test)

#指标重要性
print (rf.feature_importances_)
bar=Bar()
bar.add('指标重要性',feature, rf.feature_importances_.round(2),is_label_show=True,label_text_color='#000')
bar.render('指标重要性.html')

from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score
print ("决策树模型评估--训练集：")
print ('训练r^2:',rf.score(X_train,y_train))
print ('均方差',mean_squared_error(y_train,y_train_pred))
print ('绝对差',mean_absolute_error(y_train,y_train_pred))
print ('解释度',explained_variance_score(y_train,y_train_pred))

print ("决策树模型评估--验证集：")
print ('验证r^2:',rf.score(X_test,y_test))
print ('均方差',mean_squared_error(y_test,y_test_pred))
print ('绝对差',mean_absolute_error(y_test,y_test_pred))
print ('解释度',explained_variance_score(y_test,y_test_pred))

'''预测'''
data_pred=pd.read_csv('datasets/air.csv',index_col=0,encoding='gb2312')
index=data_pred.index
y_pred=rf.predict(data_pred.values)

#将预测结果保存到文件中
result_reg=pd.DataFrame(index)
result_reg['AQI']=y_pred
result_reg.to_csv('datasets/result_reg_city.txt',encoding='gb2312')
print (result_reg)


#可视化预测结果
from pyecharts import Geo
import pandas as pd
df=pd.read_csv('datasets/result_reg_city.txt',index_col=0,encoding='gb2312')
print (df.head())
geo = Geo(
    "全国主要城市空气质量",
    "",
    title_color="#fff",
    title_pos="center",
    width=1200,
    height=600,
    background_color="#404a59",
)
geo.add(
    "",
    df.iloc[:,0],
    df.iloc[:,1],
    visual_range=[0, 300],
    visual_text_color="#111",
    symbol_size=15,
    is_visualmap=True, 
    is_piecewise=True,
    #visual_split_number=6
    pieces=[{"max": 50, "min": 0, "label": "优:0-50"},
            {"max": 100, "min": 50, "label": "良:51-100"},
            {"max": 150, "min": 100, "label": "轻度污染:101-150"},
            {"max": 200, "min": 150, "label": "中度污染:151-200"},
            {"max": 300, "min": 200, "label": "重度污染:201-300"},
            {"max": 1000, "min": 300, "label": "严重污染:>300"},        
        ]
)
geo.render('全国重点城市AQI预测结果的可视化.html')