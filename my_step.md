# my_step

[TOC]




## 全局设置

### 常用包

```python
# -*- coding: utf-8 -*-

import datetime
import warnings
warnings.filterwarnings('ignore')  # 忽略警告提示
import datetime
import pandas as pd
pd.set_option('display.max_columns', None)  # 设置查看列不省略
# pd.set_option('display.max_rows', None)  # 设置查看行不省略
import numpy as np
from scipy import stats
from scipy import special
import seaborn as sns
sns.set_style('darkgrid')  # 风格设置
import missingno as msno
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 
plt.rcParams['axes.unicode_minus'] = False 
import xgboost as xgb
import lightgbm as lgb

```

### 合并训练集和测试集

```python
# 合并训练集和测试集，方便对两个数据集一起处理
df = train_df.append(test_df)
```

```python
# 训练集和测试集放在一起，方便构造特征
train_df['is_train'] = 1
test_df['is_train'] = 0
df = pd.concat([train_df, test_df], ignore_index=True)
```

```python
# 将处理后的 df 赋值给 train_df, test_df
train_df = df[df['is_train'] == 1]
train_df.drop(['is_train'], axis=1, inplace=True)
test_df = df[df['is_train'] == 0]
test_df.drop(['is_train'], axis=1, inplace=True)
```

### 去除文本列中数据两端的空白

```python
for _col in df.columns:
    if df[_col].dtype == 'object':
        print(_col)
        df[_col] = df[_col].str.strip()
```




## 数据探索

### 查看形状

```python
df.shape
```

```python
print('train_df shape:', train_df.shape)
print('test_df shape:', test_df.shape)
print('df.shape:', df.shape)
```

### 查看几行

```python
df
df.head()
df.tail()
```

### 查看是否有重复的记录

```python
df.duplicated()

df.duplicated().value_counts()

# 取出不重复的记录
df = df[~df.duplicated()]
```

### 获取数据的描述统计信息

```python
df.describe()
```

```python
def statistics_info(data):
    print('_min', np.min(data))
    print('_max:', np.max(data))
    print('_mean', np.mean(data))
    print('_ptp', np.ptp(data))
    print('_std', np.std(data))
    print('_var', np.var(data))


statistics_info(df['t1'])
```

```python
q1 = df['f1'].quantile(q=0.25)  # 第一四分位数 (Q1)
q2 = df['f1'].quantile(q=0.5)  # 第二四分位数 (Q2)  # 中位数
q3 = df['f1'].quantile(q=0.75)  # 第三四分位数 (Q3)
```

#### 第一四分位数

$$
Q1=(n-1)*0.25+1
$$
#### 标准差（Standard Deviation）

$$
std=\sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i-\mu)^2}
$$

### 查看数据类型、是否有空值

```python
df.info()
```

### 检查每一列

```python
for col_name in df.columns:
    print(col_name)
    col = df.loc[:, col_name].value_counts()
    print('行数：{0}'.format(col.sum()))
    print('数据类型：{0}'.format(df[col_name].dtypes))
    print('内容：\n{0}'.format(col))
    print('-' * 16)
```

### 列数据类型转换

```python
df[['str_col']].astype('float64')
```

```python
df['int64_col'] = df['int64_col'].astype('object')
```

```python
df[['int64_col1', 'int64_col2']] = df[['int64_col1', 'int64_col2']].astype('object')
```

### 强制类型转换 

```python
pd.to_numeric(customerDF['str_col'], errors='coerce')  # errors='coerce' -> 将无效解析设置为NaN

df['date_datetime'] = pd.to_datetime(df['date_str'])
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
```

### 类别型特征的列转为category数据类型

```python
categorical_features = [
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6'
]
for c in categorical_features:
    # if df[c].isnull().any():
    #     df[c] = df[c].fillna('MISSING')
    df[c] = df[c].astype('category')
```

### 根据列的数据类型选择列

有时候需要人为根据实际含义来区分`数字特征`和`类型特征`

```python
df.select_dtypes(exclude='object')

# 数值型特征
numeric_feature_df = df.select_dtypes(include=[np.number])
numeric_feature_df.columns
numeric_feature_list = numeric_feature_df.columns.to_list()
numeric_feature_list.remove('is_train')

# 类别型特征
categorical_feature_df = df.select_dtypes(include=[np.object])
categorical_feature_df.columns
categorical_feature_list = categorical_feature_df.columns.to_list()
categorical_feature_list.remove('ori_id')
# 每个类别的种类数
for c in categorical_feature_list:
    print(f'{c}  {len(df[c].value_counts())}')
# 去除种类过多的类，方便可视化
sub_categorical_feature_list = [c for c in categorical_feature_list if c not in ['f1', 'f2', ]]
```

### 日期时间处理

#### 日期时间拆分（年月日时分秒星期）

```python
tmp_col_name = 'f1'

datetime_col_name = f'{tmp_col_name}_datetime'

df[datetime_col_name] = pd.to_datetime(df[datetime_col_name], format='%Y-%m-%d %H:%M:%S')  # 列数据类型转换

# 日期
df[datetime_col_name + '_date'] = df[datetime_col_name].dt.date.astype('object')  # 年月日
df[datetime_col_name + '_year'] = df[datetime_col_name].dt.year.astype('object')  # 年
df[datetime_col_name + '_month'] = df[datetime_col_name].dt.month.astype('object')  # 月
df[datetime_col_name + '_day'] = df[datetime_col_name].dt.day.astype('object')  # 日

# 时间
df[datetime_col_name + '_time'] = df[datetime_col_name].dt.time.astype('object')  # 时分秒
df[datetime_col_name + '_hour'] = df[datetime_col_name].dt.hour.astype('object')  # 时
df[datetime_col_name + '_minute'] = df[datetime_col_name].dt.minute.astype('object')  # 分
df[datetime_col_name + '_second'] = df[datetime_col_name].dt.second.astype('object')  # 秒

df[datetime_col_name + '_weekday'] = df[datetime_col_name].dt.weekday.astype('object')  # 星期  # 0--6
```

#### 时间差（时间跨度）

```python
# pd.to_datetime(datetime.date.today())-pd.to_datetime(df['datetime_col'])
pd.to_datetime(datetime.datetime.fromisoformat('2022-01-01')) - pd.to_datetime(df['datetime_col'])
# 0        993 days
# 1       1123 days
# 2       1081 days
# 3       1212 days
# 4       1156 days
#            ...   
# 12101   1191 days
# 12102   1226 days
# 12103   1064 days
# 12104    896 days
# 12105   1226 days
# Name: datetime_col, Length: 12106, dtype: timedelta64[ns]
```

```python
(pd.to_datetime(datetime.datetime.fromisoformat('2022-01-01')) - pd.to_datetime(df['datetime_col'])) > datetime.timedelta(days=1000)
# 0        False
# 1         True
# 2         True
# 3         True
# 4         True
#          ...  
# 12101     True
# 12102     True
# 12103     True
# 12104    False
# 12105     True
# Name: datetime_col, Length: 12106, dtype: bool
```

```python
# 字符串转datetime

?datetime.datetime.fromisoformat
# Docstring: string -> datetime from datetime.isoformat() output
# Type:      builtin_function_or_method

datetime.datetime.strptime('2022-01-01','%Y-%m-%d')  # '%Y-%m-%d %H:%M:%S'
# datetime.datetime(2022, 1, 1, 0, 0)

?datetime.datetime.strptime
# Docstring: string, format -> new datetime parsed from a string (like time.strptime()).
# Type:      builtin_function_or_method
```

### tsfresh时间序列特征工程工具

```python
# tsfresh -> 时间序列特征工程工具
```

### 空值统计

```python
pd.isna(df).sum()
df.isna().sum()

pd.notna(df)
df.notna()
```

### missingno空值可视化处理

```python
# import missingno as msno  # pip install missingno
msno.matrix(df, labels=True)  # DataFrame的无效性的矩阵可视化
msno.bar(df)  # 条形图
msno.heatmap(df)  # 空值间的相关性热力图
```

https://blog.csdn.net/Andy_shenzl/article/details/81633356

### 查看 xx_col 列有空值的行

```python
df[df['xx_col'].isnull().values == True]
```

### 排序

```python
df.sort_values(by=['xx_col'], na_position='first')
```

### 对某一列的值进行计数

```python
df['f1'].value_counts()
```

### 条件筛选

```python
mask_f1_list = ['a', 'b', 'c']
# where in
df[df['f1'].isin(mask_f1_list)]
# where not in
df[-df['f1'].isin(mask_f1_list)]
```

### 折线图

可以考虑将t1随日变化的折线图和t1随月变化的折线图画在一张图里。

```python
# 设置画框尺寸
fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(1, 1, 1)

# 使用折线图展示t1随时间的走势
plt.plot(df_day['date'], df['t1'], linewidth=1.3, label='Daily t1')
plt.plot(df_month['date'], df_month['t1'], marker='o', linewidth=1.3, label='Monthly t1')
# plt.grid()
ax.legend()  # 添加图例
ax.set_title('this is title')
```

```python
# lineplot 会将相同x值下的多个y值的统计量（默认为均值）作为折线图中的点
sns.lineplot(x='cat_col', y='num_col', hue='cat_col', data=df)
```

```python
sns.pointplot(x='cat_col', y='num_col', hue='cat_col', data=df)
```

### 散点图

```python
sns.scatterplot(x='num_col', y='num_col', hue='cat_col', data=df)

sns.scatterplot(x='cat_col', y='num_col', hue='cat_col', data=df)
```

```python
sns.stripplot(x='cat_col', y='num_col', hue='cat_col', data=df)
```

```python
sns.swarmplot(x='cat_col', y='num_col', hue='cat_col', data=df)
```

### 柱形图

```python
# 数值型特征t1的分布

plt.hist(df['t1'], orientation='vertical', histtype='bar', color='red')

df['f1'].plot.hist()

sns.distplot(df['f1'])
```

```python
# 类别型特征的每个类别频数可视化(countplot)
def count_plot(x,  **kwargs):
    sns.countplot(x=x)
    x = plt.xticks(rotation=90)


f = pd.melt(df, value_vars=categorical_feature_list)
g = sns.FacetGrid(f, col='variable',  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(count_plot, 'value')
```

```python
# 类别型特征的柱形图可视化  # 每个类的t1值的统计值（默认为平均值）
def bar_plot(x, y, **kwargs):
    # sns.barplot(x='cat_col', y='num_col', hue='cat_col', data=df)
    sns.barplot(x=x, y=y)
    x = plt.xticks(rotation=90)


f = pd.melt(df, id_vars=['t1'], value_vars=categorical_feature_list)
g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, 'value', 't1')
```

### 圆饼图

```python
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
labels = ['t1', 't2', 't3', 't4', 't5', 't6']
sizes = [2, 5, 12, 70, 2, 9]  # 数值
explode = (0, 0, 0, 0.1, 0, 0)
plt.axes(aspect='equal')  # 使得饼图是一个正圆
plt.pie(sizes,
        explode=explode,
        labels=labels,
        autopct='%1.1f%%',
        shadow=False,
        startangle=150)
plt.title('this is title')
plt.show()
```

```python
def pie(x, explode=None, labels=None, colors=None, autopct=None,
        pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=None,
        radius=None, counterclock=True, wedgeprops=None, textprops=None,
        center=(0, 0), frame=False, rotatelabels=False, hold=None, data=None):
    '''
    x       :(每一块)的比例，如果sum(x) > 1会使用sum(x)归一化；
    labels  :(每一块)饼图外侧显示的说明文字；
    explode :(每一块)离开中心距离；
    startangle :起始绘制角度,默认图是从x轴正方向逆时针画起,如设定=90则从y轴正方向画起；
    shadow  :在饼图下面画一个阴影。默认值：False，即不画阴影；
    labeldistance :label标记的绘制位置,相对于半径的比例，默认值为1.1, 如<1则绘制在饼图内侧；
    autopct :控制饼图内百分比设置,可以使用format字符串或者format function
            '%1.1f'指小数点前后位数(没有用空格补齐)；
    pctdistance :类似于labeldistance,指定autopct的位置刻度,默认值为0.6；
    radius  :控制饼图半径，默认值为1；
    counterclock ：指定指针方向；布尔值，可选参数，默认为：True，即逆时针。将值改为False即可改为顺时针。
    wedgeprops ：字典类型，可选参数，默认值：None。参数字典传递给wedge对象用来画一个饼图。例如：wedgeprops={'linewidth':3}设置wedge线宽为3。
    textprops ：设置标签（labels）和比例文字的格式；字典类型，可选参数，默认值为：None。传递给text对象的字典参数。
    center ：浮点类型的列表，可选参数，默认值：(0,0)。图标中心位置。
    frame ：布尔类型，可选参数，默认值：False。如果是true，绘制带有表的轴框架。
    rotatelabels ：布尔类型，可选参数，默认为：False。如果为True，旋转每个label到指定的角度。
    '''
    pass
```

### 箱型图

```python
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(12, 6)

sns.boxplot(data=df['f1'], ax=axes[0])
axes[0].set(xlabel='f1')
sns.boxplot(data=df['f2'], ax=axes[1])
axes[1].set(xlabel='f2')
sns.boxplot(data=df['f3'], ax=axes[2])
axes[2].set(xlabel='f3')
```

```python
fig, axes = plt.subplots(2, 2)
fig.set_size_inches(15, 12)

sns.boxplot(x='f1', y='t1', data=df, orient='v', width=0.6, ax=axes[0, 0])
sns.boxplot(x='f2', y='t1', data=df, orient='v', width=0.6, ax=axes[0, 1])
sns.boxplot(x='f3', y='t1', data=df, orient='v', width=0.6, ax=axes[1, 0])
sns.boxplot(x='f4', y='t1', data=df, orient='v', width=0.6, ax=axes[1, 1])
```

```python
sns.boxplot(x=df['f1'], y=df['t1'])
plt.xticks(rotation=90)  # x轴标签逆时针旋转90°
```

```python
df.boxplot(['f1', 'f2'])
```

```python
# 类别型特征箱形图可视化
# categorical_features = [
#     'f1', 'f2', 'f3', 'f4', 'f5', 'f6'
# ]
# for c in categorical_features:
#     # if df[c].isnull().any():
#     #     df[c] = df[c].fillna('MISSING')
#     df[c] = df[c].astype('category')


def box_plot(x, y, **kwargs):
    # sns.boxplot(x=x, y=y)
    sns.boxenplot(x=x, y=y)
    x = plt.xticks(rotation=90)  # x轴标签逆时针旋转90°


f = pd.melt(df, id_vars=['t1'], value_vars=categorical_feature_list)
g = sns.FacetGrid(f,
                  col='variable',
                  col_wrap=2,
                  sharex=False,
                  sharey=False,
                  size=5)
g.map(box_plot, 'value', 't1')
```

### 小提琴图

```python
# 类别特征的小提琴图可视化
categorical_feature_list
target = 't1'
for c in categorical_feature_list:
    sns.violinplot(x=c, y=target, data=df)
    plt.show()
```

```python
# sns.catplot
```

### `groupby`和`agg`

注意：
使用`agg`的`sum`，`mean`方法会自动填充空值为`0`；使用`agg`的`max`，`min`方法不会自动填充空值（pandas版本：1.0.3）

```python
# 对 df 使用 groupby 和 agg
agged1 = df.groupby(
    ['f1', 'f2'],  # 以f1, f2的值进行分组
).agg(
    {
        't1': 'mean',  # 计算组内行的t1列的平均值
        't2': 'max',
        't3': 'min',
        # 't3': 'sum',  # 将会覆盖对t3列求min的结果
    }
).rename(
    columns={
        't1': 't11',  # 将输出列t1重命名为t11（t1列的输出值是上面指定的mean）
        't2': 't22',
    }
)

# agged2 = ...

fig, axes = plt.subplots(1, 2, sharey=True)  # sharey=True -> x或y轴属性将在所有子图(subplots)中共享
agged1.plot(figsize=(15, 5), title='agged1 title', ax=axes[0])
agged2.plot(figsize=(15, 5), title='agged2 title', ax=axes[1])

agged1.plot.bar(stacked=True, title='this is title')  # 堆叠柱形图
```

```python
# 对df使用groupby, 对series使用agg
df.groupby(
    ['f1', 'f2'],  # 以f1, f2的值进行分组
)['t1'].agg(  # 选取t1列，对t1列进行agg
    [
        'max',  # 计算组内t1列的最大值
        'min',
        'count',
        'min',  # 会输出两列min
        't1_mean': 'mean',
        't1_nunique': 'nunique',  # 输出t1有几个不重复的元素，输出位int数字
        't1_q1': lambda x: np.quantile(x, 0.10),
        't1_q2': lambda x: np.quantile(x, 0.20),
    ]
).rename(
    columns={
        'min': 'min2',  # 将输出列min重命名为min2（两列min都会被重命名）
        'count': 'count2',
    }
)
```

### `groupby`和`count`进行统计

```python
# 用f1的值进行分组，统计组内 t1, t2, t3 的记录条数(不统计NaN)
df.groupby(['f1'])[['t1', 't2', 't3']].count()
```

### 密度分布（数值型特征）

- distplot
    distribution+plot
    核密度估计图（kde，kernel density estimation） + 直方图（histogram）

- kdeplot
    可以画双变量核密度估计图

```python
fig, axes = plt.subplots(2, 2)
fig.set_size_inches(12, 10)

sns.distplot(df['t1'], ax=axes[0, 0])
sns.distplot(df['t2'], ax=axes[0, 1])
sns.distplot(df['t3'], ax=axes[1, 0])
sns.distplot(df['t4'], ax=axes[1, 1])

axes[0, 0].set(xlabel='t1', title='Distribution of t1', )
axes[0, 1].set(xlabel='t2', title='Distribution of t2')
axes[1, 0].set(xlabel='t3', title='Distribution of t3')
axes[1, 1].set(xlabel='t4', title='Distribution of t4')
```

### 对分布进行拟合

```python
import scipy.stats as st
y = df['t1']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
```

### 偏度和峰度

数据的偏度和峰度——df.skew()、df.kurt() https://www.cnblogs.com/wyy1480/p/10474046.html

```python
# 正态分布的偏度和峰度都为零
df['t1'].skew()  # 偏度（Skewness）  # Skewness>0 -> 正偏
df['t1'].kurt()  # 峰度（Kurtosis）  # Kurtosis>0 -> 尖顶峰

df.skew()
df.kurt()
```

```python
sns.distplot(df.skew(), color='blue', axlabel='Skewness')
sns.distplot(df.kurt(), color='orange', axlabel='Kurtness')
```

```python
# 查看数值型特征的 偏度和峰值
# numeric_feature_list = df.select_dtypes(include=[np.number])
for col in numeric_feature_list:
    print('{:15}'.format(col),
          'Skewness: {:05.2f}'.format(df[col].skew()), '   ',
          'Kurtosis: {:06.2f}'.format(df[col].kurt()))
```

### FacetGrid

sns.FacetGrid() https://blog.csdn.net/weixin_42398658/article/details/82960379

```python
# 相当于用f1进行分组，在每个组里画t1的分布图
g2 = sns.FacetGrid(df, col="f1", col_wrap=4, sharex=False, sharey=False)
g2.map(sns.distplot, 't1')
```

```python
# 相当于用f1进行分组，在每个组里画散点图（横坐标为f2，纵坐标为t1）
g2 = sns.FacetGrid(train_df, col='f1', col_wrap=4, sharex=False, sharey=False)
g2.map(plt.scatter, 'f2', 't1', alpha=0.3)
```

```python
# 每个数字型特征的分布可视化
# numeric_feature_list = df.select_dtypes(include=[np.number])
f = pd.melt(df, value_vars=numeric_feature_list)  # 将“宽表”变成“长表”
g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')
```

```python
# 相当于用f1进行分组，在每个组里画散点图（f2值不同的点用不同的颜色，横坐标为f3，纵坐标为t1）
g2 = sns.FacetGrid(df, col='f1', hue='f2', col_wrap=4, sharex=False, sharey=False)
g2.map(plt.scatter, 'f3', 't1', alpha=0.3)
g2.add_legend()  # 添加图例
```

```python
# 用f1，f2进行分组
g2 = sns.FacetGrid(df, row='f1', col='f2', hue='f3', sharex=False, sharey=False)
g2.map(plt.scatter, 'f4', 't1', alpha=0.3)
g2.add_legend()
```

### regplot

```python
# 散点图+线性回归
fig, ((ax1, ax2), (ax3, ax4), ) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
sns.regplot(x='f1',
            y='t1',
            data=train_df,
            scatter=True,
            fit_reg=True,
            ax=ax1)
sns.regplot(x='f2',
            y='t1',
            data=train_df,
            scatter=True,
            fit_reg=True,
            ax=ax2)
sns.regplot(x='f3',
            y='t1',
            data=train_df,
            scatter=True,
            fit_reg=True,
            ax=ax3)
sns.regplot(x='f4',
            y='t1',
            data=train_df,
            scatter=True,
            fit_reg=True,
            ax=ax4)
```

```python
def my_sns_regplot(data_df, numeric_feature_list: list, y_label: str):
    """
    数值型特征的 散点图+线性回归
    :param data_df:
    :param numeric_feature_list:
    :param y_label:
    :return:
    """
    for nf in numeric_feature_list:
        print(nf)
        sns.regplot(x=nf, y=y_label, data=df)
        plt.show()

my_sns_regplot(df, numeric_feature_list, 'target')
```

```python
# sns.lmplot
```

### 数值型特征间两两相关性矩阵热力图

```python
corr = df.corr()
plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, vmax=1, cmap='YlGnBu')
```

```python
# sns.clustermap
```

### 目标与各个数值型特征的相关性大小

```python
plt.figure(figsize=(8, 6))
df.corr()['target'].sort_values(ascending=False).plot(kind='bar')  # 降序
# np.abs(df.corr()['target']).sort_values(ascending=False).plot(kind='bar')  # 取绝对值
plt.show()
```

### 选k个和 target 的相关系数最高的数值型特征

```python
k = 10
df.corr().nlargest(k, 'target')['target'].index
```

```python
# 取绝对值
k = 10
np.abs(df.corr()['target']).nlargest(k).index
```

### pairplot

sns.pairplot() https://www.jianshu.com/p/6e18d21a4cad

类似的方法有`jointplot`

```python
# 特征间两两相关性图（数值型特征，类别型特征）
feature_list = ['f1', 'f2', 'f3']
sns.pairplot(df[feature_list], plot_kws={'alpha': 0.1})
```

```python
# 特征与目标相关性图
sns.pairplot(df, x_vars=['f1', 'f2', 'f3', 'f4', 'f5', 'f6'],  # df.columns.tolist()
             y_vars=['t1', 't2', 't3'], plot_kws={'alpha': 0.1})
```

### pandas_profiling生成数据报告

```python
import pandas_profiling
pfr = pandas_profiling.ProfileReport(train_df)
pfr.to_file('example.html')
```




## 特征工程

1. 异常处理：

   - 通过箱线图（或 3-Sigma）分析删除异常值；
   - BOX-COX 转换（处理有偏分布）；
   - 长尾截断；

2. 特征归一化/标准化：

   - 标准化（转换为标准正态分布）；

   - 归一化（转换到 [0,1] 区间）；

   - 针对幂律分布，可以采用公式：
$$
log(\frac{1+x}{1+median})
$$

3. 数据分桶：

   - 等频分桶；
   - 等距分桶；
   - Best-KS 分桶（类似利用基尼指数进行二分类）；
   - 卡方分桶；

4. 缺失值处理：

   - 不处理（针对类似 XGBoost 等树模型）；
   - 删除（缺失数据太多）；
   - 插值补全，包括均值/中位数/众数/建模预测/多重插补/压缩感知补全/矩阵补全等；
   - 分箱，缺失值一个箱；

5. 特征构造：

   - 构造统计量特征，报告计数、求和、比例、标准差等；
   - 时间特征，包括相对时间和绝对时间，节假日，双休日，距离月末/月初的时间，距离最近节假日的时间，时间间隔（出产日期-出售日期）等；
   - 地理信息，包括分箱，分布编码等方法；
   - 非线性变换，包括 log/平方/根号等；
   - 特征组合，特征交叉；
   - 仁者见仁，智者见智。

6. 特征筛选

   - 过滤式（filter）：先对数据进行特征选择，然后在训练学习器，常见的方法有 Relief/方差选择发/相关系数法/卡方检验法/互信息法；
   - 包裹式（wrapper）：直接把最终将要使用的学习器的性能作为特征子集的评价准则，常见方法有 LVM（Las Vegas Wrapper） ；
   - 嵌入式（embedding）：结合过滤式和包裹式，学习器训练过程中自动进行了特征选择，常见的有 lasso 回归；

7. 降维

   - PCA/ LDA/ ICA；
   - 特征选择也是一种降维

### 数据清洗的“完全合一”规则

1. 完整性：单条数据是否存在空值，统计的字段是否完善。
2. 全面性：观察某一列的全部数值，通过常识来判断该列是否有问题，比如：数据定义、单位标识、数据本身。
3. 合法性：数据的类型、内容、大小的合法性。比如数据中是否存在非ASCII字符，性别存在了未知，年龄超过了150等。
4. 唯一性：数据是否存在重复记录，因为数据通常来自不同渠道的汇总，重复的情况是常见的。行数据、列数据都需要是唯一的。

### 空值nan的处理规则

1. 填充

   nan存在的个数如果很小一般选择填充。

   1. 用 均值/中位数/众数 填充
   2. 训练机器学习模型进行空值填充
      使用无空值的行作为训练集，训练模型，预测有空值的行的空值。

2. 删除

   如果nan存在的过多、可以考虑删掉。

3. 分箱/分桶

   缺失值一个箱。

4. 让模型自动处理

   如果使用lgb等树模型可以直接空缺，让树自己去优化。

### 选取没有空值的行

```python
mask = pd.notna(df['t1'])
df[mask]
```

### dropna删除有空值的行

```python
df.dropna(axis=0, how='any', inplace=True)
```

### 删除 df 的行或列

```python
# 删除行
df.drop([1, 3], axis=0, inplace=False)  # 删除index值为1和3的两行
```

```python
# 删除列
df.drop(['f1'], axis=1, inplace=True)  # 删除f1列
del df['f2']  # 删除f2列
df_f3 = df.pop('f3')  # 删除df的f3列，f3列的值返回给df_f3
```

### 替换

[df_replace](https://github.com/MrCat9/Pandas_Note/blob/master/df_replace/df_replace.ipynb)

### 把 col_a 中的空值用 col_b 的值替换（df的修改操作）

```python
df.loc[:, 'col_a'].replace(to_replace=np.nan, value=df.loc[:, 'col_b'], inplace=True)
```

### 空值填充

```python
# DataFrame.interpolate
# interpolate默认线性填充
# bfill为向前填充，ffill为向后填充
# 下行代码实现中间空值取上下平均值填充；头部空值取后面非空值向前填充；尾部空值取前面非空值向后填充
df[feature_col_list] = df[feature_col_list].interpolate().bfill()
```

```python
# sklearn.impute
```

### 删除异常值

```python
# 去掉3个标准差以外数据  # 刪除整行
mask = np.abs(df['t1'] - df['t1'].mean()) <= (3 * df['t1'].std())
df2 = df[mask]
df3 = df[~mask]
```

```python
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot(scale=3)进行清洗
    可以删除整行，也可以将异常值替换成NaN
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """
    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr  # 下限
        val_up = data_ser.quantile(0.75) + iqr  # 上限
        rule_low = (data_ser < val_low)  # 过小
        rule_up = (data_ser > val_up)  # 过大
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name].copy()
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = data_series.index[rule[0] | rule[1]]  # 异常值的索引
    print("Outliers number is: {}".format(len(index)))
    # data_n = data_n.drop(index)  # 删除异常值
    # data_n.reset_index(drop=True, inplace=True)
    data_n.loc[index, col_name] = np.nan  # 将异常值替换为空
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = data_series.index[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(outliers.describe())
    index_up = data_series.index[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(outliers.describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    # sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    # sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    sns.boxenplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxenplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    
    return data_n


train_df = outliers_proc(train_df, 'f1', scale=3)
```

### 不是正态分布

数据波动大的话容易产生过拟合，所以对数据进行变换，使得数据相对稳定。

许多回归模型默认假设`target`服从正态分布

#### 对数变换

```python
# 对数变换
train_y = df['t1']
sns.distplot(train_y)
plt.show()

train_y_log = np.log(train_y)  # train_y要大于0
sns.distplot(train_y_log)
plt.show()
```

```python
# 先取对数，再做归一化
df['f1'].plot.hist()

df['f1'] = np.log(df['f1'] + 1)    # 取对数  # ln(0+1)=0
df['f1'] = ((df['f1'] - np.min(df['f1'])) / (np.max(df['f1']) - np.min(df['f1'])))  # 归一化
df['f1'].plot.hist()
```

```python
def my_log(_df, _c: str):
    # 处理不是正态分布
    # 使用 log 变换

    new_c = _c + '_log'
    
    # ---------------- 变换前 ----------------
    fig = plt.figure(figsize=(15, 5))
    
    # pic1
    plt.subplot(1, 2, 1)
    sns.distplot(_df[_c], fit=stats.norm)
    (mu, sigma) = stats.norm.fit(_df[_c])
    plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    
    # pic2
    plt.subplot(1, 2, 2)
    res = stats.probplot(_df[_c], plot=plt)  # qq图
    
    plt.suptitle('Before')
    
    print(f"Skewness of {_c}: {_df[_c].skew()}")
    print(f"Kurtosis of {_c}: {_df[_c].kurt()}")
    
    # ---------------- 进行 log 变换 ----------------
    _df[new_c] = np.log(_df[_c] + 1)
    
    fig = plt.figure(figsize=(15, 5))
    
    # pic1
    plt.subplot(1, 2, 1)
    sns.distplot(_df[new_c], fit=stats.norm)
    (mu, sigma) = stats.norm.fit(_df[new_c])
    plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    
    # pic2
    plt.subplot(1, 2, 2)
    res = stats.probplot(_df[new_c], plot=plt)
    
    plt.suptitle('After')
    
    print(f"Skewness of {new_c}: {_df[new_c].skew()}")
    print(f"Kurtosis of {new_c}: {_df[new_c].kurt()}")
    
    return _df

df = my_log(df, 'f1')
```

```python
# 正变换
# y=log(x+1)
x1 = 3600.0
y1 = np.log(x1 + 1)  # 8.188966863648876

# 逆变换
# y=exp(x)-1
x2 = 8.188966863648876
y2 = np.exp(x2) - 1  # 3600.000000000002
```

#### BOX-COX

https://blog.csdn.net/Jim_Sun_Jing/article/details/100665967

```python
# -*- coding: utf-8 -*-
from scipy import stats
from scipy import special
```

```python
# 变换前

fig = plt.figure(figsize=(15, 5))

# pic1
plt.subplot(1, 2, 1)
sns.distplot(train_df['target'], fit=stats.norm)
(mu, sigma) = stats.norm.fit(train_df['target'])
plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

# pic2
plt.subplot(1, 2, 2)
res = stats.probplot(train_df['target'], plot=plt)  # qq图

plt.suptitle('Before')

print(f"Skewness of target: {train_df['target'].skew()}")
print(f"Kurtosis of target: {train_df['target'].kurt()}")
```

```python
# 进行Box-Cox变换
# scipy.stats.boxcox
train_df['target_boxcox'], boxcox_lambda = stats.boxcox(train_df['target'])
print('boxcox_lambda:', boxcox_lambda)

fig = plt.figure(figsize=(15, 5))

# pic1
plt.subplot(1, 2, 1)
sns.distplot(train_df['target_boxcox'], fit=stats.norm)
(mu, sigma) = stats.norm.fit(train_df['target_boxcox'])
plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

# pic2
plt.subplot(1, 2, 2)
res = stats.probplot(train_df['target_boxcox'], plot=plt)

plt.suptitle('After')

print(f"Skewness of target: {train_df['target_boxcox'].skew()}")
print(f"Kurtosis of target: {train_df['target_boxcox'].kurt()}")
```

```python
# 进行Box-Cox变换
# scipy.special.boxcox1p
boxcox_lambda = stats.boxcox_normmax(train_df['target'] + 1)  # 寻找最佳变换参数λ
print('boxcox_lambda:', boxcox_lambda)
train_df['target_boxcox'] = special.boxcox1p(train_df['target'], boxcox_lambda)

fig = plt.figure(figsize=(15, 5))

# pic1
plt.subplot(1, 2, 1)
sns.distplot(train_df['target_boxcox'], fit=stats.norm)
(mu, sigma) = stats.norm.fit(train_df['target_boxcox'])
plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

# pic2
plt.subplot(1, 2, 2)
res = stats.probplot(train_df['target_boxcox'], plot=plt)

plt.suptitle('After')

print(f"Skewness of target_boxcox: {train_df['target_boxcox'].skew()}")
print(f"Kurtosis of target_boxcox: {train_df['target_boxcox'].kurt()}")
```

```python
def my_boxcox(_df, _c: str):
    # 处理不是正态分布
    # 使用 BOX-COX 变换

    new_c = _c + '_boxcox'

    # ---------------- 变换前 ----------------
    fig = plt.figure(figsize=(15, 5))

    # pic1
    plt.subplot(1, 2, 1)
    sns.distplot(_df[_c], fit=stats.norm)
    (mu, sigma) = stats.norm.fit(_df[_c])
    plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')

    # pic2
    plt.subplot(1, 2, 2)
    res = stats.probplot(_df[_c], plot=plt)  # qq图

    plt.suptitle('Before')

    print(f"Skewness of {_c}: {_df[_c].skew()}")
    print(f"Kurtosis of {_c}: {_df[_c].kurt()}")

    # ---------------- 进行Box-Cox变换 ----------------
    # scipy.special.boxcox1p
    boxcox_lambda = stats.boxcox_normmax(_df[_c] + 1)  # 寻找最佳变换参数λ
    print(_c, 'boxcox_lambda:', boxcox_lambda)
    _df[new_c] = special.boxcox1p(_df[_c], boxcox_lambda)

    fig = plt.figure(figsize=(15, 5))

    # pic1
    plt.subplot(1, 2, 1)
    sns.distplot(_df[new_c], fit=stats.norm)
    (mu, sigma) = stats.norm.fit(_df[new_c])
    plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')

    # pic2
    plt.subplot(1, 2, 2)
    res = stats.probplot(_df[new_c], plot=plt)

    plt.suptitle('After')

    print(f"Skewness of {new_c}: {_df[new_c].skew()}")
    print(f"Kurtosis of {new_c}: {_df[new_c].kurt()}")

    return _df

df = my_boxcox(df, 'f1')


# # 将处理后的 df 赋值给 train_df, test_df
# train_df = df[df['is_train'] == 1]
# train_df.drop(['is_train'], axis=1, inplace=True)
# test_df = df[df['is_train'] == 0]
# test_df.drop(['is_train'], axis=1, inplace=True)
# 
# # 对 f1 使用 BOX-COX 变换
# train_df = my_boxcox(train_df, 'f1')
# 
# # 训练集和测试集放在一起，方便构造特征
# train_df['is_train'] = 1
# test_df['is_train'] = 0
# df = pd.concat([train_df, test_df], ignore_index=True)
```

```python
# 正变换
# 进行Box-Cox变换
# scipy.special.boxcox1p
boxcox_lambda = stats.boxcox_normmax(train_df['target'] + 1)  # 寻找最佳变换参数λ
print('boxcox_lambda:', boxcox_lambda)  # 0.08127698855857035
x1 = 3600.0
y1 = special.boxcox1p(x1, boxcox_lambda)  # 11.634388741942136

# 逆变换
# special.inv_boxcox1p
x2 = 11.634388741942136
y2 = special.inv_boxcox1p(x2, boxcox_lambda)  # 3600.000000000002
```

### 特征构造

1. 位移求导 = 速度
2. 速度求导 = 加速度
3. 结束时间 - 开始时间 = 持续时间
4. 时间 -> 绝对时间，相对时间，节假日，周末，距离月末/月初的时间，距离最近节假日的时间，时间间隔（出产日期-出售日期）等
5. 邮编 -> 省市区县
6. groupby + merge -> 品牌/地区/… 的 最大值`.max()`/最小值`.min()`/和`.sum()`/均值`.mean()`/中位数`.median()`/众数`.mode()`/标准差`.std()`/…
7. 数据分桶/数据分箱

```python
# 数据分桶
# 为什么要做数据分桶
# 1. 离散后稀疏向量内积乘法运算速度更快，计算结果也方便存储，容易扩展；
# 2. 离散后的特征对异常值更具鲁棒性，如 age>30 为 1 否则为 0，对于年龄为 200 的也不会对模型造成很大的干扰；
# 3. LR 属于广义线性模型，表达能力有限，经过离散化后，每个变量有单独的权重，这相当于引入了非线性，能够提升模型的表达能力，加大拟合；
# 4. 离散后特征可以进行特征交叉，提升表达能力，由 M+N 个变量变成 M*N 个变量，进一步引入非线形，提升了表达能力；
# 5. 特征离散后模型更稳定，如用户年龄区间，不会因为用户年龄长了一岁就变化

# 当然还有很多原因，LightGBM 在改进 XGBoost 时就增加了数据分桶，增强了模型的泛化性

# bin_list = [i * 10 for i in range(21)]  # 0--200
bin_list = list(range(0, 201, 10))  # 0--200
bin_list = [-np.inf] + bin_list + [np.inf]
df['f1_bin'] = pd.cut(df['f1'], bin_list, labels=False).astype('object')  # 左开右闭 区间  # 0不会被分到第一个箱中，200会被分到最后一个箱中
df[['f1', 'f1_bin']].head()
```

```python
def my_groupby_agg_merge(data_df: pd.DataFrame, groupby_list: list, target_list: list) -> pd.DataFrame:
    """
    groupby + agg + merge
    :param data_df: 
    :param groupby_list: 类别型特征
    :param target_list: 数值型特征
    :return: 
    """
    for gc in groupby_list:
        for tc in target_list:
            tmp_agged = df.groupby(
                [gc],
            )[tc].agg(  # 选取t1列，对t1列进行agg
                [
                    'count',
                    'max',  # 计算组内t1列的最大值
                    'min',
                    'sum',
                    'mean',
                    'median',
                    'std',
                ]
            ).rename(
                columns={
                    'count': f'{gc}_{tc}_count',
                    'max': f'{gc}_{tc}_max',
                    'min': f'{gc}_{tc}_min',
                    'sum': f'{gc}_{tc}_sum',
                    'mean': f'{gc}_{tc}_mean',
                    'median': f'{gc}_{tc}_median',
                    'std': f'{gc}_{tc}_std',
                }
            )

            tmp_agged = tmp_agged.reset_index()

            data_df = data_df.merge(tmp_agged, how='left', on=gc)
            
    return data_df


groupby_list = ['f1', 'f2', 'f3', 'f4']
target_list = ['t1', 't2']

df = my_groupby_agg_merge(df, groupby_list, target_list)
df
```

### 特征编码

[类别变量编码](https://github.com/MrCat9/Data_Analysis_Note/blob/master/category_encoding/category_encoding.ipynb)

> 标签编码（Label Encoding）、独热编码（One-Hot Encoding）、序号编码（Ordinal Encoding）、二进制编码（Binary Encoding）、......

```python
feature_col_name_list = ['f1', 'f2', 'f3', 'f4']
for col in feature_col_name_list:
    if df[col].nunique() == 2:
        df[col] = pd.factorize(df[col])[0]  # 标签编码（Label Encoding）
    else:
        df = pd.get_dummies(df, columns=[col])  # 独热编码（One-Hot Encoding）
```

### 减少df内存占用

```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1e6 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        col_type_str = str(col_type)
        
        # if col_type != 'object':
        # if col_type == np.number:
        if col_type_str.startswith('int') or col_type_str.startswith('float'):
            print(f'number col: {col}, col_type: {col_type}')
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type == 'object':
            print(f'object col: {col}, col_type: {col_type}')
            df[col] = df[col].astype('category')
        else:
            print(f'unprocess col: {col}, col_type: {col_type}')

    end_mem = df.memory_usage().sum() / 1e6 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
```

### 特征筛选

- 过滤式（filter）：先对数据进行特征选择，然后在训练学习器，常见的方法有 Relief/方差选择发/相关系数法/卡方检验法/互信息法；
- 包裹式（wrapper）：直接把最终将要使用的学习器的性能作为特征子集的评价准则，常见方法有 LVM（Las Vegas Wrapper） ；
- 嵌入式（embedding）：结合过滤式和包裹式，学习器训练过程中自动进行了特征选择，常见的有 lasso 回归；

1. 相关性`df.corr()`
2. mlxtend`mlxtend.feature_selection`
3. 

### 降维

- PCA/ LDA/ ICA

```python
from sklearn.decomposition import PCA, SparsePCA, FastICA, FactorAnalysis
```




## 模型

### L1和L2正则化

https://blog.csdn.net/jinping_shi/article/details/52433975

```
L1正则化和L2正则化可以看做是损失函数的惩罚项。所谓『惩罚』是指对损失函数中的某些参数做一些限制。对于线性回归模型，使用L1正则化的模型建叫做Lasso回归，使用L2正则化的模型叫做Ridge回归（岭回归）。
```

```
L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择。一定程度上，L1也可以防止过拟合。
L2正则化可以防止模型过拟合（overfitting）。
```

```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

model = LinearRegression().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
plt.figure(figsize=(16, 8))
sns.barplot(abs(model.coef_), continuous_feature_names_list)

model = Lasso().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
plt.figure(figsize=(16, 8))
sns.barplot(abs(model.coef_), continuous_feature_names_list)

model = Ridge().fit(train_X, train_y_ln)
print('intercept:' + str(model.intercept_))
plt.figure(figsize=(16, 8))
sns.barplot(abs(model.coef_), continuous_feature_names_list)
```

### 决策树中的`model_importance`可用于特征选择

### 绘制学习率曲线与验证曲线

```python
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):  
    plt.figure()  
    plt.title(title)  
    if ylim is not None:  
        plt.ylim(*ylim)  
    plt.xlabel('Training example')  
    plt.ylabel('score')  
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring=make_scorer(mean_absolute_error))  
    print('train_sizes:')
    print(train_sizes)
    print('train_scores:')
    print(train_scores)
    print('test_scores:')
    print(test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)  
    train_scores_std = np.std(train_scores, axis=1)  
    test_scores_mean = np.mean(test_scores, axis=1)  
    test_scores_std = np.std(test_scores, axis=1)  
    plt.grid()  # 区域  
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean,'o-',color='g',
             label='Cross-validation score')
    plt.legend(loc='best')
    return plt  


plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y_ln[:1000], ylim=(0.0, 0.5), cv=5, n_jobs=1)
```

### 看预测值与真实值之间的差距

```python
subsample_index = np.random.randint(low=0, high=len(train_y), size=50)

plt.scatter(train_X['f1'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['f1'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue', alpha=0.4)
plt.xlabel('f1')
plt.ylabel('t1')
plt.legend(['True t1','Predicted t1'], loc='upper right')
plt.show()
```

```python
# sns.residplot
```

### 模型保存

```python
from joblib import dump, load
dump(model,'model/AdaBoostClassifier.pkl')
```

### 模型读取

```python
from joblib import dump, load
model=load('model/AdaBoostClassifier.pkl')
pred_test_y = model.predict(test_X)
```

### 调参

- 贪心算法 https://www.jianshu.com/p/ab89df9759c8
- 网格调参 https://blog.csdn.net/weixin_43172660/article/details/83032029
- 贝叶斯调参 https://blog.csdn.net/linxid/article/details/81189154
- 贝叶斯调参bayes_opt https://www.cnblogs.com/yangruiGB2312/p/9374377.html

考虑使用`TPOT自动调参`选定模型和大致的参数，再用`bayes_opt贝叶斯调参`，`GridSearchCV调参`进一步优化参数。

#### GridSearchCV调参

##### 分类

```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# 构造各种分类器
classifiers = [
    SVC(random_state=1, kernel='rbf'),
    DecisionTreeClassifier(random_state=1, criterion='gini'),
    RandomForestClassifier(random_state=1, criterion='gini'),
    KNeighborsClassifier(metric='minkowski'),
    AdaBoostClassifier(random_state=1),
]

# 分类器名称
classifier_names = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'kneighborsclassifier',
    'adaboostclassifier',
]

# 分类器参数
# 注意分类器的参数，字典键的格式，GridSearchCV对调优的参数格式是"分类器名"+"__"+"参数名"
classifier_param_grid = [
    {'svc__C': [0.1], 'svc__gamma': [0.01]},
    {'decisiontreeclassifier__max_depth': [6, 9, 11]},
    {'randomforestclassifier__n_estimators': range(1, 11)},
    {'kneighborsclassifier__n_neighbors': [4, 6, 8]},
    {'adaboostclassifier__n_estimators': [70, 80, 90]}
]
```

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score


# 对具体的分类器进行 GridSearchCV 参数调优
def GridSearchCV_work(pipeline, train_X, train_y, test_X, test_y, param_grid, score='accuracy_score'):
    response = {}
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring=score, n_jobs=-1, error_score=0.)
    gridsearch.fit(train_X, train_y)  # 寻找最优的参数和最优GridSearchCV的准确率分数
    
    # 最佳分数
    print('Best Score: {}'.format(gridsearch.best_score_))
    # 得到最佳分数的最优参数
    print('Best Parameters: {}'.format(gridsearch.best_params_))
    # 拟合的平均时间（秒）
    print('Average Time to Fit (s): {}'.format(round(gridsearch.cv_results_['mean_fit_time'].mean(), 4)))
    # 预测的平均时间（秒）
    # 从该指标可以看出模型在真实世界的性能
    print('Average Time to Score (s): {}'.format(round(gridsearch.cv_results_['mean_score_time'].mean(), 4)))
    
    # 采用predict函数（特征是测试数据集）来预测标识，预测使用的参数是上一步得到的最优参数
    predict_y = gridsearch.predict(test_X)
    response['best_estimator'] = gridsearch.best_estimator_
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y.values, predict_y)
    print('accuracy_score: %0.4lf' % response['accuracy_score'])
    
    return response


for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    # 采用 StandardScaler 方法对数据规范化：均值为0，方差为1的正态分布
    pipeline = Pipeline([
        # ('scaler', StandardScaler),
        # ('pca',PCA),
        (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_X, train_y, test_X, test_y, model_param_grid, score='accuracy_score')
    print(result)
    print('-' * 16)
```

##### 回归
```python
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.ensemble import GradientBoostingRegressor


# 构造各种分类器
regressor = [
    RandomForestRegressor(oob_score=True, n_jobs=-1, random_state=42),
    xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4),
    GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss='ls', max_depth=7, max_features=0.4, min_samples_leaf=11, min_samples_split=9, n_estimators=100, subsample=0.8)
]
# 分类器名称
regressor_names = [
            'randomforestregressor', 
            'xgbregressor',
            'gradientboostingregressor',
]
# 分类器参数
# 注意分类器的参数，字典键的格式，GridSearchCV对调优的参数格式是"分类器名"+"__"+"参数名"
regressor_param_grid = [
    {
        'randomforestregressor__n_estimators': [1300, 1500, 1700],
        'randomforestregressor__max_depth': range(20, 30, 4),
    },

    {
        'xgbregressor__subsample': [i / 10.0 for i in range(6, 11)],
        'xgbregressor__colsample_bytree': [i / 10.0 for i in range(6, 11)],
    },

    {
        # 'gradientboostingregressor__alpha': [i / 100.0 for i in range(80, 100, 5)],
        # 'gradientboostingregressor__learning_rate': [0.1, 0.01, 0.001],
        'gradientboostingregressor__max_depth': list(range(5, 10)),
        'gradientboostingregressor__max_features': [i / 10.0 for i in range(3, 8)],
        # 'gradientboostingregressor__min_samples_leaf': list(range(7, 16)),
        # 'gradientboostingregressor__min_samples_split': list(range(7, 15)),
        'gradientboostingregressor__n_estimators': list(range(60, 300, 10)),
        'gradientboostingregressor__subsample': [i / 10.0 for i in range(6, 10)],
    },
]
```

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error


# 对具体的分类器进行 GridSearchCV 参数调优
def GridSearchCV_work(pipeline, train_X, train_y, test_X, test_y, param_grid, score='neg_mean_absolute_error'):
    response = {}
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring=score, n_jobs=-1, error_score=0.)
    gridsearch.fit(train_X, train_y)  # 寻找最优的参数和最优GridSearchCV的准确率分数
    
    # 最佳分数
    print('Best Score: {}'.format(gridsearch.best_score_))
    # 得到最佳分数的最优参数
    print('Best Parameters: {}'.format(gridsearch.best_params_))
    # 拟合的平均时间（秒）
    print('Average Time to Fit (s): {}'.format(round(gridsearch.cv_results_['mean_fit_time'].mean(), 4)))
    # 预测的平均时间（秒）
    # 从该指标可以看出模型在真实世界的性能
    print('Average Time to Score (s): {}'.format(round(gridsearch.cv_results_['mean_score_time'].mean(), 4)))
    
    # 采用predict函数（特征是测试数据集）来预测标识，预测使用的参数是上一步得到的最优参数
    predict_y = gridsearch.predict(test_X)
    response['best_estimator'] = gridsearch.best_estimator_
    response['predict_y'] = predict_y
    response['mean_absolute_error'] = mean_absolute_error(test_y.values, predict_y)
    print('mean_absolute_error: %0.4lf' % response['mean_absolute_error'])
    
    return response
 
    
for model, model_name, model_param_grid in zip(regressor, regressor_names, regressor_param_grid):
    # 采用 StandardScaler 方法对数据规范化：均值为0，方差为1的正态分布
    pipeline = Pipeline([
            # ('scaler', StandardScaler),
            #('pca', PCA),
            (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_X, train_y, test_X, test_y, model_param_grid, score='neg_mean_absolute_error')
    print(result)
    print('-' * 16)
```

#### TPOT自动调参

https://github.com/EpistasisLab/tpot

##### 分类
```python
from tpot import TPOTClassifier
tpot = TPOTClassifier(verbosity=2, max_time_mins=2, n_jobs=-1)
tpot.fit(train_X, train_y)
print(tpot.score(test_X, test_y))

tpot.export('tpot_exported_pipeline.py')  # 导出自动调参的模型
```

```python
# tpot 导出的模型
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
training_features, testing_features, training_target, testing_target = train_test_split(features, target, random_state=None)

# Average CV score on the training set was: 0.7995943204868154
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GaussianNB()),
    ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.45, min_samples_leaf=18, min_samples_split=2, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

from sklearn.metrics import accuracy_score
accuracy_score(testing_target, results)
```

##### 回归
```python
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(features, target, train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, n_jobs=-1, random_state=42)
tpot.fit(train_X, train_y)
print(tpot.score(test_X, test_y))  # -mse

tpot.export('tpot_exported_pipeline.py')  # 导出自动调参的模型
```

```python
# tpot 导出的模型
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1)
# training_features, testing_features, training_target, testing_target = \
#             train_test_split(features, tpot_data['target'], random_state=42)
training_features = train_X
testing_features = test_X
training_target =  train_y
testing_target = test_y

# Average CV score on the training set was: -11.510054166227079
exported_pipeline = make_pipeline(
    StandardScaler(),
    XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=8, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.7000000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(testing_target, results)
```

### 贝叶斯调参

#### 回归

```python
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor


def my_bayes_cv4lgbmr(num_leaves, max_depth, n_estimators, min_child_samples, subsample):
    val = cross_val_score(
        LGBMRegressor(
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            n_estimators=int(n_estimators),
            min_child_samples=int(min_child_samples),
            subsample=subsample,
            
            objective='regression_l1',
            learning_rate=0.01,
            random_state=None,
            n_jobs=-1,
        ),
        X=train_df[select_feature_list],
        y=train_df['target1_boxcox'],
        verbose=0,
        cv=5,
        scoring=make_scorer(mean_absolute_error)
    ).mean()
    return -val  # mae越小越好，而贝叶斯调参是求解最大值，所以取负


# 实例化一个bayes优化对象
lgbmr_bo = BayesianOptimization(
    my_bayes_cv4lgbmr,
    {
        'num_leaves': (2, 1000),
        'max_depth': (2, 1000),
        'n_estimators': (2, 1000),
        'min_child_samples': (2, 200),
        'subsample': (0.1, 1),
    }
)


# 运行bayes优化
lgbmr_bo.maximize()


# 输出最佳结果
lgbmr_bo.max

# 使用调参结果建模
my_lgbmr = LGBMRegressor(
    max_depth=556,
    min_child_samples=2,
    n_estimators=1000,
    num_leaves=751,
    subsample=1.0,

    objective='regression_l1',
    learning_rate=0.01,
    random_state=None,
    n_jobs=-1,
)
my_lgbmr.fit(training_features, training_target)
# training_features, testing_features, training_target, testing_target

# 预测
testing_pre = my_lgbmr.predict(testing_features)

# 评估
mean_absolute_error(testing_target, testing_pre)

# 模型评估
np.mean(cross_val_score(my_lgbmr, X=training_features, y=training_target, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
```

#### 分类

```python
# https://www.cnblogs.com/yangruiGB2312/p/9374377.html

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
#
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

# 产生随机分类数据集，10个特征， 2个类别
train_X, train_y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=121)

# 用模型的默认参数训练
rf = RandomForestClassifier()
print(np.mean(cross_val_score(rf, train_X, train_y, cv=20, scoring='roc_auc')))  # 0.9571532051282048

def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),  # float
            max_depth=int(max_depth),
            random_state=2),
        train_X,
        train_y,
        scoring='roc_auc',
        cv=5).mean()
    return val

rf_bo = BayesianOptimization(
    rf_cv, {
        'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 15)
    })

rf_bo.maximize()

rf_bo.max
'''
{'target': 0.9603981858185818,
 'params': {'max_depth': 5.323451089365594,
  'max_features': 0.35032152841688646,
  'min_samples_split': 9.097395601034854,
  'n_estimators': 146.20001157722896}}
'''

# 使用贝叶斯优化器计算出的参数
rf = RandomForestClassifier(max_depth=5,
                           max_features=0.35,
                           min_samples_split=9,
                           n_estimators=146,
                           n_jobs=-1)
print(np.mean(cross_val_score(rf, train_X, train_y, cv=20, scoring='roc_auc')))  # 0.9618334615384614

# ----------------------------------------------------------------

# 对比 网格搜索（GridSearchCV）
from sklearn.model_selection import GridSearchCV
parameters = {
    'n_estimators': list(range(100, 151, 10)) + [145, 146, 147],
    'min_samples_split': range(5, 12, 2),
    'max_features': list(i/10 for i in range(1, 6, 1)) + [0.35, 0.62],
    'max_depth': list(range(1, 16, 3)) + [5, 9]
}

rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters, cv=5, n_jobs=-1)
clf.fit(train_X, train_y)

clf.best_params_
'''
{'max_depth': 10,
 'max_features': 0.35,
 'min_samples_split': 5,
 'n_estimators': 146}
'''

rf = RandomForestClassifier(max_depth=10,
                            max_features=0.35,
                            min_samples_split=5,
                            n_estimators=146,
                            n_jobs=-1)

print(np.mean(cross_val_score(rf, train_X, train_y, cv=20, scoring='roc_auc')))  # 0.9569120512820515
```

### 模型融合

1. 简单加权融合:
   - 回归（分类概率）：算术平均融合（Arithmetic mean），几何平均融合（Geometric mean）；
   - 分类：投票（Voting）(from sklearn.ensemble import VotingClassifier)
   - 综合：排序融合(Rank averaging)，log融合

2. stacking/blending:

   - 构建多层模型，并利用预测结果再拟合预测。

3. boosting/bagging（在xgboost，Adaboost,GBDT中已经用到）:

   - 多树的提升方法

在做结果融合的时候，有一个很重要的条件是模型结果的得分要比较近似，然后结果的差异要比较大，这样的结果融合往往有比较好的效果提升。

基于模型层面的融合最好不同模型类型要有一定的差异，用同种模型不同的参数的收益一般是比较小的。

#### Stacking模型融合

对于第二层Stacking的模型不宜选取的过于复杂，这样会导致模型在训练集上过拟合，从而使得在测试集上并不能达到很好的效果。

##### 回归

```python
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold

# LASSO Regression :
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
# Elastic Net Regression 弹性网回归
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
# Kernel Ridge Regression 岭回归 alpha超参数
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# Gradient Boosting Regression 梯度增强回归
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=666)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        # 使用K-fold的方法来进行交叉验证，将每次验证的结果作为新的特征来进行处理
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        
        # 可以考虑对初级模型的输出（out_of_fold_predictions）做变换（如取log）

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        # 将交叉验证预测出的结果 和 训练集中的标签值进行训练
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    # 从得到的新的特征  采用新的模型进行预测  并输出结果
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                 meta_model=lasso)
stacked_averaged_models.fit(training_features, training_target)
results = stacked_averaged_models.predict(testing_features)

# 评估（分类）
# from sklearn.metrics import accuracy_score
# accuracy_score(testing_target, results)

# 评估（回归）
from sklearn.metrics import mean_absolute_error
mean_absolute_error(testing_target, results)

# 模型保存与读取
from joblib import dump, load
# 模型保存
dump(stacked_averaged_models, 'model/stacked_averaged_models.pkl')
# 模型读取
model = load('model/stacked_averaged_models.pkl')
results = stacked_averaged_models.predict(testing_features)
mean_absolute_error(testing_target, results)
```

##### 分类（mlxtend）

```python
import warnings
warnings.filterwarnings('ignore')
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

# 以python自带的鸢尾花数据集为例
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

label = ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']
clf_list = [clf1, clf2, clf3, sclf]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2)
grid = itertools.product([0, 1], repeat=2)

clf_cv_mean = []
clf_cv_std = []
for clf, label, grd in zip(clf_list, label, grid):

    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print("Accuracy: %.2f (+/- %.2f) [%s]" %
          (scores.mean(), scores.std(), label))
    clf_cv_mean.append(scores.mean())
    clf_cv_std.append(scores.std())

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(label)

plt.show()
```




## 评估

分类Classification、聚类Clustering、回归Regression

### sklearn的model_evaluation

https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation

### 交叉验证`cross_val_score`

```python
# 回归
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

cv_int = 5
scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv=cv_int, scoring=make_scorer(mean_absolute_error))

print('AVG:', np.mean(scores))

scores = pd.DataFrame(scores.reshape(1,-1))
scores.columns = ['cv' + str(x) for x in range(1, cv_int + 1)]
scores.index = ['MAE']
scores
```

```python
# 回归
np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
```

```python
# 分类
np.mean(cross_val_score(model, X=train_X, y=train_y, cv=20, scoring='roc_auc'))
```




## 参考

[Kaggle - 电信用户流失分析与预测](https://zhuanlan.zhihu.com/p/68397317)

[Kaggle - 自行车租赁预测](https://www.kesci.com/home/project/5ea1807a105d91002d4f5142/code)

[天池 - 二手车交易价格预测](https://tianchi.aliyun.com/competition/entrance/231784/information)











