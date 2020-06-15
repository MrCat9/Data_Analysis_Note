# my_step

[TOC]




## 全局设置

### 常用包

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
```

### 忽略警告提示

```python
import warnings
warnings.filterwarnings('ignore')
```

### 设置查看列不省略

```python
import pandas as pd
pd.set_option('display.max_columns', None)
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




## 数据探索

### 查看形状

```python
df.shape
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
```

### 获取数据的描述统计信息

```python
df.describe()
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

### 强制类型转换 

```python
pd.to_numeric(customerDF['str_col'], errors='coerce')  # errors='coerce' -> 将无效解析设置为NaN

df['date_datetime'] = pd.to_datetime(df['date_str'])
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
```

### 类别特征的列转为category数据类型

```python
categorical_features = [
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6'
]
for c in categorical_features:
    if df[c].isnull().any():
        df[c] = df[c].fillna('MISSING')
    df[c] = df[c].astype('category')
```

### 根据列的数据类型选择列

有时候需要人为根据实际含义来区分`数字特征`和`类型特征`

```python
df.select_dtypes(exclude='object')

# 数字特征
numeric_features = df.select_dtypes(include=[np.number])
numeric_features.columns

# 类型特征
categorical_features = df.select_dtypes(include=[np.object])
categorical_features.columns
```

### 日期时间处理

```python
df['datetime_col'] = pd.to_datetime(df['datetime_col'], format='%Y-%m-%d %H:%M:%S')  # 列数据类型转换

# 日期 
df['date'] = df['datetime_col'].dt.date  # 年月日
df['year'] = df['datetime_col'].dt.year  # 年
df['month'] = df['datetime_col'].dt.month  # 月
df['day'] = df['datetime_col'].dt.day  # 日

# 时间
df['time'] = df['datetime_col'].dt.time  # 时分秒
df['hour'] = df['datetime_col'].dt.hour  # 时
df['minute'] = df['datetime_col'].dt.minute  # 分
df['second'] = df['datetime_col'].dt.second  # 秒

df['weekday'] = df['datetime_col'].dt.weekday  # 星期
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
import missingno as msno  # pip install missingno
msno.matrix(df, labels=True)  # DataFrame的无效性的矩阵可视化
msno.bar(df)  # 条形图
msno.heatmap(data)  # 空值间的相关性热力图
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

### 柱形图

```python
# 查看t1（数值型）的频数
plt.hist(df['t1'], orientation='vertical', histtype='bar', color='red')
```

```python
# 类别特征的每个类别频数可视化(countplot)
def count_plot(x,  **kwargs):
    sns.countplot(x=x)
    x=plt.xticks(rotation=90)


f = pd.melt(df, value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(count_plot, "value")
```

```python
# 类别特征的柱形图可视化  # 每个类的t1值的平均值
def bar_plot(x, y, **kwargs):
    sns.barplot(x=x, y=y)
    x=plt.xticks(rotation=90)


f = pd.melt(df, id_vars=['t1'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, "value", "t1")
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
        center=(0, 0), frame=False, rotatelabels=False, hold=None, data=None)

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
# 类别特征箱形图可视化
categorical_features = [
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6'
]
for c in categorical_features:
    if df[c].isnull().any():
        df[c] = df[c].fillna('MISSING')
    df[c] = df[c].astype('category')


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)  # x轴标签逆时针旋转90°


f = pd.melt(df, id_vars=['t1'], value_vars=categorical_features)
g = sns.FacetGrid(f,
                  col="variable",
                  col_wrap=2,
                  sharex=False,
                  sharey=False,
                  size=5)
g.map(boxplot, "value", "t1")
```

### 小提琴图

```python
# 类别特征的小提琴图可视化
catg_list = categorical_features
target = 't1'
for catg in catg_list :
    sns.violinplot(x=catg, y=target, data=df)
    plt.show()
```



### groupby和agg

注意：
使用`agg`的`sum`，`mean`方法会自动填充空值为`0`；使用`agg`的`max`，`min`方法不会自动填充空值（pandas版本：1.0.3）

```python
# 对df使用groupby和agg
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
    ]
).rename(
    columns={
        'min': 'min2',  # 将输出列min重命名为min2（两列min都会被重命名）
        'count': 'count2',
    }
)
```

### groupby和count进行统计

```python
# 用f1的值进行分组，统计组内 t1, t2, t3 的记录条数(不统计NaN)
df.groupby(['f1'])[['t1', 't2', 't3']].count()
```

### 密度分布

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
# 查看特征的 偏度和峰值
numeric_features = df.select_dtypes(include=[np.number])
for col in numeric_features:
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
g2 = sns.FacetGrid(train_df, col="f1", col_wrap=4, sharex=False, sharey=False)
g2.map(plt.scatter, 'f2', 't1', alpha=0.3)
```

```python
# 每个数字特征的分布可视化
numeric_features = df.select_dtypes(include=[np.number])
f = pd.melt(df, value_vars=numeric_features)  # 将“宽表”变成“长表”
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
```

```python
# 相当于用f1进行分组，在每个组里画散点图（f2值不同的点用不同的颜色，横坐标为f3，纵坐标为t1）
g2 = sns.FacetGrid(df, col='f1', hue='f2', col_wrap=4, sharex=False, sharey=False)
g2.map(plt.scatter, 'f3', 't1', alpha=0.3)
g2.add_legend()  # 添加图例
```

```python
# 用f1，f2进行分组
g2 = sns.FacetGrid(df, row='f1', col="f2", hue='f3', sharex=False, sharey=False)
g2.map(plt.scatter, 'f4', 't1', alpha=0.3)
g2.add_legend()
```

### pairplot

sns.pairplot() https://www.jianshu.com/p/6e18d21a4cad

```python
# 特征间两两相关性图
feature_list = ['f1', 'f2', 'f3']
sns.pairplot(df[feature_list], plot_kws={'alpha': 0.1})
```

```python
# 特征与目标相关性图
sns.pairplot(df, x_vars=['f1', 'f2', 'f3', 'f4', 'f5', 'f6'],
             y_vars=['t1', 't2', 't3'], plot_kws={'alpha': 0.1})
```

### regplot

```python
# 散点图+线性回归
fig, ((ax1, ax2), (ax3, ax4), ) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
sns.regplot(x='f1',
            y='t1',
            data=train_df[['f1', 't1']],
            scatter=True,
            fit_reg=True,
            ax=ax1)
sns.regplot(x='f2',
            y='t1',
            data=train_df[['f2', 't1']],
            scatter=True,
            fit_reg=True,
            ax=ax2)
sns.regplot(x='f3',
            y='t1',
            data=train_df[['f3', 't1']],
            scatter=True,
            fit_reg=True,
            ax=ax3)
sns.regplot(x='f4',
            y='t1',
            data=train_df[['f4', 't1']],
            scatter=True,
            fit_reg=True,
            ax=ax4)
```

### 特征间两两相关性矩阵热力图

```python
corr = df.corr()
plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, vmax=1, cmap='YlGnBu')
```

### 目标与各个特征的相关性大小

```python
plt.figure(figsize=(8, 6))
df.corr()['target'].sort_values(ascending=False).plot(kind='bar')  # 降序
# np.abs(df.corr()['target']).sort_values(ascending=False).plot(kind='bar')  # 取绝对值
plt.show()
```

### 选k个和 target 的相关系数最高的特征

```python
k = 10
df.corr().nlargest(k, 'target')['target'].index
```

```python
# 取绝对值
k = 10
np.abs(df.corr()['target']).nlargest(k).index
```

### pandas_profiling生成数据报告

```python
import pandas_profiling
pfr = pandas_profiling.ProfileReport(train_df)
pfr.to_file("example.html")
```




## 特征工程

### 数据清洗的“完全合一”规则

1. 完整性：单条数据是否存在空值，统计的字段是否完善。
2. 全面性：观察某一列的全部数值，通过常识来判断该列是否有问题，比如：数据定义、单位标识、数据本身。
3. 合法性：数据的类型、内容、大小的合法性。比如数据中是否存在非ASCII字符，性别存在了未知，年龄超过了150等。
4. 唯一性：数据是否存在重复记录，因为数据通常来自不同渠道的汇总，重复的情况是常见的。行数据、列数据都需要是唯一的。

### 空值nan的处理规则

1. nan存在的个数如果很小一般选择填充。
2. 如果nan存在的过多、可以考虑删掉。
3. 如果使用lgb等树模型可以直接空缺，让树自己去优化。

### 训练机器学习模型进行空值填充

使用无空值的行作为训练集，训练模型，预测有空值的行的空值。

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

### 把 col_a 中的空值用 col_b 的值替换（df的修改操作）

```python
df.loc[:, 'col_a'].replace(to_replace=np.nan, value=df.loc[:, 'col_b'], inplace=True)
```

###  删除异常值

```python
# 去掉3个标准差以外数据
mask = np.abs(df['t1'] - df['t1'].mean()) <= (3 * df['t1'].std())
df2 = df[mask]
df3 = df[~mask]
```

```python
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot(scale=3)进行清洗
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
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = data_series.index[rule[0] | rule[1]]  # 异常值的索引
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = data_series.index[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = data_series.index[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n


train_df = outliers_proc(train_df, 'power', scale=3)

```

### 不是正态分布

```python
# 数据波动大的话容易产生过拟合
# 所以对数据进行变换，使得数据相对稳定
# 可以选择对数变换
train_y = df['t1']
sns.distplot(train_y)
plt.show()

train_y_log = np.log(train_y)
sns.distplot(train_y_log)
plt.show()
```

### tsfresh时间序列特征工程工具

```python
# tsfresh -> 时间序列特征工程工具
```

### 特征编码

```python
feature_col_name_list = ['f1', 'f2', 'f3', 'f4']
for col in feature_col_name_list:
    if df[col].nunique() == 2:
        df[col] = pd.factorize(dfCate[col])[0]
    else:
        df = pd.get_dummies(df, columns=[col])
```




## 模型

### 模型保存

```python
from joblib import dump, load
dump(model,'model/AdaBoostClassifier.pkl')
```

### 模型读取

```python
from joblib import dump, load
model=load('model/AdaBoostClassifier.pkl')
pred_test_y = model.predict(test_x)
```

### 调参

考虑使用`TPOT自动调参`选定模型和大致的参数，再用`GridSearchCV调参`进一步优化参数。

### GridSearchCV调参

#### 分类

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
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score='accuracy_score'):
    response = {}
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring=score, n_jobs=-1, error_score=0.)
    gridsearch.fit(train_x, train_y)  # 寻找最优的参数和最优GridSearchCV的准确率分数
    
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
    predict_y = gridsearch.predict(test_x)
    response['best_estimator'] = gridsearch.best_estimator_
    response['predict_y'] = predict_y
    response['mean_absolute_error'] = accuracy_score(test_y.values, predict_y)
    print('accuracy_score: %0.4lf' % response['accuracy_score'])
    
    return response


for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    # 采用 StandardScaler 方法对数据规范化：均值为0，方差为1的正态分布
    pipeline = Pipeline([
        # ('scaler', StandardScaler),
        # ('pca',PCA),
        (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid, score='accuracy_score')
    print(result)
    print('-' * 16)
```

#### 回归
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
#注意分类器的参数，字典键的格式，GridSearchCV对调优的参数格式是"分类器名"+"__"+"参数名"
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
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score='neg_mean_absolute_error'):
    response = {}
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring=score, n_jobs=-1, error_score=0.)
    gridsearch.fit(train_x, train_y)  # 寻找最优的参数和最优GridSearchCV的准确率分数
    
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
    predict_y = gridsearch.predict(test_x)
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
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid, score='neg_mean_absolute_error')
    print(result)
    print('-' * 16)
```

### TPOT自动调参

https://github.com/EpistasisLab/tpot

#### 分类
```python
from tpot import TPOTClassifier
tpot = TPOTClassifier(verbosity=2, max_time_mins=2, n_jobs=-1)
tpot.fit(train_x, train_y)
print(tpot.score(test_x, test_y))

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

#### 回归
```python
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(features, target, train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, n_jobs=-1, random_state=42)
tpot.fit(train_x, train_y)
print(tpot.score(test_x, test_y))  # -mse

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
training_features = train_x
testing_features = test_x
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

### Stacking模型融合

#### 回归

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
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

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

# from sklearn.metrics import accuracy_score
# accuracy_score(testing_target, results)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(testing_target, results)

# 模型保存
dump(stacked_averaged_models,'model/stacked_averaged_models.pkl')

# 模型读取
model = load('model/stacked_averaged_models.pkl')
results = stacked_averaged_models.predict(testing_features)
mean_absolute_error(testing_target, results)
```




## 评估

分类Classification、聚类Clustering、回归Regression

### sklearn的model_evaluation

https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation


