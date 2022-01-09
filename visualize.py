import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("items.csv", index_col='id', encoding='utf-8')

# so luong san pham
xx = df.category.value_counts().sort_values(ascending =True)
ax =sns.barplot(x=xx.index,y=xx.values)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set(xlabel='category', ylabel='count')
plt.show()

# phan phoi cac thuoc tinh numerical

fig = plt.figure(figsize=(20,25))
features_numerical = df.select_dtypes(exclude=['object']).copy()
for i in range(len(features_numerical.columns)):
    fig.add_subplot(5,4,i+1)
    sns.histplot(x=features_numerical.columns[i],data=df,bins=50)
    plt.xlabel(features_numerical.columns[i])
plt.tight_layout()
plt.show()

# phan phoi gia tien

ax=sns.scatterplot(x='price', y='n_sold',data=df,hue='category',palette='coolwarm',style='category')
ax.legend(prop={'size':7})
plt.title('Phân tích giá tiền của các sản phẩm theo từng loại category')
plt.show()

# phan tich muc do cham soc khach hang

color={'trong vài giờ':'blue','trong vài ngày':'g', 'trong vài phút':'orange'}
sns.boxplot(x='shop_time_feedback',y='shop_rate_feedback',data=df,palette=color)
plt.title('Phân tích mức độ chăm sóc khách hàng của các shop')
plt.show()

# phan tich time_feedback

xx = df.shop_time_feedback.value_counts()
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=[10,6])
labels = xx.index
plt.pie(x=xx.values,autopct="%1.1f%%", explode=[0.05]*3,labels=labels,pctdistance=0.5,shadow=True)
plt.title('Phân phối thời gian trả lời tin nhắn của các shop')
plt.show()

# phan tich n_sold va avg_raing

sns.set_style('ticks')
sns.jointplot(x='avg_rating', y='n_sold',hue='category',data=df,palette='coolwarm')
plt.show()

# phan tich cac shop

sns.scatterplot(x='shop_n_product', y='n_sold',data=df)
plt.title('Mối quan hệ giữa số sản phẩm của shop và số sản phẩm bán')
plt.show()

# phan tich tuong tac giua rate_with_cmt va rate_with_imgvid

sns.jointplot(x='rate_with_cmt', y='rate_with_imgvid',data=df)
plt.show()

# phan phoi muc gia san pham dua vao tung loai category

ax = sns.stripplot(x='price',y='category',data=df)
plt.title('Phân phối mức giá sản phậm dựa vào từng loại category')
plt.show()

# phan tich avg_rating va n_loved

sns.scatterplot(x='avg_rating',y='n_loved',data=df)
plt.title('Mối quan hệ giữa avg_rating và n_loved')
plt.show()