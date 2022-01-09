import pandas as pd
import numpy as np
import re
import os

files = []
for file in os.listdir('items_raw'):
    df = pd.read_csv(os.path.join('items_raw', file))
    files.append(df)

items = pd.concat(files)
items = items.drop('n_reviews', axis=1)

# xóa các hàng trống
df = items.dropna()

# chuyển dạng các thuộc tính thành string để xử lý chuỗi
col = ['category', 'n_loved', 'n_sold', 'name', 'price', 'rate_1', 'rate_2', 'rate_3', 'rate_4', 'rate_5',
       'rate_with_cmt',
       'rate_with_imgvid', 'shop_age', 'shop_follower', 'shop_n_product', 'shop_n_review', 'shop_name',
       'shop_rate_feedback', 'shop_time_feedback']

df[col] = df[col].astype('string')

func_1 = lambda x: re.findall('.*\((.*)\)', x)[0]
df['n_loved'] = df['n_loved'].apply(func_1)
df['rate_1'] = df['rate_1'].apply(func_1)
df['rate_2'] = df['rate_2'].apply(func_1)
df['rate_3'] = df['rate_3'].apply(func_1)
df['rate_4'] = df['rate_4'].apply(func_1)
df['rate_5'] = df['rate_5'].apply(func_1)
df['rate_with_cmt'] = df['rate_with_cmt'].apply(func_1)
df['rate_with_imgvid'] = df['rate_with_imgvid'].apply(func_1)

# biến đổi thuộc tính shop_age về số tháng tuổi
df['shop_age'] = df['shop_age'].apply(
    lambda x: int(x.split(' ')[0]) if x.split(' ')[1] == 'tháng' else int(x.split(' ')[0]) * 12 + np.random.randint(-6,
                                                                                                                    6))


# func_2 dùng để chuyển các giá trị hàng nghìn và hàng triệu có đuôi k và tr thành số
def func_2(out):
    if out[-1] == 'k':
        out = out[:-1]
        arr = out.split(',')
        if len(arr) == 2:
            out = int(arr[0]) * 1e3 + int(arr[1]) * 1e2
        else:
            out = int(arr[0]) * 1e3
    elif out[-2:] == 'tr':
        out = out[:-2]
        arr = out.split(',')
        if len(arr) == 2:
            out = int(arr[0]) * 1e6 + int(arr[1]) * 1e5
        else:
            out = int(arr[0]) * 1e6
    return float(out)


df['n_loved'] = df['n_loved'].apply(func_2)
df['n_sold'] = df['n_sold'].apply(func_2)
df['rate_1'] = df['rate_1'].apply(func_2)
df['rate_2'] = df['rate_2'].apply(func_2)
df['rate_3'] = df['rate_3'].apply(func_2)
df['rate_4'] = df['rate_4'].apply(func_2)
df['rate_5'] = df['rate_5'].apply(func_2)
df['rate_with_cmt'] = df['rate_with_cmt'].apply(func_2)
df['rate_with_imgvid'] = df['rate_with_imgvid'].apply(func_2)
df['shop_follower'] = df['shop_follower'].apply(func_2)
df['shop_n_product'] = df['shop_n_product'].apply(func_2)
df['shop_n_review'] = df['shop_n_review'].apply(func_2)


# hàm func_3 để xử lý thuộc tính price, nếu có khoảng giá trị price thì lấy trung bình
def func_3(s):
    a = ''.join(s.split('.'))
    a = a.split(' ')
    a[0] = a[0][1:]
    if len(a) > 1:
        a[2] = a[2][1:]
        return (int(a[0]) + int(a[2])) / 2
    return int(a[0])


df['price'] = df['price'].apply(func_3)
# lấy số phần trăm của thuộc tính shop_rate_feedback
df['shop_rate_feedback'] = df['shop_rate_feedback'].apply(lambda x: int(x[:-1]))

# chuyển các thuộc tính số từ float về int
df['n_loved'] = df['n_loved'].astype('int64')
df['n_sold'] = df['n_sold'].astype('int64')
df['price'] = df['price'].astype('int64')
df['rate_1'] = df['rate_1'].astype('int64')
df['rate_2'] = df['rate_2'].astype('int64')
df['rate_3'] = df['rate_3'].astype('int64')
df['rate_4'] = df['rate_4'].astype('int64')
df['rate_5'] = df['rate_5'].astype('int64')
df['rate_with_cmt'] = df['rate_with_cmt'].astype('int64')
df['rate_with_imgvid'] = df['rate_with_imgvid'].astype('int64')
df['shop_follower'] = df['shop_follower'].astype('int64')
df['shop_n_product'] = df['shop_n_product'].astype('int64')
df['shop_n_review'] = df['shop_n_review'].astype('int64')

df = df[df['shop_age'] < 400]
df = df[df['n_sold'] < 13000]
df = df[df['shop_follower'] < 1400000]

df = df.set_index(np.arange(5983))
df.index.name = 'id'

# df.to_csv('items.csv')
