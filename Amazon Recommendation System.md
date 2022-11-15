```python
import numpy as np
import pandas as pd
```


```python
data=pd.read_csv('C:/Users/Rakesh/Datasets/ratings_Electronics.csv')
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AKM1MP6P0OYPR</th>
      <th>0132793040</th>
      <th>5.0</th>
      <th>1365811200</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A2CX7LUOHB2NDG</td>
      <td>0321732944</td>
      <td>5.0</td>
      <td>1341100800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A2NWSAGRHCP8N5</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1367193600</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2WNBOD3WNDNKT</td>
      <td>0439886341</td>
      <td>3.0</td>
      <td>1374451200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A1GI0U4ZRJA8WN</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1334707200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A1QGNMC6O1VW39</td>
      <td>0511189877</td>
      <td>5.0</td>
      <td>1397433600</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.columns = ['user_id', 'product_id','ratings','timestamp']
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>product_id</th>
      <th>ratings</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A2CX7LUOHB2NDG</td>
      <td>0321732944</td>
      <td>5.0</td>
      <td>1341100800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A2NWSAGRHCP8N5</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1367193600</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2WNBOD3WNDNKT</td>
      <td>0439886341</td>
      <td>3.0</td>
      <td>1374451200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A1GI0U4ZRJA8WN</td>
      <td>0439886341</td>
      <td>1.0</td>
      <td>1334707200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A1QGNMC6O1VW39</td>
      <td>0511189877</td>
      <td>5.0</td>
      <td>1397433600</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (7824481, 4)




```python
df=data[:int(len(data) *.1)]
```


```python
df.shape
```




    (782448, 4)




```python
counts = df['user_id'].value_counts()
data = df[df['user_id'].isin(counts[counts >= 50].index)]
data.groupby('product_id')['ratings'].mean().sort_values(ascending=False)
final_ratings = data.pivot(index='user_id', columns = 'product_id', values='ratings').fillna(0)

num_of_ratings = np.count_nonzero(final_ratings)
possible_ratings = final_ratings.shape[0] * final_ratings.shape[1]
density = (num_of_ratings/possible_ratings)
density *= 100
final_ratings_T = final_ratings.transpose()
```


```python
grouped = data.groupby('product_id').agg({'user_id':'count'}).reset_index()
grouped.rename(columns = {'user_id':'score'},inplace=True)
training_data = grouped.sort_values(['score', 'product_id'], ascending=[0,1])
training_data['Rank'] = training_data['score'].rank(ascending=0,method='first')
recommendations = training_data.head()
```


```python
def recommend(id):
    recommend_products = recommendations
    recommend_products['user_id']=id
    column = recommend_products.columns.tolist()
    column = column[-1:] + column[:-1]
    recommend_products = recommend_products[column]
    return recommend_products

```


```python
print(recommend(11))
```

          user_id  product_id  score  Rank
    113        11  B00004SB92      6   1.0
    1099       11  B00008OE6I      5   2.0
    368        11  B00005AW1H      4   3.0
    612        11  B0000645C9      4   4.0
    976        11  B00007KDVI      4   5.0
    

    C:\Users\Rakesh\AppData\Local\Temp\ipykernel_18284\1876584664.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      recommend_products['user_id']=id
    
