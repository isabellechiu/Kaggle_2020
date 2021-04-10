# VinBigData Chest X-ray Abnormalities Detection

https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

## EDA(Exploratory data analysis)
1. fill NAN with 0: fillna(0, inplace=True)
2. check df info: .info()
3. check col value and unique name: .value_counts() ;.unique() 
4. drop useless col: train[train.class_id!=14].reset_index(drop = True)
5. Visualising high-dimensional datasets using PCA and t-SNE in Python
```
features = ['x_min', 'y_min', 'x_max', 'y_max']
X = train_find[features]
y = train_find['class_id']
data_X = X
data_y = y.loc[data_X.index]
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_X.values)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
pca_one = pca_result[:,0]
pca_two = pca_result[:,1] 
pca_three = pca_result[:,2] 
scatter = plt.scatter(pca_one, pca_two, marker = 'o',s = 50, c=data_y.tolist(), alpha= 0.5,cmap='viridis')
## legend_elements (prop: =colors
plt.legend(handles=scatter.legend_elements()[0], labels=class_list_find)
```
![image](https://user-images.githubusercontent.com/39623214/114257747-a5f07000-99f4-11eb-9305-f6a193fa32ba.png)

6. try normalization

```
x_min, x_max = pca_result.min(0), pca_result.max(0)
X_norm = (pca_result - x_min) / (x_max - x_min)  
plt.figure(figsize=(8, 8))
plt.axis('off')
scatter = plt.scatter(X_norm[:,0], X_norm[:,1], marker = 'o',s = 50, c=data_y.tolist(), alpha= 0.5,cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=class_list_find)
```
![image](https://user-images.githubusercontent.com/39623214/114257822-07184380-99f5-11eb-898c-bec622d2e9fc.png)

7. visualization with seaborn
````
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=pca_one, y=pca_two,
    hue=data_y,
    palette=sns.color_palette("hls", 15),
    data=data_y,
    legend=class_list,
    alpha=0.3
)
````
![image](https://user-images.githubusercontent.com/39623214/114257842-2e6f1080-99f5-11eb-82da-7ecc97fde98b.png)

8. 3D
````
from mpl_toolkits.mplot3d import Axes3D

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=pca_one, 
    ys=pca_two, 
    zs=pca_three, 
    c=data_y.tolist(), 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()
````

![image](https://user-images.githubusercontent.com/39623214/114257953-e8ff1300-99f5-11eb-8765-fe1dcfb3805a.png)

9. t-sne 
```
time_start = time.time()
tsne_result = TSNE(n_components=2, verbose=10, perplexity=40, n_iter=300).fit_transform(data_X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
tsne_2d_one = tsne_result[:,0]
tsne_2d_two = tsne_result[:,1]
import matplotlib.pyplot as plt
plt.figure(figsize = (15, 15))
plt.axis('off')
scatter = plt.scatter(tsne_2d_one, tsne_2d_two, marker = 'o',s = 50, c=data_y.tolist(), alpha= 0.5,cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=class_list)
````
![image](https://user-images.githubusercontent.com/39623214/114257905-a4737780-99f5-11eb-9bf9-1b09ea727f95.png)

```
import seaborn as sns

plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_2d_one, y=tsne_2d_two,
    hue=data_y,
    palette=sns.color_palette("hls", 15),
    data=data_y,
    legend=class_list,
    alpha=0.3
)
````
![image](https://user-images.githubusercontent.com/39623214/114257925-bead5580-99f5-11eb-904d-739bb72bfed2.png)

