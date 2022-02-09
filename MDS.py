import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

def read():
    path = input("Вкажіть шлях до файла з даними, розмірність яких бажаєте\nзменшити методом багатовимірного масштабування\n")
    excel_data = pd.read_excel(path)
    print("Дані успішно прочитані")
    print(excel_data)
    Y=excel_data['Age'].values
    return excel_data, Y

def TrancData(array_data):
    trancf=OrdinalEncoder()
    trancf.fit(array_data)    
    return trancf.transform(array_data)

def mds(n,data, distance):
    mod=MDS(n_components=n, 
          metric=True, 
          n_init=4, 
          max_iter=300, 
          verbose=0, 
          eps=1e-12, 
          n_jobs=None, 
          dissimilarity='precomputed',normalize=True)    
    if distance==1:
        dist_manhattan=manhattan_distances(data)       
        data_new = mod.fit_transform(dist_manhattan) 
        matrix_distance=mod.dissimilarity_matrix_                                
    else: 
        dist_euclidean = euclidean_distances(data)
        data_new = mod.fit_transform(dist_euclidean)
        matrix_distance=mod.dissimilarity_matrix_       
    return data_new, matrix_distance,mod.stress_



def plot2D(data_2,data_3, Y):
    fig, axes = plt.subplots(ncols=2, figsize=(14, 7))
    ax1, ax2 = axes
    im1 = ax1.scatter(data_2[:, 0], data_2[:, 1],  c=Y,cmap='brg')
    im2 = ax2.scatter(data_3[:, 0], data_3[:, 1],  c=Y,cmap='brg')    
    plt.colorbar(im1, ax=ax1)
    ax1.set_title(
    'MDS(Manhattan distances)',
    fontsize=10)    
    plt.colorbar(im2, ax=ax2 )
    ax2.set_title(
    'MDS(Euclidean distances)',
    fontsize=10, )    
    plt.show()

def plot3D(data_2, Y,title):    
    ax = plt.subplot (projection = '3d')  
    ax1=ax.scatter (data_2[:, 0], data_2[:, 1], data_2[:,2], c=Y,cmap='brg') 
    plt.colorbar(ax1)
    ax.set_title(
    title,
    fontsize=10)
    plt.show()

def plotMatrix(D1,D2):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    ax1, ax2 = axes
    im1 = ax1.matshow(D1)
    im2 = ax2.matshow(D2)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    ax1.set_title(
    'Матриця несхожості\n\n через Манхеттенську відстань',pad=30,
    fontsize=10)
    ax2.set_title(
    'Матриця несхожості\n\n через Евклідову відстань',pad=30,
    fontsize=10)
    plt.show()
    
def main():
    df, Y=read()
    data=TrancData(df) 
    n=int(input("Введіть розмірність нового набору даних:\n"))
    g_mds, D1,S1=mds(n,data, 1) 
    eu_mds,D2,S2=mds(n,data, 2)
    print("Матриця (Manhattan MDS):\n",D1)
    print("Матриця (Euclidean MDS):\n",D2)
    print("Stress (MDS Manhattan distances):", S1)
    print("Stress (MDS Euclidean distances):", S2)
    print("Новий набір даних (Manhattan MDS):\n",g_mds)
    print("Новий набір даних (Euclidean MDS):\n",eu_mds)
    plotMatrix(D1,D2)
    if n==2: plot2D(g_mds,eu_mds,Y)
    elif n==3:
        plot3D(g_mds,Y,'MDS Manhattan distances')
        plot3D(eu_mds,Y,'MDS Euclidean distances')
    else: print("Для такої розмірності даних графічне відображення не реалізовано")
       

if __name__ == '__main__':
    main()
