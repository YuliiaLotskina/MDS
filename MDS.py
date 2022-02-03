import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
def read():
    path = input("Вкажіть шлях до файла з даними, розмірність яких бажаєте\nзменшити методом багатовимірного масштабування\n")
    excel_data = pd.read_excel(path)
    print("Дані успішно прочитані")
    print(excel_data)
    S=excel_data['Age'].values
    """
    i=0
    Y=[]
    diag="Healthy"
    #diag="Live"
    name='Diabetes'
    #name='Class'
    gf1=len(excel_data[name])
    for i in range(gf1):
        if excel_data[name].values[i] == diag:Y.append(1)
        else: Y.append(0)
        """
    return excel_data, S

def TrancData(df):
    trancf=OrdinalEncoder()
    trancf.fit(df)    
    return trancf.transform(df)
def mds(n,data, inf):
    mod=MDS(n_components=n, 
          metric=True, 
          n_init=4, 
          max_iter=300, 
          verbose=0, 
          eps=1e-12, 
          n_jobs=None, 
          dissimilarity='precomputed',normalize=True)    
    if inf==1:
        dist_g=manhattan_distances(data)       
        data_2 = mod.fit_transform(dist_g) 
        D=mod.dissimilarity_matrix_                                
    else: 
        dist_e = euclidean_distances(data)
        data_2 = mod.fit_transform(dist_e)
        D=mod.dissimilarity_matrix_       
    return data_2, D,mod.stress_



def plot2D(data_2,data_3, Y):
    fig, axes = plt.subplots(ncols=2, figsize=(14, 7))
    ax1, ax2 = axes
    j=0
    old1=30
    old2=50
    dat1=pd.DataFrame()
    dat2=pd.DataFrame()
    dat11=pd.DataFrame()
    dat22=pd.DataFrame()
    dat111=pd.DataFrame()
    dat222=pd.DataFrame()
    d_len=len(data_2[:,0])
    for j in range(d_len):
        if Y[j]<old1:
            dat1=dat1.append(pd.DataFrame([data_2[j]]))
            dat2=dat2.append(pd.DataFrame([data_3[j]]))
        elif old1<=Y[j]<old2: 
            dat11=dat11.append(pd.DataFrame([data_2[j]]))
            dat22=dat22.append(pd.DataFrame([data_3[j]]))
        else: 
            dat111=dat111.append(pd.DataFrame([data_2[j]]))
            dat222=dat222.append(pd.DataFrame([data_3[j]]))
    size_marker=50    
    im1 = ax1.scatter(dat1.iloc[:, 0], dat1.iloc[:, 1],s=size_marker,marker="*")
    im1 = ax1.scatter(dat11.iloc[:, 0], dat11.iloc[:, 1],s=size_marker,marker="+")
    im1 = ax1.scatter(dat111.iloc[:, 0], dat111.iloc[:, 1],s= size_marker,marker="s")
    im2 = ax2.scatter(dat2.iloc[:, 0], dat2.iloc[:, 1],s= size_marker,marker="*")
    im2 = ax2.scatter(dat22.iloc[:, 0], dat22.iloc[:, 1],s= size_marker,marker="+")
    im2 = ax2.scatter(dat222.iloc[:, 0], dat222.iloc[:, 1],s= size_marker, marker="s")    
    #plt.colorbar(im1, ax=ax1)
    ax1.legend(["до 30 років","від 30 років до 50","від 50 років"],bbox_to_anchor=(1, 0), loc="center")
    ax1.set_title(
    'MDS(Manhattan distances)',
    fontsize=10)    
   #plt.colorbar(im2, ax=ax2 )
    ax2.legend(["до 30 років","від 30 років до 50","від 50 років"],bbox_to_anchor=(1, 0), loc="center")
    ax2.set_title(
    'MDS(Euclidean distances)',
    fontsize=10 )    
    plt.show()
    
def NStress(data,inf,m):
    stress = []
    
# Max value for n_components
    if inf==1:
        dist=manhattan_distances(data)
        #gower.gower_matrix ( data )
                                  
    else: 
        dist = euclidean_distances(data)
        
    max_range = 21
    
    for dim in range(1, max_range):
    # Set up the MDS object
        mds = MDS(n_components=dim, 
                  metric=True, 
                  n_init=4, 
                  max_iter=300, 
                  verbose=0, 
                  eps=1e-12, 
                  n_jobs=None, 
                  dissimilarity='precomputed',normalize=True)
            #MDS(n_components=dim, dissimilarity='precomputed', random_state=0, normalize=True)
    # Apply MDS
        
        data_2 = mds.fit_transform(dist)
        
        
    # Retrieve the stress value
        stress.append(mds.stress_)
    print(stress)
# Plot stress vs. n_components    
    return stress
def plotStress(max_range,stress):
    
    plt.plot(range(1, max_range), stress)
    plt.xticks(range(1, max_range, 2))
    plt.xlabel('розмірність набору даних')
    plt.title('Стрес MDS')
    plt.ylabel('стрес')
    plt.legend(('Manhattan distances'))
    plt.savefig("MDS 2D.png")
    plt.show()
    
   
def plot3D(data_2, Y,title):    
    ax1 = plt.subplot (projection = '3d')  
    j=0
    old1=30
    old2=50
    dat1=pd.DataFrame()
    dat2=pd.DataFrame()
    dat3=pd.DataFrame()
    d_len=len(data_2[:,0])
    for j in range(d_len):
        if Y[j]<old1:
            dat1=dat1.append(pd.DataFrame([data_2[j]]))
        elif old1<=Y[j]<old2: 
            dat2=dat2.append(pd.DataFrame([data_2[j]]))
        else: 
            dat3=dat3.append(pd.DataFrame([data_2[j]]))
            
    size_marker=30    
    im1 = ax1.scatter(dat1.iloc[:, 0], dat1.iloc[:, 1],s=size_marker,marker="+")
    im1 = ax1.scatter(dat2.iloc[:, 0], dat2.iloc[:, 1],s=size_marker,marker="*")
    im1 = ax1.scatter(dat3.iloc[:, 0], dat3.iloc[:, 1],s= size_marker,marker="s")
    #ax1=ax.scatter (data_2[:, 0], data_2[:, 1], data_2[:,2], c=Y,cmap='brg') 
    #plt.colorbar(ax1)
    ax1.legend(["до 30 років","від 30 років до 50","від 50 років"],bbox_to_anchor=(0, 0), loc="center")
    ax1.set_title(
    title,
    fontsize=10)
    plt.show()
    plt.savefig("MDS 3D.png")
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
    
def save(data,data1,inf,inf2):
    #data=([data_1[:,0],data_1[:,1],data_2[:,0],data_2[:,1]])
    df = pd.DataFrame(data)
    df1= pd.DataFrame(data1)
    with pd.ExcelWriter('./MDS2.xlsx') as writer:  # doctest: +SKIP
     df.to_excel(writer, sheet_name=inf)
     df1.to_excel(writer, sheet_name=inf2)
    print("Дані збережено в файлі 'MDS1.xlsx' ")
def main():
    df, Y=read()
    data=TrancData(df) 
    n=int(input("Введіть розмірність нового набору даних:\n"))
    #m_mds=my_mds(data,n)
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
    #save(g_mds,eu_mds,"Manhattan","Euclidean")
     
    #plotStress(21,NStress(data,2,20))
    
    #save_rezult(m_mds, eu_mds, manh_mds)    
    

    

if __name__ == '__main__':
    main()
