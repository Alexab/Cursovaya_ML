import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_source = "my_csv.csv"
data = pd.read_csv(data_source)
print(data.info(), end="\n-------------\n")

def task_1(): # Сколько записей в базе

    print(data.info(), end="\n-------------\n")
    print(len(data))


def task_2(): # Построение гистограмм
    groups = list(data.columns)
    for group in groups:
        plt.figure(num=group)
        plt.hist(x=data[group], bins=None)
        plt.xlabel('value')
        plt.ylabel('quantity')
    plt.show()



def corr(): # task_5 Матрица корреляции

    corr = data.corr()
    sns.heatmap(corr,annot=True,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()



task_2()

#plt.figure(num='longitude')
#plt.hist(x=data['longitude'], bins=None)
#plt.show()
