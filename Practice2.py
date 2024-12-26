import numpy as np
from pandas import date_range
from pandas import DataFrame
from pandas import Series
from pandas import Timestamp
from pandas import Categorical
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# s = pd.Series([5,8,6,7,np.nan,6,4,6])
# print(s)

#========================================= Data Frame =========================================

dates = date_range('2023-11-12', periods=6)
# print(dates)

df = DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
# print(df)

# print(pd.Series(1, index=list(range(4)), dtype="float32"))
# print(np.array([3] * 4, dtype="int32"))
df2 = DataFrame(
    {
        "A": 1.0,
        "B": Timestamp("20130102"),
        "C": Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": Categorical(["test", "train", "test", "train"]),
        "F": "foo"
        #,["Test"] not equal to length "4" is unacceptable
    }
)
# print(df2)
# print(df2.dtypes)

#Try to get data from internet
# iris = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
# print(iris.head())
# print(iris.tail(3))
# print(iris.index)
# print(iris.columns)
# print(iris.head().to_numpy())
# print(iris.head().describe())
# print(iris.head().T)
# print(iris.head().sort_index(axis=1, ascending=False)) #Sort by column names alphabet
# print(iris.head().sort_index(axis="index", ascending=False)) #Sort by row (index)
# print(iris.head().sort_values(by="sepal.length"))
# print(iris[iris["sepal.length"] > 6])

#========================================== soft_max ==========================================

def soft_max(np_array):
    exp_array = np.exp(np_array)
    expTotal = np.sum(exp_array)
    return exp_array/expTotal
iris = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
# Apply softmax to the numeric columns of the first 5 rows
numeric_cols = iris.select_dtypes(include=[np.number]).columns  # Select only numeric columns

# Update DataFrame using correct shape handling
iris.loc[:4, numeric_cols] = iris.loc[:4, numeric_cols].apply(
    lambda row: pd.Series(soft_max(row.to_numpy()), index=numeric_cols), axis=1
)

print(iris.head())

#================================== Series Calculation ========================================

# Series_obj1 = Series([1, 3, 5, 7], index=['a','b','c','d'])
# Series_obj2 = Series([2, 4, 6, 8], index=['a','b','c','d'])
# Series_obj3 = Series_obj2.T
# print(Series_obj1 + Series_obj2)
# print(Series_obj1 * Series_obj2)
# print(Series_obj1 * Series_obj3)

#=================================== DataFrame Calculation ====================================

# DataFrame_obj1 = DataFrame([[1,3,5,7],[2,4,6,8]], columns=list("ABCD"))
# DataFrame_obj2 = DataFrame([[1,3],[5,7],[2,4],[6,8]], index=list("ABCD"))
# # print(DataFrame_obj1)
# # print(DataFrame_obj2)
# result = DataFrame_obj1.dot(DataFrame_obj2)
# print(result)

#=================================Pandas=====================================

# from pandas import Series, DataFrame
# # DataFrame can have both row and column header in sametime
# Stud_data = {'grade': ['Freshman', 'Sophomore', 'Junior', 'Senior', 'Graduate'], 'year': [2017, 2016, 2016, 2016, 2017], 'number': [180, 170, 160, 170, 50]}

# Stud_frame = DataFrame(Stud_data)
# # print(Stud_frame)

# # DataFrame(Stud_data, columns=['year', 'grade', 'number'])
# Stud_frame2 = DataFrame(Stud_data, columns=['year', 'grade', 'number', 'registered'], index=['one', 'two', 'three', 'four', 'five'])
# # print(Stud_frame2)

# # 結合Series和DataFrame:
# # 使用Series賦予DataFrame某一欄資料值
# NewRegistered = Series([140, 120, 45], index=['two', 'four', 'five'])
# Stud_frame2['registered'] = NewRegistered
# print(Stud_frame2)

# Series_obj4 = Series(['a', 'b', 'c'], index=[0,2,3])
# print("Ffill:")
# print(Series_obj4.reindex(range(6), method="ffill"), "\n")
# print("Bfill:")
# print(Series_obj4.reindex(range(6), method="bfill"), "\n")
# print("Backfill:")
# print(Series_obj4.reindex(range(6), method="backfill"), "\n")
# print("Nearest:")
# print(Series_obj4.reindex(range(6), method="nearest"), "\n")
# print("Pad:")
# print(Series_obj4.reindex(range(6), method="pad"))

#=================================Quizs=====================================

# from pandas import DataFrame
# quiz1 = {'name':['Tony', 'Eric', 'Eugenia'], 'score':[75,60,80]}
# quiz1 = DataFrame(quiz1)
# quiz2 = {'name':['Tony', 'Eric', 'Eugenia'], 'score':[60,70,60]}
# quiz2 = DataFrame(quiz2)
# quiz3 = DataFrame(quiz1, columns=['name', 'score1', 'score2', 'sum'])
# quiz3['score1'] = quiz1['score']
# quiz3['score2'] = quiz2['score']
# quiz3['sum'] = (quiz1.score + quiz2.score)
# # print(quiz3)

# quiz4 = (quiz3.score1 + quiz3.score2)/2
# quiz3['Average'] = quiz4

# # print(quiz3.sort_values(by='Average'))
# print(quiz3)

# print(quiz3.drop(quiz3.columns[[1,2,3]],axis=1))

#====================================PIL=====================================

# # Save using PIL (this ensures proper BMP format)
# # imgplot = plt.imshow(pil_image)
# # plt.show()
# img_path = "./2024_Predator_option_01_3840x2400.jpg"
# # Load the image  # Replace with your image path
# img = Image.open(img_path)
# plt.imshow(img)
# plt.axis('off')
# plt.title("Displayed Image")
# plt.show()

#====================================Plot=====================================

# x=np.arange(1,10,0.1)
# y=np.square(x)
# plt.plot(x,y,'*',color='red')
# plt.show()

# x=[1,2,3,4,5]
# y=[5,4,3,2,1]
# plt.plot(x,y,linestyle='--',markersize=20,marker='*',markeredgecolor='#36EE12',label="x,y as *")
# plt.show()


# x=np.array([5,4,3,2,1])
# y=pow(x,3)
# plt.title("Course Practice", loc="right")
# plt.axis([1,6,1,150])
# plt.plot(x,y,label="y = x ** 3")
# plt.legend()