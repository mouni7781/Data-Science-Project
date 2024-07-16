import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings
import joypy
from pycaret.regression import *
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb
print(os.listdir())

def preprocessing(file):
    print(file.info())
    print(file.duplicated().sum())
    Country_name=file.Country.value_counts().index 
    Country_val=file.Country.value_counts().values
    fig,ax=plt.subplots(figsize=(10,10))
    ax.pie(Country_val[:10],labels=Country_name[:10],autopct='%1.2f%%')
    plt.show()
    sns.pairplot(file,size=3)
    plt.show()
    for i in file.select_dtypes(include="object").columns:
        print(file[i].value_counts())
        print("***"*10)       
#exp_data_analysis(file):
    print(file.describe(include="object"))
    for i in  file.select_dtypes(include="number").columns:  #boxplot
       sns.boxplot(data=file,x=i)
       plt.show()
    print(file.select_dtypes(include="number").columns)
    for i in ['Order ID', 'Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue','Total Cost', 'Total Profit']:
        sns.scatterplot(data=file,x=i,y="Total Profit")      #Scatterplot
        plt.show()
    s=file.select_dtypes(include="number").corr()        #correlation values
    sns.heatmap(s,annot=True)                            #heatmap
    plt.show()
    print(file.isnull().sum())                      #checking for null values
    #joypy.joyplot(file,column=['Total Revenue','Total Cost','Total Profit'],by='Item Type',figsize=(9,6),xlabelsize=12,ylim='own',ylabelsize=12,grid='both',yrot=10,fill=True,legend=True,overlap=2)
def wisker(col):
    q1,q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw
#def outlier_remover(lw,uw):
 #   for i in ['Total Revenue','Total Cost', 'Total Profit']:
   #     lw,uw=wisker(file[i])
  #      file[i]=np.where(file[i]<lw,lw,file[i])
    #    file[i]=np.where(file[i]>uw,uw,file[i])
    #for i in  file.select_dtypes(include="number").columns:
     #   sns.boxplot(data=file,x=i)
      #  plt.show()
#encode the file
    #a=pd.get_dummies(data=file,columns=['Country'],drop_first=True)
    return file
def sales_off_onli(file, channel):
    sales = file[file['Sales Channel'] == 'Offline']
    sales_profit = sales.groupby('Region')['Total Profit'].sum().reset_index()
    sales_profit.columns = ['Region', f'{channel.lower()}_profit']
    fig, ax = plt.subplots()
    ax.barh(sales_profit['Region'], sales_profit[f'{channel.lower()}_profit'], color='skyblue')
    ax.set_xlabel(f'{channel.lower()} Profit')
    ax.set_ylabel('Region')
    ax.set_title(f'{channel.lower()} Sales Profit by Region')
    plt.show()
    
def changing_dtype_split(file):                                        #extracting required from columns(date- month & year)
    file['Order Date']=pd.to_datetime(file['Order Date'])   
    file['Item Type']=file['Item Type'].astype(str)                   #changing datatype to str as requirement               
    file['Order month']=file['Order Date'].dt.month 
    file['Order year']=file['Order Date'].dt.year
    file['Order MonthYear']=file['Order Date'].dt.strftime('%Y-%m')  
    print(file['Order month'],file['Order year'])
    return file
def rev_per_ym(file,type):
    plt.bar(file['Order'f'{type}'],file['Total Revenue'])
    plt.title('orders Revenue per'f'{type}')
    plt.ylabel('total revenue')
    plt.xlabel('order 'f'{type}')
    plt.show()
def profit_per_ym(file,type):
    file.groupby('Order 'f'{type}')['Total Profit'].mean().plot()
    plt.title("orders profit per"f"{type}")
    plt.xlabel('order'f'{type}') 
    plt.ylabel('Total profit per' f'{type.lower()}')
    plt.show()
def predictive_analysis(file):
    le = LabelEncoder()
    file["Item Type"] = le.fit_transform(file["Item Type"])
    file["Sales Channel"] = le.fit_transform(file["Sales Channel"])
    file["Order Priority"] = le.fit_transform(file["Order Priority"])
    columns=['Region','Country','Order ID','Ship Date']
    file.drop(columns,axis=1)
    print(file.head())
    file_auto_ml=file
    reg = setup(data=file, target='Total Profit', log_experiment=False)
    compare_models()
    # Using Lasso Least Angle Regression algorithm to train model 
    llar_model = create_model('llar')
    tuned_llar_model = tune_model(llar_model)
    plot_model(tuned_llar_model)
    plot_model(tuned_llar_model, plot="error")
    plot_model(tuned_llar_model,plot='feature')
    predict_model(tuned_llar_model)
    
def linear_reg(file):
    le = LabelEncoder()
    file["Item Type"] = le.fit_transform(file["Item Type"])
    file["Sales Channel"] = le.fit_transform(file["Sales Channel"])
    file["Order Priority"] = le.fit_transform(file["Order Priority"])
    columns=['Region','Country','Order ID','Ship Date']
    file.drop(columns,axis=1)
    
    X = file[['Item Type', 'Sales Channel', 'Order Priority', 'Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Order month', 'Order year']]
    y = file['Total Profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    regression = LinearRegression()
    regression.fit(X_train,y_train)
    mse = cross_val_score(regression,X_train,y_train,scoring="neg_mean_squared_error",cv=5)
    print(np.mean(mse))
    reg_pred = regression.predict(X_test)
    print(reg_pred)
    sns.displot(reg_pred - y_test,kind='kde', height=5, aspect=2)
    score = r2_score(reg_pred,y_test)
    plt.scatter(y_test, reg_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()
# Calculate the percentage of accuracy
    accuracy_pct = score * 100
    print("Accuracy: {:.2f}%".format(accuracy_pct))
def main():
    path="C:\\Users\\mouni\\Downloads\\Amazon_Sales_data.csv"
    warnings.filterwarnings("ignore")
    file=pd.read_csv(path)
   # preprocessing(file)
    #sales_off_onli(file, 'Offline')
    #sales_off_onli(file, 'Online')
    changing_dtype_split(file)
    #profit_per_ym(file,"year")
   # profit_per_ym(file,"month")
    #profit_per_ym(file,"MonthYear")
    #pd.set_option('display.max_rows', None)
    #print(file['Region'].value_counts())
    #print(file['Country'].value_counts())
    predictive_analysis(file)
    linear_reg(file)
if __name__== "__main__":
    main()
