####### Uluslararası bir e-ticaret şirketi, müşteri veri tabanlarından önemli bilgileri keşfetmek istiyor.
#######Şirket elektronik ürünler satmaktadır.


####### Değişkenler #######

#ID: Her bir müşteriye tanımlanan kimlik numarasıdır.
#Warehouse block(Depo Bloğu): Şirketin A,B,C,D,E olarak 5 bloğa bölünmüş büyük bir deposu vardır.
#Mode of shipment(Sevkiyat Şekli): Firma ürünlerini gemi,hava ve karayolu gibi çeşitli şekillerde sevk etmektedir.
#Customer care calls(Müşteri Hizmetleri Aramaları): Gönderinin sorgulanması için yapılan arama sayısıdır.
#Customer rating(Müşteri Değerlendirmesi):Şirketin her müşterisinden aldığı değerlendirme puanıdır. 1 en düşük- 5 en yüksektir.
#Cost of the product(Ürünün Maliyeti): Ürünün ABD Doları cinsinden maliyetidir.
#Prior purchases(Önceki Satın Almalar): Önceki satın alma sayısıdır.
#Product importance(Ürün Önemi): Şirket, ürünü düşük/orta/yüksek gibi parametrelerde sınıflandırmıştır.
#Gender: Cinsiyet bilgisidir.
#Discount offered(Sunulan İndirim): Söz konusu üründe sunulan indirim miktarıdır.
#Weight in gms: Gram cinsinden ağırlıktır.
#Reached on time: Hedef değişkendir.Burada 1 ürünün zamanında ULAŞMADIĞINI, 0 ise zamanında ULAŞTIĞINI gösterir.




#!pip install catboost
#!pip install xgboost
#!pip install lightgbm

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)


df = pd.read_csv("Train.csv")
df.head()
df.tail()

# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



#NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#####

plt.figure(figsize = (15,8))
plt.subplot(1,3,1)
sns.countplot(data = df, x = 'Warehouse_block')
plt.xlabel('Warehouse Block', fontsize = 13)

plt.subplot(1,3,2)
sns.countplot(data = df, x = 'Mode_of_Shipment')
plt.xlabel('Shipment', fontsize = 13)

plt.subplot(1,3,3)
sns.countplot(data = df, x = 'Product_importance')
plt.xlabel('Product Importance', fontsize = 13)

plt.show()

#Çıkarım:
#En çok ürün F depo bloğundadır.
#Teslimatlar en çok Gemi yolu ile yapılmaktadır.
#There are lot of low importance products and medium importance relative to the high importance products.

#####
# Ürünün maliyeti nasıl dağılmaktadır?
plt.figure(figsize = (15,8))
sns.displot(data=df, x='Cost_of_the_Product', kind = 'kde')
plt.xlabel("Cost of the product", fontsize = 13)
plt.show()

#####################################

# Teslim edilen malların ağırlık dağılımının görselleştirilmesi.

plt.figure(figsize = (15,8))
sns.displot(data=df, x='Weight_in_gms', kind = 'kde')
plt.xlabel("Weight", fontsize = 13)
plt.show()


#Numerical Variable Analysis
def numerical_vis(variable):
    plt.figure(figsize = (9,3))
    plt.hist(df[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable),fontsize=12,fontweight="bold",color="#1F1F1F")
    plt.show()
numerical = ["Customer_care_calls","Cost_of_the_Product","Prior_purchases","Discount_offered","Weight_in_gms"]
for num in numerical:
    numerical_vis(num)

var = df.dtypes


#####################################

corr = df.corr()
plt.figure(figsize = (20,10))
sns.heatmap(data = corr, cmap = 'coolwarm', annot = True)
plt.show()

#Çıkarım:
#Aşağıdaki değişkenler arasında pozitif korelasyon görülebilir:
#Reached on time & Discount offered 0.4
#Customer Care calls & cost of the product 0.32
#Prior Purchases & customer care calls 0.18
#Cost of the product and prior purchase 0.12
#Reached on time and customer rating 0.013
#Customer rating and customer care calls 0.012


#####################################
sns.displot(data = df, x = 'Discount_offered',
            hue = 'Reached.on.Time_Y.N',
            kind = 'kde')
plt.xlabel('Discount', fontsize = 8)
plt.title('İndirim verilmesi teslim süresini etkiliyor mu?', fontsize = 8)
plt.show()

#Çıkarım:
#1 ürünün zamanında ULAŞMADIĞINI, 0 ise ULAŞTIĞINI gösterir.
#We can see that on a normal basis a discount is given in both scenarios. between the range of 0 to 10.
#Teslimatların zamanında ulaşmadığı durumlarda çoğu zaman 20 dolardan fazla indirim yapıldığına dair bir gözlem yapılabilir.



#####################################

plt.pie(df.Gender.value_counts(),explode=[.1,.3],startangle=90,autopct='%.2f%%',labels=['female','male'],radius=10,colors=['pink','blue'])
plt.axis('equal')
plt.title('Gender',fontdict={'fontsize':22,'fontweight':'bold'})
plt.show()

df.Gender.value_counts()


####################################

plt.figure(figsize = (12,6))
sns.barplot(data=df, x = 'Customer_care_calls', y = 'Cost_of_the_Product', ci=False)
plt.xlabel('Customer Care Calls', fontsize = 12)
plt.ylabel('Cost of the product', fontsize = 12)
plt.title('Ürünün maliyeti müşteri hizmetleri aramalarını nasıl etkiler?', fontsize = 20)
plt.show()

####################################

plt.figure(figsize = (15,8))
sns.countplot(data = df, x = 'Product_importance', hue = 'Reached.on.Time_Y.N')
plt.xlabel('Product Importance', fontsize = 13)
plt.show()

#####################################

plt.figure(figsize = (15,15))
sns.catplot(data = df, x = 'Warehouse_block',
           col = 'Mode_of_Shipment',
           hue = 'Reached.on.Time_Y.N',
           kind = 'count')
plt.xlabel('Warehouse Block', fontsize = 10)
plt.show()

########################################################

df.Mode_of_Shipment[df.Mode_of_Shipment == 'Flight'] = 1
df.Mode_of_Shipment[df.Mode_of_Shipment == 'Ship'] = 2
df.Mode_of_Shipment[df.Mode_of_Shipment == 'Road'] = 3

df.Product_importance[df.Product_importance == 'high'] = 1
df.Product_importance[df.Product_importance== 'low'] = 2
df.Product_importance[df.Product_importance == 'medium'] = 3
df

df.Gender[df.Gender == 'F'] = 1
df.Gender[df.Gender== 'M'] = 2
df

df.Warehouse_block[df.Warehouse_block == 'A'] = 1
df.Warehouse_block[df.Warehouse_block== 'B'] = 2
df.Warehouse_block[df.Warehouse_block == 'C'] = 3
df.Warehouse_block[df.Warehouse_block== 'D'] = 4
df.Warehouse_block[df.Warehouse_block== 'F'] = 5
df


df["Mode_of_Shipment"]= df["Mode_of_Shipment"].astype('int64')
df["Product_importance"]= df["Product_importance"].astype('int64')
df["Gender"]= df["Gender"].astype('int64')
df["Warehouse_block"]= df["Warehouse_block"].astype('int64')
df.head()

df.dtypes

######################################################
df = df.drop('ID',axis=1)
df
X=df.drop('Reached.on.Time_Y.N',axis=1)
y=df['Reached.on.Time_Y.N']


lgbm_model = LGBMClassifier(random_state=40)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model,X,y, cv=5, scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model,lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=40).fit(X,y)

cv_results = cross_validate(lgbm_final,X,y,cv=5,scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


#Out[53]: 0.6151321261730538
#Out[54]: 0.6112145077800225
#Out[55]: 0.7243992826327267



###########




######

#lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
#               "n_estimators": [200, 300, 350, 400],
#               "colsample_bytree": [0.9, 0.8, 1]}

#lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

#lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=40).fit(X, y)

#cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

#cv_results['test_accuracy'].mean()
#cv_results['test_f1'].mean()
#cv_results['test_roc_auc'].mean()


#######

#lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)
#lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 7000]}

#lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

#lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=40).fit(X, y)

#cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

#cv_results['test_accuracy'].mean()
#cv_results['test_f1'].mean()
#cv_results['test_roc_auc'].mean()


#############




#############



#############
df["NEW_DISCOUNTED_PRICE"] = df["Cost_of_the_Product"] - (df["Cost_of_the_Product"] * df["Discount_offered"] / 100)


#############

df["NEW_DISCOUNT_AMOUNT"] = df["Cost_of_the_Product"] * df["Discount_offered"] / 100

############
##########

df["NEW_RATING_DISCOUNT_RELATIONSHIP"] = df["NEW_DISCOUNT_AMOUNT"] * df["Customer_rating"]

#########

df["NEW_AVERAGE_ORDER_VALUE"] = df["Prior_purchases"] * df["Cost_of_the_Product"]

#########

df["NEW_AVERAGE_WEIGHT_DISCOUNT"] = df["Weight_in_gms"] / df["Cost_of_the_Product"]

#########

df["NEW_CALLS_DISCOUNT_RELATIONSHIP"] = df["Customer_care_calls"] * df["Discount_offered"]


#########

df["NEW_AVERAGE_WAREHOUSE_RATING"] = df["Warehouse_block"] * df["Customer_care_calls"] / df["Customer_rating"]




#############

df = df.drop('ID',axis=1)
df = df.drop('Gender',axis=1)
df = df.drop('Mode_of_Shipment', axis=1)
df = df.drop('Warehouse_block', axis = 1)


df
X=df.drop('Reached.on.Time_Y.N',axis=1)
y=df['Reached.on.Time_Y.N']


lgbm_model = LGBMClassifier(random_state=40)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model,X,y, cv=5, scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model,lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=40).fit(X,y)

cv_results = cross_validate(lgbm_final,X,y,cv=5,scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#cv_results['test_accuracy'].mean()
#Out[75]: 0.6181328703129522
#cv_results['test_f1'].mean()
#Out[76]: 0.6194279585217419
#cv_results['test_roc_auc'].mean()
#Out[77]: 0.7300085271906231


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final,X)

df.head()













