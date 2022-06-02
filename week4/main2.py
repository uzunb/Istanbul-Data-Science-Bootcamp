import pandas as pd 
import numpy as np
import pickle
import lightgbm as lgb

df = pd.read_csv(r"week4\house_price_1.csv")

dropColumns = ["Id", "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2", "BldgType", "OverallCond", "RoofStyle", 
               "RoofMatl", "Exterior1st", "Exterior2nd","MasVnrType", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1",
              "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "Electrical", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath"] + ["SaleCondition", "SaleType", "YrSold", "MoSold", "MiscVal", "MiscFeature", "Fence", "PoolQC", "PoolArea", "ScreenPorch", "3SsnPorch", "EnclosedPorch", "OpenPorchSF", "WoodDeckSF", "PavedDrive", "GarageCond", "GarageQual", "GarageType", "FireplaceQu", "Functional", "KitchenAbvGr", "BedroomAbvGr"]

droppedDf = df.drop(columns=dropColumns, axis=1)

droppedDf.isnull().sum().sort_values(ascending=False)
droppedDf["Alley"].fillna("NO", inplace=True)
droppedDf["LotFrontage"].fillna(df.LotFrontage.mean(), inplace=True)
droppedDf["GarageFinish"].fillna("NO", inplace=True)
droppedDf["GarageYrBlt"].fillna(df.GarageYrBlt.mean(), inplace=True)
droppedDf["BsmtQual"].fillna("NO", inplace=True)
droppedDf["MasVnrArea"].fillna(0, inplace=True)
droppedDf['MasVnrAreaCatg'] = np.where(droppedDf.MasVnrArea>1000,'BIG',
                                      np.where(droppedDf.MasVnrArea>500,'MEDIUM',
                                              np.where(droppedDf.MasVnrArea>0,'SMALL','NO')))

droppedDf = droppedDf.drop(['SalePrice'],axis=1)
inputDf = droppedDf.iloc[[0]].copy()

for i in inputDf:
    if inputDf[i].dtype == "object":
        inputDf[i] = droppedDf[i].mode()[0]
    elif inputDf[i].dtype == "int64" or inputDf[i].dtype == "float64":
        inputDf[i] = droppedDf[i].mean()

obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
for feature in obj_feat:
    inputDf[feature] = inputDf[feature].astype('category')

# load the model weights and predict the target
modelName = r"week4\finalized_model.model"
loaded_model = pickle.load(open(modelName, 'rb'))
prediction = loaded_model.predict(inputDf)
print(prediction)

    
