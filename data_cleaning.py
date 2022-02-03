import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def read_crime_data():
    crime_df = pd.read_csv("data\crime_data.csv")

    df_continuous = crime_df.drop(columns=['state','county','community','communityname','fold','State_names','State_short_code'])

    df_full = df_continuous.drop(columns=['LemasSwornFT','LemasSwFTFieldOps','LemasSwFTFieldOps','LemasTotalReq','LemasTotReqPerPop',
                                            'PolicReqPerOffic','PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp',
                                            'PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz','PolicAveOTWorked',
                                            'PolicCars','PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','PolicBudgPerPop','LemasSwFTPerPop',
                                            'LemasSwFTFieldPerPop']
                                            ).dropna()
    
    return crime_df, df_continuous, df_full

"""
More data cleaning notes and ideas:
create a middle aged element, 1 minus the rest of the age groups


"""
def data_prep_crime_data():
    """
    read in crime dataset (some cleaning done in R)
    remove data that is not useful for PCA and do train, test splits
    Out:
        package data in to Dictionary
        xTrain: independent variables split for training
        yTrain: independent variables split for training
        xTest dependent variables split for testing
        yTest: dependent variables split for testing
    """
    crime_df = pd.read_csv("data\crime_data.csv")

    df_continuous = crime_df.drop(columns=['state','county','community','communityname','fold','State_names','State_short_code'])

    crime_df_clean = df_continuous.drop(columns=['LemasSwornFT','LemasSwFTFieldOps','LemasSwFTFieldOps','LemasTotalReq','LemasTotReqPerPop',
                                            'PolicReqPerOffic','PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp',
                                            'PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz','PolicAveOTWorked',
                                            'PolicCars','PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','PolicBudgPerPop','LemasSwFTPerPop',
                                            'LemasSwFTFieldPerPop']
                                            ).dropna()

    
    crime_df_clean_target = pd.DataFrame(crime_df_clean, columns = ['ViolentCrimesPerPop'])
    new_dtypes = {"ViolentCrimesPerPop": np.float64}
    crime_df_clean_target = crime_df_clean_target.astype(new_dtypes)
    crime_df_clean.drop(crime_df_clean.loc[crime_df_clean['ViolentCrimesPerPop']==1].index, inplace=True)
    crime_df_clean_target.drop(crime_df_clean_target.loc[crime_df_clean_target['ViolentCrimesPerPop']==1].index, inplace=True)    
    
    crime_df_clean = crime_df_clean.drop(columns=['ViolentCrimesPerPop'])
    return crime_df_clean_target, crime_df_clean
                                    
if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    print("describe:")
    print(df_full.describe())
    print("\n")
    print('info:')
    print(df_full.info())
    print("\n")
    print("NaN Count:")
    print(df_full.isnull().sum())
