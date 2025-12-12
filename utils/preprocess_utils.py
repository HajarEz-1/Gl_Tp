
from sklearn.preprocessing import LabelEncoder


def encode_smoker(df):
    le = LabelEncoder()
    df['smoker_encoded'] = le.fit_transform(df['smoker'])
    return df, le

def select_features(df, features=['age', 'bmi', 'smoker_encoded']):
    return df[features]
