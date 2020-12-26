from sklearn.preprocessing import LabelEncoder



def prepare_column(dataframe, column_name, is_text):
    if dataframe[column_name].isna().sum() > 0:
        dataframe[column_name].fillna(dataframe[column_name].mode()[0], inplace=True)  # handle the missing values
    if is_text:  # then encode the column
        enc = LabelEncoder()
        dataframe[column_name] = enc.fit_transform(dataframe[column_name])


def prepare_columns(dataframe, columns_dict):
    for column, column_type in columns_dict.items():
        prepare_column(dataframe, column, column_type)
