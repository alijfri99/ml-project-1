from sklearn.preprocessing import OrdinalEncoder


def prepare_column(dataframe, column_name, is_categorical):
    dataframe[column_name].fillna(dataframe[column_name].mode()[0], inplace=True)  # handle the missing values
    if is_categorical:  # then encode the column
        enc = OrdinalEncoder()
        enc.fit(dataframe[[column_name]])
        dataframe[column_name] = enc.transform(dataframe[[column_name]])
