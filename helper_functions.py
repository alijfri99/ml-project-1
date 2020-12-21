from sklearn.preprocessing import OrdinalEncoder


def prepare_column(dataframe, column_name, is_text):
    dataframe[column_name].fillna(dataframe[column_name].mode()[0], inplace=True)  # handle the missing values
    if is_text:  # then encode the column
        enc = OrdinalEncoder()
        enc.fit(dataframe[[column_name]])
        dataframe[column_name] = enc.transform(dataframe[[column_name]])


def prepare_columns(dataframe, columns_dict):
    for column, column_type in columns_dict.items():
        prepare_column(dataframe, column, column_type)
