from sklearn.model_selection import train_test_split


def clean_and_split_dataset(dataframe):
    dataframe.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

    X = dataframe.drop(columns=['medv'])
    y = dataframe['medv']

    return train_test_split(X, y, train_size=0.7, random_state=1, shuffle=True)