from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:

    def __init__(self, categorical_columns, numerical_columns):
        # Initialize the attributes
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
        self.numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('one-hot-encoder', self.categorical_preprocessor, self.categorical_columns),
            ('standard_scaler', self.numerical_preprocessor, self.numerical_columns)])

    def get_processor(self) -> ColumnTransformer:
        return self.preprocessor
