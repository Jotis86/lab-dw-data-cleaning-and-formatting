import pandas as pd

def load_data(url):
    return pd.read_csv(url)

def standardize_column_names(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'\bst\b', 'state', regex=True)
    return df

def clean_gender(df):
    df['gender'] = df['gender'].replace({'F': 'F', 'M': 'M', 'Femal': 'F', 'Male': 'M', 'female': 'F', 'male': 'M'})
    return df

def clean_state(df):
    state_replacements = {
        'AZ': 'Arizona',
        'Cali': 'California',
        'WA': 'Washington'
    }
    df['state'] = df['state'].replace(state_replacements)
    return df

def clean_education(df):
    df['education'] = df['education'].replace({'Bachelors': 'Bachelor'})
    return df

def clean_customer_lifetime_value(df):
    df['customer_lifetime_value'] = df['customer_lifetime_value'].str.replace('%', '').astype(float)
    return df

def clean_vehicle_class(df):
    df['vehicle_class'] = df['vehicle_class'].replace({'Sports Car': 'Luxury', 'Luxury SUV': 'Luxury', 'Luxury Car': 'Luxury'})
    return df

def clean_number_of_open_complaints(df):
    df['number_of_open_complaints'] = df['number_of_open_complaints'].apply(lambda x: int(x.split('/')[1]) if isinstance(x, str) else x)
    df['number_of_open_complaints'] = pd.to_numeric(df['number_of_open_complaints'], errors='coerce')
    return df

def handle_null_values(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def convert_numeric_to_int(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].astype(int)
    return df

def handle_duplicates(df):
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return df

def save_cleaned_data(df, filename):
    df.to_csv(filename, index=False)

def main(url, output_filename):
    df = load_data(url)
    df = standardize_column_names(df)
    df = clean_gender(df)
    df = clean_state(df)
    df = clean_education(df)
    df = clean_customer_lifetime_value(df)
    df = clean_vehicle_class(df)
    df = clean_number_of_open_complaints(df)
    df = handle_null_values(df)
    df = convert_numeric_to_int(df)
    df = handle_duplicates(df)
    save_cleaned_data(df, output_filename)
    return df