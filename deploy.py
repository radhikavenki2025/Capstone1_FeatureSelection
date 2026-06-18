import pickle
import pandas as pd

def prepro_input(new_student_raw, feature_columns):
    required_raw_cols = [
                        'age', 'study_hrs', 'sleep_hrs', 'media_hrs', 'exercise_fq',
                        'att_percent', 'part_time_job', 'extracurricular',
                        'diet_quality', 'internet_quality', 'exam_score',
                        'gender', 'parent_ed'
                        ]
    missing_raw = [col for col in required_raw_cols if col not in df.columns]
    if missing_raw:
        raise ValueError(
                        f"raw_data is missing required input field(s): {missing_raw}. "
                        f"These must be provided directly -- they cannot be inferred."
                        )

    # --- One-hot encode gender and parent_ed, same as training ---
    if 'gender' in df.columns:
        df = pd.get_dummies(df, columns=['gender'], prefix='gender', drop_first=False)

    if 'parent_ed' in df.columns:
        df['parent_ed'] = df['parent_ed'].fillna('Unknown')
        df = pd.get_dummies(df, columns=['parent_ed'], prefix='par_edu', drop_first=False)

    # Convert any boolean columns created by get_dummies into 0/1 ints
    bool_cols = df.select_dtypes('bool').columns
    df = df.astype({col: int for col in bool_cols})

    # --- Binary Yes/No columns ---
    binary_cols = ['part_time_job', 'extracurricular']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'No': 0, 'Yes': 1}).astype(int)

    # --- Ordinal columns ---
    if 'diet_quality' in df.columns:
        df['diet_quality'] = df['diet_quality'].map({'Poor': 0, 'Fair': 1, 'Good': 2}).astype(int)

    if 'internet_quality' in df.columns:
        df['internet_quality'] = df['internet_quality'].map({'Poor': 0, 'Average': 1, 'Good': 2}).astype(int)

    
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

       df = df[feature_columns]

    return df


def load_model():

    filename = "student_mentalhealth.sav"

    model = pickle.load(open(filename,'rb'))

    return model
    

def load_columns():

    columns = pickle.load(open("feature_columns.pkl",'rb'))

    return columns
    

def predict_student(new_student_raw):

    model = load_model()

    prepro_input = preprocess_input(new_student_raw)

    prediction = model.predict(prepro_input)

    return prediction