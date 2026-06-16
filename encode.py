import pandas as pd
import numpy as np

def data_rename(st_habits1):

### Drop student id column 

    st_habits1 = st_habits1.drop('student_id', axis=1)

### Standardise and rename column names for ease of use
    
    st_habits1.rename( columns={ 'study_hours_per_day': 'study_hrs',
                                 'sleep_hours':'sleep_hrs',
                                 'exercise_frequency': 'exercise_fq',
                                 'attendance_percentage':'att_percent',
                                 'parental_education_level': 'parent_ed',
                                 'extracurricular_participation':'extracurricular',
                               },inplace=True )

### Compute media hrs column combining social media hrs and netflix hrs 
    
    st_habits1['media_hrs'] = st_habits1['social_media_hours']+ st_habits1['netflix_hours']

    st_habits1= st_habits1.drop(columns=['social_media_hours', 'netflix_hours'])

    cols = st_habits1.columns.tolist()

    cols.remove('media_hrs')  
    cols.insert(4, 'media_hrs') 

    st_habits1 = st_habits1[cols]

    return st_habits1

    

def data_encoding(st_habits1):
    
### one hot encoding // no order // gender & parent education columns

    st_habits1 = pd.get_dummies(st_habits1, columns=['gender'], prefix='gender', drop_first=False)

    st_habits1['parent_ed'] = st_habits1['parent_ed'].fillna('Unknown')
    st_habits1 = pd.get_dummies(st_habits1, columns=['parent_ed'], prefix='par_edu', drop_first=False)

    st_habits1 = st_habits1.astype({col: int for col in st_habits1.select_dtypes('bool').columns})

### converts the boolean (true / false) to 0/1

### Binary columns // yes - no // parttime job , extracurricular

    binary_cols = ['part_time_job', 'extracurricular']
    for col in binary_cols:
        st_habits1[col] = st_habits1[col].map({'No': 0, 'Yes': 1}).astype(int)

### Ordinal columns // diet quality , internet quality

    st_habits1['diet_quality']    = st_habits1['diet_quality'].map({'Poor': 0, 'Fair': 1, 'Good': 2}).astype(int)
    st_habits1['internet_quality'] = st_habits1['internet_quality'].map({'Poor': 0, 'Average': 1, 'Good': 2}).astype(int)


    return st_habits1