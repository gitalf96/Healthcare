import pandas as pd
import numpy as np 
##import matplotlib as mp 
##import seaborn as sns 
import streamlit as st
from streamlit_option_menu import option_menu 
#import datetime
# import altair as alt 
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pycaret
from pycaret.classification import *
# from ydata_profiling import ProfileReport

st.set_page_config(page_title="Healthcare Dashboard",layout='wide')
st.title(':red[Healthcare Analysis and Result Prediction]')

st.markdown(
    """
    <style>
    div[data-testid="stApp"]  {
        background-color: rgba(0,0,0, 0.9);
            }
   </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] 
    div[class="st-emotion-cache-u5opgr eczjsme11"]{
    background-image: linear-gradient(#8993ab,#8993ab); 
    color: white
    }
    </style>
    
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
div[data-testid="column"] {
   background-color: rgba(0,0,0, 0.9);
   border: 3px solid rgba(64,224,208,0.9);
   padding: 3% 2% 3% 3%;
   border-radius:4px;
   color: rgb((255,0,0));
   overflow-wrap: break-word;
}

/* breakline for metric text         */
div[data-testid="element-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: red;
}
</style>
"""
, unsafe_allow_html=True)

dt=pd.read_csv('healthcare_dataset.csv')
dt.head()
dt.info()
dt.isnull().sum()
dt.duplicated().sum()
# dt.sample(5).T

# s = dt.groupby("medical_condition")["blood_type"].count()



dt.describe()

new_names = {'Name': 'name', 'Age': 'age', 'Gender': 'gender',
             'Blood Type': 'blood_type', 'Medical Condition': 'medical_condition',
             'Date of Admission': 'date_of_admission',
             'Doctor': 'doctor', 'Hospital': 'hospital', 
            'Insurance Provider': 'insurance', 'Billing Amount': 'bill',
             'Room Number': 'room', 'Admission Type': 'admission_type',
             'Discharge Date': 'discharge_date', 'Medication': 'medication',
             'Test Results': 'test_results', 'BMI': 'BMI'}
dt.rename(columns=new_names, inplace=True)
dt.info()

dt['date_of_admission'] = pd.to_datetime(dt['date_of_admission'])
# dt.date_of_admission = dt.date_of_admission.astype('datetime') 
dt['admit_year'] = dt['date_of_admission'].dt.year
# dt.head()

# for date in ['discharge_date']:
#  dt[date]=pd.to_datetime(dt[date])
       
# dt['stay_time'] = dt['discharge_date'] - dt['date_of_admission']

# ProfileReport(dt)

category_AGE=[]

for i in dt['age']:
    if i >= 18 and i<=30:
        category_AGE.append('Young')
    elif i>30 and i <=50:
        category_AGE.append('Middle Age')
    else:
        category_AGE.append('Old')

dt['Age_Category']=category_AGE
# dt.describe()
# dt['Age_Category'].value_counts()

nome=[]
for i in dt['name']:
    if len(i)>1:
        nome.append(1)
    else:
        nome.apppend(1)
dt['nome']=nome

# dt.info()
# dt['nome'].head()
# dt.columns
dt.rename(columns={'nome': 'patients'}, inplace=True)

with st.sidebar:
    selected=option_menu(
        menu_title="Menu",
        options=["Dashboard","Prediction"],
        icons=["bar-chart","graph-up"],
        menu_icon="house",
        default_index=0,
    )
    
  
if selected=="Dashboard":

        selected=option_menu(
        menu_title="Medical Condition",
        options=["Diabetes","Asthma","Obesity","Arthritis","Hypertension","Cancer"],
        icons=["heart-pulse-fill","heart-pulse-fill","heart-pulse-fill","heart-pulse-fill","heart-pulse-fill","heart-pulse-fill"],
        menu_icon="thermometer-high",
        default_index=0,
        orientation="horizontal"
        )

# with st.horizontal:
#   player_filter = st.selectbox("Select Medical Condition", pd.unique(dt['Medical Condition']))

        placeholder = st.empty()

        with placeholder.container():

            d1,d2,d3=st.columns(3) 
            if selected=="Diabetes":
                d1.metric(label="No.of Patients:",
                value=int(dt['medical_condition'].value_counts()['Diabetes']))
            if selected=="Asthma":
                d1.metric(label="No.of Patients:",
                value=int(dt['medical_condition'].value_counts()['Asthma']))
            if selected=="Obesity":
                d1.metric(label="No.of Patients:",
                value=int(dt['medical_condition'].value_counts()['Obesity']))
            if selected=="Arthritis":
                d1.metric(label="No.of Patients:",
                value=int(dt['medical_condition'].value_counts()['Arthritis']))
            if selected=="Hypertension":
                d1.metric(label="No.of Patients:",
                value=int(dt['medical_condition'].value_counts()['Hypertension']))
            if selected=="Cancer":
                d1.metric(label="No.of Patients:",
                value=int(dt['medical_condition'].value_counts()['Cancer']))

            if selected=="Diabetes":    
                d3.metric(label="Most Diagonised gender",
                value=dt.query("medical_condition==['Diabetes']")["gender"].value_counts().idxmax())
            if selected=="Asthma":
                d3.metric(label="Most Diagonised gender",
                value=dt.query("medical_condition==['Asthma']")["gender"].value_counts().idxmax())
            if selected=="Obesity":
               d3.metric(label="Most Diagonised gender",
               value=dt.query("medical_condition==['Obesity']")["gender"].value_counts().idxmax())
            if selected=="Arthritis":
               d3.metric(label="Most Diagonised gender",
               value=dt.query("medical_condition==['Arthritis']")["gender"].value_counts().idxmax())
            if selected=="Hypertension":
                d3.metric(label="Most Diagonised gender",
                value=dt.query("medical_condition==['Hypertension']")["gender"].value_counts().idxmax())
            if selected=="Cancer":
               d3.metric(label="Most Diagonised gender",
               value=dt.query("medical_condition==['Cancer']")["gender"].value_counts().idxmax())


            if selected=="Diabetes":    
                d2.metric(label="Most Cases in the year",
                value=dt.query("medical_condition==['Diabetes']")["admit_year"].mode())
            if selected=="Asthma":
                d2.metric(label="Most Cases in the year",
                value=dt.query("medical_condition==['Asthma']")["admit_year"].mode())
            if selected=="Obesity":
                d2.metric(label="Most Cases in the year",
                value=dt.query("medical_condition==['Obesity']")["admit_year"].mode())
            if selected=="Arthritis":
                d2.metric(label="Most Cases in the year",
                value=dt.query("medical_condition==['Arthritis']")["admit_year"].mode())
            if selected=="Hypertension":
                d2.metric(label="Most Cases in the year",
                value=dt.query("medical_condition==['Hypertension']")["admit_year"].mode())
            if selected=="Cancer":
                d2.metric(label="Most Cases in the year",
                value=dt.query("medical_condition==['Cancer']")["admit_year"].mode())

             
            with st.expander('Outcome of the Test Results based on Medications'):
                # column1,column2= st.columns(2)

            # Allow the user to select a gender.
            
                selected_medication = st.radio('Select Medical Condition:', dt.medication.unique(), index = 0)

                medicationresult = dt[dt['medication'] == selected_medication]

                    
            # # Allow the user to select a city.
            # select_gender = column1.selectbox('Select Gender', dt.sort_values('gender').gender.unique())

            # # Apply city filter
            # gender_medication  = medicationresult[medicationresult['gender'] == select_gender]

            # Use the city_gender_product dataframe as it has filters for gender and city.
            fig = px.histogram(medicationresult.sort_values('test_results') ,x='test_results', 
                               y='patients', color = 'test_results',)
            
            if selected_medication == 'Aspirin':
                st.write('Test results of patients treated with Aspirin!')
            elif selected_medication == 'Paracetamol':
                st.write('Test results of patients treated with Paracetamol!')
            elif selected_medication=='Ibuprofen':
                st.write('Test results of patients treated with Ibuprofen!')
            elif selected_medication=='Lipitor':
                st.write('Test results of patients treated with Lipitor!')
            elif selected_medication=='Penicillin':
                st.write('Test results of patients treated with Penicillin!')


            st.plotly_chart(fig, use_container_width=True) 

dt=dt.drop(columns=['doctor'])
dt=dt.drop(columns=['hospital'])
dt=dt.drop(columns=['insurance'])
dt=dt.drop(columns=['room'])
dt=dt.drop(columns=['admission_type'])
dt=dt.drop(columns=['bill'])
dt=dt.drop(columns=['name'])
dt=dt.drop(columns=['admit_year'])
dt=dt.drop(columns=['date_of_admission'])
dt=dt.drop(columns=['discharge_date'])

# dt.info()
dt=dt.apply(LabelEncoder().fit_transform)

# dt.head()
# dt['test_results'].value_counts()

# X=dt.drop(['test_results'],axis=1)
# y=dt['test_results']
# # from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# #Random Forest
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 1)
# model1 = RandomForestClassifier(n_estimators=100)
# model1.fit(X_train,y_train)
# predictions = model1.predict(X_test)
# accuracy=accuracy_score(y_test,predictions)
# accuracy

# #Decision Tree
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# clf = DecisionTreeClassifier()
# clf = clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# dt.head()
# tgt=dt.columns[-4]
# tgt

# Pycaret compare models
grid = setup(data=dt, target=dt.columns[-4], html=False, verbose=False)

best = compare_models()

if selected=="Prediction":

    st.write(best)




