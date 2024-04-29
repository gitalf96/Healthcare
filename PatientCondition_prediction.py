import pandas as pd
# import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu 
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import lightgbm as lgb
# from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
# import warnings
# warnings.filterwarnings("ignore")



st.set_page_config(page_title="Healthcare Dashboard",layout='wide')
st.title(':red[Patient Condition Analysis and Prediction]')

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

df=pd.read_csv('healthcare_dataset.csv')

#Rename columns
new_names = {'NAME': 'name', 'AGE': 'age', 'GENDER': 'gender',
             'BLOOD_TYPE': 'blood_type', 'MEDICAL_CONDITION': 'medical_condition',
             'DATE_ADMISSION': 'date_of_admission',
             'DOCTOR': 'doctor', 'HOSPITAL': 'hospital', 
            'INSURANCE_PROVIDER': 'insurance', 'BILLING_AMOUNT': 'bill',
             'ROOM_NUMBER': 'room', 'ADMISSION_TYPE': 'admission_type',
             'DISCHARGE_DATE': 'discharge_date', 'MEDICATION': 'medication',
             'TEST_RESULTS': 'test_results', 'BMI': 'BMI'}
df.rename(columns=new_names, inplace=True)

df['date_of_admission'] = pd.to_datetime(df['date_of_admission'])
df['admit_year'] = df['date_of_admission'].dt.year

#Categorize Age
category_AGE=[]

for i in df['age']:
    if i >= 18 and i<=30:
        category_AGE.append('Young')
    elif i>30 and i <=50:
        category_AGE.append('Middle Age')
    else:
        category_AGE.append('Old')

df['Age_Category']=category_AGE

#Encoding PAtients
nome=[]
for i in df['name']:
    if len(i)>1:
        nome.append(1)
    else:
        nome.apppend(1)
df['nome']=nome
df.rename(columns={'nome': 'patients'}, inplace=True)

with st.sidebar:
    selected=option_menu(
        menu_title="Menu",
        options=["Dashboard","Prediction"],
        icons=["bar-chart","graph-up"],
        menu_icon="menu-down",
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

        placeholder = st.empty()

        with placeholder.container():

            d1,d2,d3=st.columns(3) 
            if selected=="Diabetes":
                d1.metric(label="No.of Patients:",
                value=int(df['medical_condition'].value_counts()['Diabetes']))
            if selected=="Asthma":
                d1.metric(label="No.of Patients:",
                value=int(df['medical_condition'].value_counts()['Asthma']))
            if selected=="Obesity":
                d1.metric(label="No.of Patients:",
                value=int(df['medical_condition'].value_counts()['Obesity']))
            if selected=="Arthritis":
                d1.metric(label="No.of Patients:",
                value=int(df['medical_condition'].value_counts()['Arthritis']))
            if selected=="Hypertension":
                d1.metric(label="No.of Patients:",
                value=int(df['medical_condition'].value_counts()['Hypertension']))
            if selected=="Cancer":
                d1.metric(label="No.of Patients:",
                value=int(df['medical_condition'].value_counts()['Cancer']))

            if selected=="Diabetes":    
                d3.metric(label="Most Diagonised gender",
                value=df.query("medical_condition==['Diabetes']")["gender"].value_counts().idxmax())
            if selected=="Asthma":
                d3.metric(label="Most Diagonised gender",
                value=df.query("medical_condition==['Asthma']")["gender"].value_counts().idxmax())
            if selected=="Obesity":
               d3.metric(label="Most Diagonised gender",
               value=df.query("medical_condition==['Obesity']")["gender"].value_counts().idxmax())
            if selected=="Arthritis":
               d3.metric(label="Most Diagonised gender",
               value=df.query("medical_condition==['Arthritis']")["gender"].value_counts().idxmax())
            if selected=="Hypertension":
                d3.metric(label="Most Diagonised gender",
                value=df.query("medical_condition==['Hypertension']")["gender"].value_counts().idxmax())
            if selected=="Cancer":
               d3.metric(label="Most Diagonised gender",
               value=df.query("medical_condition==['Cancer']")["gender"].value_counts().idxmax())

            if selected=="Diabetes":    
                d2.metric(label="Most Cases in the year",
                value=df.query("medical_condition==['Diabetes']")["admit_year"].mode())
            if selected=="Asthma":
                d2.metric(label="Most Cases in the year",
                value=df.query("medical_condition==['Asthma']")["admit_year"].mode())
            if selected=="Obesity":
                d2.metric(label="Most Cases in the year",
                value=df.query("medical_condition==['Obesity']")["admit_year"].mode())
            if selected=="Arthritis":
                d2.metric(label="Most Cases in the year",
                value=df.query("medical_condition==['Arthritis']")["admit_year"].mode())
            if selected=="Hypertension":
                d2.metric(label="Most Cases in the year",
                value=df.query("medical_condition==['Hypertension']")["admit_year"].mode())
            if selected=="Cancer":
                d2.metric(label="Most Cases in the year",
                value=df.query("medical_condition==['Cancer']")["admit_year"].mode())
            
            with st.expander('Outcome of the Test Results based on Medications'):
            
                selected_medication = st.radio('Select Medical Condition:', df.medication.unique(), index = 0)

                medicationresult = df[df['medication'] == selected_medication]

            fig = px.histogram(medicationresult.sort_values('test_results') ,x='test_results', 
                               y='patients', color = 'test_results',labels={
                                # notice here the use of _("") localization function
                                "test_results": ("Patient Condition")
                                #  "patients": ("Patient Count")
                                    },)
            
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

# dt.head()  

gen=[]
for i in df['gender']:
    if i=='Female':
        gen.append(1)
    else:
        gen.append(2)

df['gender']=gen

bm=[]
for i in df['BMI']:
    if i=='High':
        bm.append(3)
    elif (1)=='Medium':
        bm.append(2)
    else:
        bm.append(1)

df['BMI']=bm

med=[]
for i in df['medication']:
    if i=='Aspirin':
        med.append(1)
    elif i=='Lipitor':
        med.append(2)
    elif i=='Penicillin':
        med.append(3)
    elif i=='Paracetamol':
        med.append(4)
    else:
        med.append(5)
df['medication']=med

mc=[]
for i in df['medical_condition']:
    if i=='Diabetes':
        mc.append(1)
    elif i=='Asthma':
        mc.append(2)
    elif i=='Obesity':
        mc.append(3)
    elif i=='Arthritis':
        mc.append(4)
    elif i=='Hypertension':
        mc.append(5)
    elif i=='Cancer':
        mc.append(6)

df['medical_condition']=mc

tr=[]
for i in df['test_results']:
    if i=='Inconclusive':
        tr.append((1))
    elif i=='Normal':
        tr.append((2))
    else:
        tr.append(0)

df['test_results']=tr

bg=[]
for i in df['BlOOD_TYPE']:
    if i=='O-':
        bg.append(1)
    elif i=='O+':
        bg.append(2)
    elif i=='B-':
        bg.append(3)
    elif i=='AB+':
        bg.append(4)
    elif i=='A+':
        bg.append(5)
    elif i=='AB-':
        bg.append(6)
    elif i=='A-':
        bg.append(7) 
    elif i=='B+':
        bg.append(8)
df['BlOOD_TYPE']=bg
#dt.head()

if selected=="Prediction":

    with st.sidebar:
        selected=option_menu(
            menu_title="      ",
            options=["Single","Batch"],
            icons=["list","grid-3x3"],
            # menu_icon="graph-up",
            orientation="horizontal",
            default_index=0,
        )
    dt=df.drop(['doctor','hospital','insurance','room','admission_type','bill','name','date_of_admission',
                'discharge_date','Age_Category','patients','admit_year'],axis=1)       

    if selected=="Single":

        x=pd.DataFrame(dt)
        x=x.drop(columns=['test_results'],axis=1)
        y=dt['test_results']
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.22,random_state=42)
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        importance = clf.feature_importances_
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc=('{:.4f}'.format(accuracy))
        # Display F1 score
        F1=f1_score(y_test,y_pred,average='macro')
        f1=('{:.4f}'.format(F1))

        #lightgbm
        model = lgb.LGBMClassifier()
        model.fit(x_train,y_train)
        Y_pred=model.predict(x_test)
        Accu=accuracy_score(Y_pred,y_test)
        Acc=('{:.4f}'.format(Accu))
        eff1=f1_score(y_test,Y_pred,average='macro')
        ef1=('{:.4f}'.format(eff1))
        
    # Collect user input for new data
        new_data={}
    
        
        new_data['age']=st.number_input(':red[Enter age]')
        new_data['gender'] = st.selectbox(':red[Select gender]',('Female','Male'))
        new_data['BlOOD_TYPE']= st.selectbox(':red[Select BlOOD_TYPE]',('O-','O+','B-','AB+','A+','AB-','A-','B+'))
        new_data['medical_condition']=st.selectbox(':red[Select medical_condition]',('Diabetes','Asthma','Obesity','Arthritis','Hypertension','Cancer'))
        new_data['medication']=st.selectbox(':red[Select medication]',('Aspirin','Lipitor','Penicillin','Paracetamol','Ibuprofen'))
        new_data['BMI']=st.selectbox(':red[Select BMI]',('High','Medium','Low'))
        
        new_data=pd.DataFrame([new_data])
        # # plot feature importance
        # pyplot.bar([x for x in range(len(importance))], importance)
        # pyplot.show()

        # feat_importances = pd.Series(clf.feature_importances_, index=dt.columns)
        # feat_importances.nlargest(4).plot(kind='barh')

        # fig = px.histogram(medicationresult.sort_values('test_results') ,x='test_results', 
        #                        y='patients', color = 'test_results',labels={
        #                         # notice here the use of _("") localization function
        #                         "test_results": ("Patient Condition")
        #                         #  "patients": ("Patient Count")
        #                             },)

        new_df=new_data.apply(LabelEncoder().fit_transform)
        # st.write(impp)
        selected=option_menu(
            menu_title="Choose ML Model",
            options=["LightGBM","Random Forest"],
            icons=["bullseye","tree"],
            menu_icon="mouse",
            default_index=0,
            orientation="horizontal"
            )
        
        if selected=="LightGBM":

            b1,b2=st.columns(2) 
            b1.metric(label=':green[Accuracy: ]',
                    value=Acc)
            b2.metric(label=':green[F1 score: ]', value=ef1)
        

            prediction = model.predict(new_df)
            if st.button('Predict Patient condition'):
                for i in prediction:
                    if i==0:
                            st.write('The Patient condition is Abnormal')
                    elif i==1:
                            st.write('The Patient condition is Inconclusive') 
                    else:
                            st.write('The Patient condition is Normal') 

        if selected=="Random Forest":
            a1,a2=st.columns(2) 
            a1.metric(label=':green[Accuracy: ]',
                    value=acc)
            a2.metric(label=':green[F1 score: ]', value=f1)

            prediction = clf.predict(new_df)
            if st.button('Predict Patient condition'):
                for i in prediction:
                    if i==0:
                            st.subheader('**The Patient condition is Abnormal**')
                    elif i==1:
                            st.subheader('**The Patient condition is Inconclusive**') 
                    else:
                            st.subheader('**The Patient condition is Normal**') 


    if selected=="Batch":

        uploaded_file = st.file_uploader("Choose a file")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())

            dff=df.apply(LabelEncoder().fit_transform)
            accuracy=pd.DataFrame(df)
            xx=dff.drop(columns=['test_results'],axis=1)
            yy=dff['test_results']
            xx_train,xx_test,yy_train,yy_test = train_test_split(xx,yy,test_size=0.33,random_state=42)
            clff = RandomForestClassifier(n_estimators=100)
            clff.fit(xx_train, yy_train)
            yy_pred = clff.predict(xx_test)
            accuracy = accuracy_score(yy_test,yy_pred)


            #LGBM
            mod = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
            mod.fit(xx_train,yy_train,eval_set=[(xx_test,yy_test),(xx_train,yy_train)],
                    eval_metric='logloss')

            LGBM2=('Testing accuracy {:.4f}'.format(mod.score(xx_test,yy_test)))

            fin = pd.DataFrame(columns=['age','gender','BlOOD_TYPE','medical_condition','medication','BMI','test_results'])
            edit_da = xx
            table_da= df.drop(columns=['test_results'],axis=1)

            if st.button('Click to Predict'):
                # fin_da = pd.DataFrame(clff.predict(edit_da))
                patient_condition=clff.predict(edit_da)
                fin['patient_condition']=patient_condition
                fin_da=fin['patient_condition']
                fin = pd.concat([table_da,fin_da], axis=1)
                fin=pd.DataFrame(fin)
                pc=[]
                for i in fin['patient_condition']:
                    if i==1:
                        pc.append('Inconclusive')
                    elif i==2:
                        pc.append('Normal')
                    else:
                        pc.append('Abnormal')
                fin['patient_condition']=pc
            with st.expander('Predicted Patient Condition'):
                st.table(fin)

    
