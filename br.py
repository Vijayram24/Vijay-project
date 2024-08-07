import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv(r"data.csv")
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'] )
df.drop("Unnamed: 32" , axis =1 , inplace = True)
corelation = df.corr()
a = st.sidebar.title("Menu")
b = st.sidebar.radio("Select" , ["Home" , "Prediction" , "About" , "Effect" , "Contect" ])
if b== "Home":
       st.image('lkqo17fe.png')
       st.title('Welcome learnear')
       st.header('Hear you can detect breast cancer')
       st.text('Frist 5 row of our test data')
       st.dataframe(df.head())
       st.text(f"shape of data is {df.shape}")
       st.text('Summary of our data')
       st.dataframe(df.describe())
       st.info('Countplot of diagnosis')
       fig =plt.figure(figsize = (40 , 10))
       sns.countplot(x=df['diagnosis'])
       st.pyplot(fig)
       st.subheader("Corelation between the featuers")
       st.dataframe(corelation)
       st.subheader('heatmap of corelation matrix')
       figu =plt.figure(figsize = (20  , 10))
       sns.heatmap(corelation , annot = True , cmap = 'coolwarm')
       plt.show()
       st.pyplot(figu)
elif b == "Prediction":
       label_mapping = dict(zip(le.classes_,df['diagnosis']))
       cor_target = abs(corelation["diagnosis"])    
       relevant_features = cor_target[cor_target>0.2]
       names = relevant_features.keys().tolist()
       names.remove('diagnosis'  )
       x= df[names]
       y = df['diagnosis']
       x_train , x_test , y_train , y_test = train_test_split(x , y , train_size = 0.8 , random_state = 0)
       model = RandomForestClassifier()
       model.fit(x_train , y_train)
       y_pred = model.predict(x_test)
       accuracy = accuracy_score(y_pred  , y_test)
       st.header("Detect the breast canser")
       data = {}
       for i in names:
          value= st.number_input(f"{i}" ) 
          data[i] = [value]
       a = pd.DataFrame(data)
       st.dataframe(a.head())
       bu = st.button("Prerdict")
       if bu:
         st.success("Predict value")
         a = model.predict(a)
         if a == 0 :
              st.write("Benign Tumors (Noncancerous)")
              st.success("You have not canser")
         else: 
              st.write("Malignant Tumors (Cancerous)")
              st.warning("You have Canser")
         st.markdown(f"Our Prediction is {accuracy * 100} % accuracte")
elif b == "About":
    st.header("About Breast Cancer")
    st.text('''Breast cancer is a type of cancer that originates in breast tissue. It occurs when
abnormal cells in the breast begin to grow uncontrollably, forming a tumor Although 
it’s more common in women, men can also develop breast cancer12. Early symptoms may 
include a new lump in the breast or underarm, changes in breast shape or appearance,
nipple discharge, skin texture changes, and breast pain. If you’re experiencing any 
concerning symptoms, I recommend seeking medical advice promptly. Regular screenings
and healthy lifestyle choices can help with prevention and early detection. Let me 
know if you need more information! 😊 ''')
elif b == "Effect" : 
    st.header("Effect of Breast cancer")
    st.text('''The effects of breast cancer can vary depending on the stage of the disease, 
treatment options, and individual factors. Here are some key points:

Physical Effects:
Tumor Growth: Breast cancer tumors can grow and invade nearby tissues,
affecting breast shape, size, and texture.
Pain and Discomfort: Tumors may cause pain, tenderness, or discomfort
in the breast or underarm area.
Lymph Node Involvement: Cancer cells can spread to nearby lymph nodes,
leading to swelling and changes in lymphatic drainage.
Metastasis: Advanced breast cancer can spread to distant organs (such
as bones, lungs, liver, or brain), causing additional symptoms.
Emotional and Psychological Effects:
Anxiety and Fear: A breast cancer diagnosis can be emotionally distressing,
leading to anxiety, fear, and uncertainty.
Depression: Coping with the disease, treatment side effects, and lifestyle
changes can contribute to depression.
Body Image Issues: Surgery (such as mastectomy) and changes in appearance may
impact self-esteem and body image.
Treatment-Related Effects:
Surgery: Surgical procedures (like lumpectomy or mastectomy) can have physical
and emotional effects.
Chemotherapy: Side effects include hair loss, fatigue, nausea, and changes in 
blood cell counts.
Radiation Therapy: Skin changes, fatigue, and localized discomfort are common.
Hormone Therapy: Used for hormone receptor-positive breast cancer, it may cause
menopausal symptoms.
Targeted Therapies: These drugs target specific proteins involved in cancer growth.
Long-Term Impact:
Survivorship: Many breast cancer survivors experience long-term effects related to 
treatment, including fatigue, cognitive changes, and joint pain.
Risk of Recurrence: Regular follow-up is essential to monitor for recurrence or 
new tumors.''')
else:
    a = st.text_input('enter your name')
    if a:
        st.title(f"Hello {a} Welcome")