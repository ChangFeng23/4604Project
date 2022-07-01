!pip install streamlit
!pip install pyngrok
!pip install plotly_express

%%writefile app.py
import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
#configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

#title of app
st.title("Data Visualization App")

#Add a sidebar
st.sidebar.subheader("Visualization Settings")

#Setup file upload
uploaded_file = st.sidebar.file_uploader(label="Upload files",
                         type=['csv','xlsx'])

global df
if uploaded_file is not None:
    print(uploaded_file)
    print("hello")
    try:
      df = pd.read_csv(uploaded_file)
    except Exception as e:
      print(e)
      df = pd.read_excel(uploaded_file)




try:
  st.write(df)
  numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
  string_columns = list(df.select_dtypes(['object']).columns)
  yes = list(df.select_dtypes(['object', 'float', 'int']).columns)
except Exception as e:
  print(e)
  st.write("please upload file to the application")
  
df1 = df.drop(columns=['MMSE', 'GDS-15'])
GDS = df[['GDS-15']]
MMSE =df[['MMSE']]


categorical = ['STATE',\
               'GENDER',\
               'Marital Status',\
               'Living',\
               'Smoking',\
               'Drinking Alcohol',\
               'Job Sector Previously',\
                #'GDS-15',\
                #'MMSE',\
               'Average Total Neighbourhood',\
               'Employment Status',\
               'MOSSF(Informational)',\
               'MOSSF (Tangible Support)',\
               'MOSSF (Affective Support)',\
               'MOSSF (Positive Social Interaction)',\
               'Neighbourhood - General Feel',\
               'Total SWLS',\
               'Total_EpQ(Data_Full)(Average)',\
               'WHODAS_baseline',\
               'sumLubben',\
               'ADL',\
               'Total Social Cohesion Scale ',\
               'Total_Loneliness '
               
               ]



df = pd.get_dummies(df[categorical])




st.title('GDS and MMSE Prediction')
st.write("""
# Explore different classifier
""")
dataset_name = st.sidebar.selectbox("Select Dataset", ("GDS-15", "MMSE"))
st.write(dataset_name)
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "Decision Tree", "Naive Bayes"))
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

def get_dataset(dataset_name):
    if dataset_name == "GDS-15":
        target = GDS
    elif dataset_name == "MMSE":
        target = MMSE
    
    x = df
    y = target
    return x, y
X,y = get_dataset(dataset_name)
st.write("Shape of dataset", X.shape)
st.write("Number of Classes", len(np.unique(y)))

def add_parameter_ui(clp_name):
    params = dict()
    if clp_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clp_name == "Naive Bayes":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        #n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params['max_depth'] = max_depth
        #params['n_estimators'] = n_estimators
    elif clp_name == "Decision Tree":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        #n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params['max_depth'] = max_depth
        #params['n_estimators'] = n_estimators
    
    

    return params
params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "Naive Bayes":
         clf = GaussianNB()
    else:
        clf= DecisionTreeClassifier(max_depth=params["max_depth"], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

#Classfication
X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=split_size, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier  = {classifier_name}")
st.write(f"accuracy ={acc}")
cm_dtc = confusion_matrix(y_test, y_pred)
st.write('Confusion matrix: ', cm_dtc)
# Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# plt.show from normal
st.pyplot(fig)

!ls

!streamlit run --server.port 80 app.py&>/dev/null&

from pyngrok import ngrok
# Setup a tunnel to the streamlit port 8501
public_url = ngrok.connect(port='8501')
public_url

!pgrep streamlit

!ps -eaf | grep streamlit

ngrok.kill()