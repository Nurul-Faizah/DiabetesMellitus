#Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from urllib import parse
from urllib.request import urlopen



#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier

github_url = f"https://github.com/Nurul-Faizah/DiabetesMellitus"
kaggle_url = f"https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"

page_bg_img = """
<style>
[data-testid = "stAppViewContainer"]{
    background-image: url("https://media.istockphoto.com/id/1332097112/photo/the-black-and-silver-are-light-gray-with-white-the-gradient-is-the-surface-with-templates.jpg?b=1&s=170667a&w=0&k=20&c=uzejUTDUlhurH8GcYgKElOMBFkK84FEp9BIzzmncGWo=");
    background-size: cover;
}
[data-testid = "stHeader"]{
    background-color:rgba(0,0,0,0);
}
[data-testid = "stAToolbar"]{
    right: 2rem;
}
[data-testid = "stSidebar"]{
    background-image:url("https://images.unsplash.com/photo-1661667592582-feb71ae595e1?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTExfHxzd2VldCUyMGRhcmt8ZW58MHx8MHx8&auto=format&fit=crop&w=500&q=60");
    background-position: center;
}
<style>
"""
st.sidebar.write("""
            # Diabetes Mellitus"""
            )
st.sidebar.write("""
            #### Diabetes melitus (DM) didefinisikan sebagai suatu penyakit atau gangguan metabolisme kronis dengan multi etiologi yang ditandai dengan tingginya kadar gula darah disertai dengan gangguan metabolisme karbohidrat, lipid, dan protein sebagai akibat insufisiensi fungsi insulin.Insufisiensi fungsi insulin dapat disebabkan oleh gangguan atau defisiensi produksi insulin oleh sel-sel beta Langerhans kelenjar pankreas,atau disebabkan oleh kurang responsifnya sel-sel tubuh terhadap insulin (WHO, 1999).
            """)


st.markdown(page_bg_img,unsafe_allow_html=True)
st.title ("Web Apps - Classification Of Diabetes Mellitus")
st.write(f"Dataset yang digunakan adalah dataset pima indians diabetes yang diambil dari situs kaggle [DATASET DIABETES]({kaggle_url}). Dataset ini memiliki 8 parameter dan 1 parameter untuk 2 class diabetes. Prediksi diabetes mellitus pada web ini di khususkan untuk wanita. Anda dapat melihat dataset dan sourcode di repository saya [GITHUB]({github_url}).")

tab_titles = [
    "Accuracy Chart",
    "Implementation",]

tabs = st.tabs(tab_titles)

with tabs[0]:
    df = pd.read_csv('https://raw.githubusercontent.com/Nurul-Faizah/Dataset/main/pima_indians_diabetes.csv')

    #Data cleaning
    zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for column in zero_not_accepted :
        df[column] = df[column].replace(0,np.NaN)
        mean = int (df[column].mean(skipna=True))
        df[column] = df[column].replace(np.NaN,mean)

    #separate target values
    y = df['Outcome'].values


    X=df.iloc[:,0:8].values 
    y=df.iloc[:,8].values

    st.write('Jumlah baris dan kolom :', X.shape)
    st.write('Jumlah kelas : ', len(np.unique(y)))

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    #split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    st.write("Data Training :", X_train.shape)
    st.write("Data Testing :", X_test.shape)

    #KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test) 
    accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_knn = round(knn.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for Naive Bayes\n',cm)
    print('accuracy_Naive Bayes: %.3f' %accuracy)
    print('precision_Naive Bayes: %.3f' %precision)
    print('recall_Naive Bayes: %.3f' %recall)
    print('f1-score_Naive Bayes : %.3f' %f1)

    #NAIVE BAYES
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    Y_pred = gaussian.predict(X_test) 
    accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for Naive Bayes\n',cm)
    print('accuracy_Naive Bayes: %.3f' %accuracy)
    print('precision_Naive Bayes: %.3f' %precision)
    print('recall_Naive Bayes: %.3f' %recall)
    print('f1-score_Naive Bayes : %.3f' %f1)

    #DECISION TREE
    decision_tree = DecisionTreeClassifier() 
    decision_tree.fit(X_train, y_train)  
    Y_pred = decision_tree.predict(X_test) 
    accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for DecisionTree\n',cm)
    print('accuracy_DecisionTree: %.3f' %accuracy)
    print('precision_DecisionTree: %.3f' %precision)
    print('recall_DecisionTree: %.3f' %recall)
    print('f1-score_DecisionTree : %.3f' %f1)

    #ENSEMBLE BAGGING
    ensemble_bagging = BaggingClassifier() 
    ensemble_bagging.fit(X_train, y_train)  
    Y_pred = ensemble_bagging.predict(X_test) 
    accuracy_bg=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_ensemble_bagging = round(ensemble_bagging.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for DecisionTree\n',cm)
    print('accuracy_DecisionTree: %.3f' %accuracy)
    print('precision_DecisionTree: %.3f' %precision)
    print('recall_DecisionTree: %.3f' %recall)
    print('f1-score_DecisionTree : %.3f' %f1)

    st.write("""
                #### Akurasi:"""
                )

    results = pd.DataFrame({
        'Model': ['K-Nearest Neighbor','Naive Bayes','Decision Tree','Ensemble Bagging'],
        'Score': [ acc_knn,acc_gaussian,acc_decision_tree, acc_ensemble_bagging ],
        "Accuracy_score":[accuracy_knn,accuracy_nb,accuracy_dt,accuracy_bg
                        ]})
    result_df = results.sort_values(by='Accuracy_score', ascending=False)
    result_df = result_df.reset_index(drop=True)
    result_df.head(9)
    st.write(result_df)

    fig = plt.figure()
    fig.patch.set_facecolor('silver')
    fig.patch.set_alpha(0.7)
    ax = fig.add_axes([0,0,1,1])
    ax.patch.set_facecolor('silver')
    ax.patch.set_alpha(0.5)
    ax.plot(['K-Nearest Neighbor','Naive Bayes','Decision Tree','Ensemble Bagging'],[accuracy_knn,accuracy_nb,accuracy_dt,accuracy_bg],color='red')
    plt.show()
    st.pyplot(fig)


with tabs[1]:
    with st.expander("Role"):
        (st.write("""
            1. Pregnancies      : Number of times pregnant (mempresentasikan berapa kali wanita tersebut hamil selama hidupnya)
            2. Glucose          : Plasma glucose concentration a 2 hours in an oral glucose tolerance test (mempresentasikan konsentrasi glukosa plasma pada 2 jam dalam tes toleransi glukosa dengan satuan mg/dL miligram per desiliter)  normalnya kadar gula dalam darah berkisar antara 70-130 miligram/desiliter..
            3. Blood Pressure   : Diastolic blood pressure (Tekanan darah adalah cara yang sangat terkenal untuk mengukur kesehatan jantung seseorang, ada juga ukuran tekanan darah yaitu diastolik dan sistolik. Dalam data ini, kita memiliki tekanan darah diastolik dalam (mm / Hg) milimeter hydrargyrum ketika jantung rileks setelah kontraksi) Angka tekanan darah normal untuk sistolik tidak lebih dari 120 mmHg dan diastolik di bawah 80 mmHg..
            4. Skin Thickness   : Triceps skin fold thickness (nilai yang digunakan untuk memperkirakan lemak tubuh (mm) milimeter yang diukur pada lengan kanan setengah antara proses olecranon dari siku dan proses akromial skapula). 
            5. Insulin          : 2-Hour serum insulin (tingkat insulin 2 jam insulin serum dalam satuan U/ml Unit per Mililiter)
            6. BMI              : Body mass index (Indeks Massa Tubuh (berat dalam kilogram(kg / tinggi dalam meter kuadrat), dan merupakan indikator kesehatan seseorang).
            7. Diabetes Pedigree Function: Diabetes pedigree function (indikator riwayat diabetes dalam keluarga) 
            8. Age              : umur wanita suku indian pima (years) 
            9. Outcome          : Class variable (0 or 1), 0 untuk negatif mengidap diabetes, dan 1 positif mengidap diabetes.
        """))
    col1,col2 = st.columns([2,2])
    model=st.selectbox(
            'Model', ('K-Nearest Neighbor','Naive Bayes','Decision Tree','Ensemble Bagging'))
    with col1:
        a = st.number_input("Pregnancies",0)
        b = st.number_input("Glucose (mg/dL)",0)
        c = st.number_input("Blood Pressure (mm / Hg)",0)
        d = st.number_input("Skin Thickness (mm)",0)
    with col2:
        e = st.number_input("Insulin (U/ml)",0)
        f = st.number_input("BMI (kg)",0.00)
        g = st.number_input("Diabetes Pedigree Function",0.00)
        h = st.number_input("Age (years)",0)
    submit = st.button('Prediction')

    if submit:
        if model == 'K-Nearest Neighbor':
            X_new = np.array([[a,b,c,d,e,f,g,h]])
            predict = knn.predict(X_new)
            if predict == 1 :
                st.write("""# Negative Diabetes Mellitus""")
            else : 
                st.write("""# Positive Diabetes Mellitus""")

        elif model == 'Naive Bayes':
            X_new = np.array([[a,b,c,d,e,f,g,h]])
            predict = gaussian.predict(X_new)
            if predict == 1 :
                st.write("""# Negative Diabetes Mellitus""")
            else : 
                st.write("""# Positive Diabetes Mellitus""")

        elif model == 'Decision Tree':
            X_new = np.array([[a,b,c,d,e,f,g,h]])
            predict = decision_tree.predict(X_new)
            if predict == 1 :
                st.write("""# Negative Diabetes Mellitus""")
            else : 
                st.write("""# Positive Diabetes Mellitus""")

        else:
            X_new = np.array([[a,b,c,d,e,f,g,h]])
            predict = ensemble_bagging.predict(X_new)
            if predict == 1 :
                st.write("""# Negative Diabetes Mellitus""")
            else : 
                st.write("""# Positive Diabetes Mellitus""")


#deskripsi data berisi penjelasan mengenai parameternya (sudah)
#membuat beberapa model (knn,naive bayes,pohon keputusan,bagging) menggunakan 2 tab untuk model (sudah)
# dan implementasinya menggunakan model dengan akurasi tertinggi (sudah)
#link data, source code ditaruh di github,link github repository (sudah)
#dataset diabetes mellitus (sudah)
        