import streamlit as st
import pickle
from PIL import Image
from streamlit_option_menu import option_menu
def project():
    selected=option_menu(
        menu_title=None,
    )
    st.title(":black[Patient Survival Prediction] ")
    image=Image.open("patient_589302497_1000.jpg")
    st.image(image,width=600)
    model=pickle.load(open('project.sav','rb'))
    scaler=pickle.load(open('minmaxscaler.sav','rb'))
    gcs= st.text_input("GCS Motor Response Value")
    gcse=st.text_input("GCS Eyes Value")
    gcsv = st.text_input("GCS Verbal Value")
    sysbp=st.text_input("Sysbolic Blood Pressure")
    spo2=st.text_input("Saturation of Peripheral Oxygen")
    sysbpn=st.text_input("Non Invasive Blood Pressure")
    tempmin=st.text_input("Minimum Temperature")
    mbp=st.text_input("Mininmum Mean Blood Pressure")
    mbpnon=st.text_input("Non Invasive Mean Blood Pressure")
    diasbpmin=st.text_input("Minimum Diastolic blood pressure")
    diasbp_noninvasive_min=st.text_input("Non Invasive Diastolic Blood Pressure")
    temp_apache=st.text_input("Temperature by APACHE standards")
    heartrate=st.text_input("Heart Rate")
    intubated=st.text_input("Intubated APACHE score")
    ventilated=st.text_input("Ventilated APACHE score")
    icu=st.text_input("ICU Death Probability")
    hosp=st.text_input("Hospital Death Probability")
    pred=st.button("Predict")
    if pred:
        prediction=model.predict(scaler.transform([[float(gcs),float(gcse),float(gcsv),float(sysbp),float(spo2),float(sysbpn),float(tempmin),float(mbp),float(mbpnon),float(diasbpmin),float(diasbp_noninvasive_min),float(temp_apache),float(heartrate),float(intubated),float(ventilated),float(icu),float(hosp)]]))
        if prediction==0:
            st.write("## Patient Will Survive")
        else:
            st.write("## Pstient Will Not Survive")
project()

