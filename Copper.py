import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
from streamlit_option_menu import option_menu

df_p= pd.read_csv("/content/Cu_Pre.csv")
df_c= pd.read_csv("/content/Cu_Class.csv")

status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
              '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
              '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
              '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
              '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

st.set_page_config(layout= "wide")
st.markdown("""
    <h1 style='text-align: center;'>Industrial Copper Modeling</h1>
    """, unsafe_allow_html=True)
st.write("")

st.markdown(f""" <style>.stApp {{
                        background:url("https://m.foolcdn.com/media/dubs/original_images/Industry_business_production_and_heavy_metallurgical_industrial_products_m.jpg");
                        background-size: cover}}
                     </style>""", unsafe_allow_html=True)

select = option_menu(
    menu_title = None,
    options = ["HOME","PREDICT SELLING PRICE","STATUS CLASSIFICATION"],
    icons =["house","bar-chart"],orientation="horizontal")
if select == "HOME":
  text_content = """
    <h2 style='text-align: center; color: white;'>Welcome to my Industrial Copper Modeling project</h2>
    <p style='font-size: 18px; text-align: center;'>In this project we revolutionize the copper industry through data-driven precision. In a landscape fraught with complexities, we address data quality issues in sales and pricing with advanced machine learning techniques like normalization, scaling, and outlier detection. Additionally, our lead classification model sifts through leads, identifying high-potential prospects to optimize resource allocation. By integrating these solutions into our Streamlit application, we empower stakeholders with actionable insights for informed decision-making and sustained growth.</p>
        """

  # Display the text using markdown function
  st.markdown(text_content, unsafe_allow_html=True)



if select == "PREDICT SELLING PRICE":
  with st.form("my_form"):
    col1,col2,col3=st.columns([5,2,5])
    with col1:
        st.write(' ')
        status = st.selectbox("Status", status_options,key=1)
        item_type = st.selectbox("Item Type", item_type_options,key=2)
        country = st.selectbox("Country", sorted(country_options),key=3)
        application = st.selectbox("Application", sorted(application_options),key=4)
        product_ref = st.selectbox("Product Reference", product,key=5)

    with col3:
        st.write( f'NOTE: Min & Max given for reference, you can enter any value', unsafe_allow_html=True )
        quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
        thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
        width = st.text_input("Enter width (Min:1, Max:2990)")
        customer = st.text_input("customer ID (Min:12458, Max:30408185)")
        submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
        st.markdown("""

        """, unsafe_allow_html=True)

        flag=0
        pattern = "^(?:\d+|\d*\.\d+)$"
        for i in [quantity_tons,thickness,width,customer]:
          if re.match(pattern, i):
            pass
          else:
            flag=1
            break

        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)

        if submit_button and flag==0:
          import pickle
          with open(r"/content/Linear.pkl", 'rb') as file:
              loaded_model = pickle.load(file)

              user_data =df_p[["thickness_log","width","country","selling_price","product_ref","application","item type","customer","quantity tons"]].values

              # Make predictions using the loaded model
              y_pred_user = loaded_model.predict(user_data)
          st.write('## :green[Predicted selling price:] ', np.exp(y_pred_user)[0])

if select == "STATUS CLASSIFICATION":
  with st.form("my_form1"):
    col1,col2,col3=st.columns([5,1,5])
    with col1:
        cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
        cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
        cwidth = st.text_input("Enter width (Min:1, Max:2990)")
        ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
        cselling = st.text_input("Selling Price (Min:1, Max:100001015)")


    with col3:
        st.write(' ')
        citem_type = st.selectbox("Item Type", item_type_options,key=21)
        ccountry = st.selectbox("Country", sorted(country_options),key=31)
        capplication = st.selectbox("Application", sorted(application_options),key=41)
        cproduct_ref = st.selectbox("Product Reference", product,key=51)
        csubmit_button = st.form_submit_button(label="PREDICT STATUS")
        st.markdown("""

        """, unsafe_allow_html=True)

    cflag=0
    pattern = "^(?:\d+|\d*\.\d+)$"
    for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:
        if re.match(pattern, k):
          pass
        else:
          cflag=1
          break

  if csubmit_button and cflag==1:
    if len(k)==0:
      st.write("please enter a valid number space not allowed")
    else:
      st.write("You have entered an invalid value: ",k)

  if csubmit_button and cflag==0:
    import pickle
    with open(r"/content/Linear.pkl", 'rb') as file:
      loaded_model = pickle.load(file)
      user_data =df_c[["thickness_log","width","country","selling_price","product_ref","application","item type","customer","quantity tons"]].values
      # Make predictions using the loaded model
      y_pred_user = loaded_model.predict(user_data)
      # Assuming '1' represents "Won" and '0' represents "Lose"
      if (y_pred_user == 1).any():
          st.write('## :green[The Status is Won]')

      else:
          st.write("## :red[The status is Lost] ")



