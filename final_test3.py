import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from streamlit_folium import folium_static
import folium

import warnings
warnings.filterwarnings("ignore")

@st.cache 
def my_prediction(encoded_month,encoded_flat_type,encoded_storey_range,encoded_town,encoded_flat_model,selected_fa,selected_lease):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OrdinalEncoder 
    from sklearn.preprocessing import OneHotEncoder 
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_absolute_error
    from sklearn.linear_model import Ridge
    data = pd.read_csv('https://raw.githubusercontent.com/euniceczy/ee4802_assignment1/master/resale_flat_prices_based_on_registration_date_from_jan_2017_onwards.csv')
    trim_data = data.drop(["block", "street_name", "remaining_lease"], axis=1) 
    trim_data.head(5)
    pipeline = ColumnTransformer([ ("o", OrdinalEncoder(), ["month","flat_type","storey_range"]), ("n", OneHotEncoder(), ["town", "flat_model"]), ], remainder='passthrough')

    X = pipeline.fit_transform(trim_data.drop(["resale_price"], axis=1))
    y = trim_data["resale_price"]

    col_name = ["month","type","storey","AMK","BED","BIS","BBT","BMH","BPJ",
    "BTM","CEN","CCK","CLE","GEY","HOU","JRE","JRW","KAL","MAR",
    "PAS","PUN","QUE","SEM","SKG","SER","TAM","TOA","WOO",
    "YIS","2-room","Adjoined","Apartment","DBSS","Improved",
    "Improved-M","Maisonette","Model A","Model A-M","Model A2",
    "Multi Gen","New Gen","Premium Apt","Premium Apt Loft",
    "Premium M","Simplified","Standard","Terrace","Type S1",
    "Type S2","Area","Lease"]
    df_X = pd.DataFrame(X.toarray(),columns=col_name)
    df_Xy = df_X.assign(resale_price = y)
    df_y = df_Xy.resale_price

    mask = np.random.rand(len(df_Xy)) <= 0.8
    X_train = df_X[mask]
    X_test = df_X[~mask]

    y_train = df_y[mask]
    y_test = df_y[~mask]

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)

    pr = LinearRegression().fit(X_poly, y_train)

    X_inp = [[0 for x in range(51)]]
    X_inp[0][0] = encoded_month 
    X_inp[0][1] = encoded_flat_type 
    X_inp[0][2] = encoded_storey_range 
    X_inp[0][encoded_town] = 1 
    X_inp[0][encoded_flat_model] = 1
    X_inp[0][49] = selected_fa
    X_inp[0][50] = selected_lease 

    test= pd.DataFrame(X_inp, columns = col_name)
    X_test_poly = poly_features.fit_transform(X_test)
    X_input = poly_features.fit_transform(test)

    rr = Ridge(alpha=0.001).fit(X_poly,y_train)
    pred_rr = rr.predict(X_input)
    return pred_rr[0]

data = pd.read_csv('https://raw.githubusercontent.com/euniceczy/ee4802_assignment1/master/resale_flat_prices_based_on_registration_date_from_jan_2017_onwards.csv')

#TITLE
st.title("EE4802 Assignment 1 HDB Price Prediction")

##############USER INPUT
#SELECTED SELLING YY-MM
selected_yymm = st.text_input('Type in Selling Month in YYYY-MM format between 2017-01 to 2022-03','2017-01')
st.write('Selected Time Period is', selected_yymm)

#SELECTED TOWN
towns = data.town.unique()
selected_town = st.selectbox("Select Town for prediction", towns)
st.write('Selected Town is ', selected_town)

############## MAP OF TOWN
if selected_town=='ANG MO KIO':
    lat, lon = 1.3691, 103.8454
elif selected_town == 'BISHAN':
    lat, lon = 1.3526, 103.8352
elif selected_town == 'BEDOK':
    lat, lon = 1.3236, 103.9273
elif selected_town == 'BUKIT BATOK':
    lat, lon = 1.3590, 103.7637
elif selected_town == 'BUKIT MERAH':
    lat, lon = 1.2819, 103.8239
elif selected_town == 'BUKIT PANJANG':
    lat, lon = 1.3774, 103.7719
elif selected_town =='BUKIT TIMAH': 
    lat, lon = 1.3294, 103.8021
elif selected_town =='CENTRAL AREA':
    lat, lon = 1.3048, 103.8318
elif selected_town == 'CHOA CHU KANG':
    lat, lon = 1.3840, 103.7470
elif selected_town == 'CLEMENTI':
    lat, lon = 1.3162, 103.7649
elif selected_town == 'GEYLANG':
    lat, lon = 1.3201, 103.8918
elif selected_town =='HOUGANG':
    lat, lon = 1.3612, 103.8863
elif selected_town =='JURONG EAST':
    lat, lon = 1.3329, 103.7436
elif selected_town =='JURONG WEST': 
    lat, lon = 1.3404, 103.7090
elif selected_town =='KALLANG/WHAMPOA':
    lat, lon = 1.3100, 103.8651
elif selected_town == 'MARINE PARADE':
    lat, lon =1.3020, 103.8971
elif selected_town == 'PASIR RIS':
    lat, lon =1.3721,103.9474
elif selected_town == 'PUNGGOL':
    lat, lon =1.3984, 103.9072
elif selected_town == 'QUEENSTOWN':
    lat, lon = 1.2942, 103.7861
elif selected_town == 'SEMBAWANG':
    lat, lon =1.4491,103.8185
elif selected_town == 'SENGKANG':
    lat, lon =1.3868, 103.8914
elif selected_town == 'SERANGOON':
    lat, lon =1.3554, 103.8679
elif selected_town =='TAMPINES':
    lat, lon =1.3496,103.9568
elif selected_town =='TOA PAYOH':
    lat, lon =1.3343, 103.8563
elif selected_town == 'WOODLANDS':
    lat, lon =1.4382, 103.7890
elif selected_town == 'YISHUN':
    lat, lon =1.4304, 103.8354

radius = 3000
m = folium.Map(location=[lat, lon], zoom_start=12)
folium.Marker([lat, lon]).add_to(m)
folium.Circle([lat, lon], radius=radius).add_to(m)  # radius is in meters
folium_static(m)
st.write('*Map shown is an estimated radius of the Town selected*')

#SELECTED FA
# selected_fa = st.number_input('Floor Area in Sqm',100.00)
min_fa=data["floor_area_sqm"].min()
max_fa=data["floor_area_sqm"].max()
selected_fa = st.slider("Select Floor Area (sqm) for prediction ", int(min_fa),int(max_fa),100) # floor area
st.write('Selected Floor Area in SQM is ', selected_fa)

#SELECTED FLAT TYPE
flat_type = data.flat_type.unique()
selected_flat_type = st.selectbox("Select Flat Type for prediction", flat_type)
st.write('Selected Flat Type is ', selected_flat_type)

#SELECTED FLAT MODEL
flat_model = data.flat_model.unique()
selected_flat_model = st.selectbox("Select Flat Model for prediction", flat_model)
st.write('Selected Flat Model is ', selected_flat_model)

#SELECTED STOREY RANGE
storey_range = data.storey_range.unique()
storey_range.sort()
selected_storey_range = st.selectbox("Select Flat Type for prediction", storey_range)
st.write('Selected Storey Range is ', selected_storey_range)

#SELECTED LEASE COMMENCEMENT DATE
lease_commencement = data.lease_commence_date.unique()
lease_commencement.sort()
selected_lease = st.select_slider('Select Lease Commencement Year', options=lease_commencement)
st.write('Selected Lease Year is ', selected_lease)

###############ENCODER
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

#ORDINAL ENCODER FOR YYMM
encoder = OrdinalEncoder()
encoder.fit_transform(data[["month"]])
encoded_month = encoder.transform([[selected_yymm]])
encoded_month = int(encoded_month[0][0])

#ORDINAL ENCODER FOR STOREY RANGE
encoder = OrdinalEncoder()
encoder.fit_transform(data[["storey_range"]])
encoded_storey_range = encoder.transform([[selected_storey_range]])
encoded_storey_range = int(encoded_storey_range[0][0])

#ORDINAL ENCODER FOR FLAT TYPE
encoder = OrdinalEncoder()
encoder.fit_transform(data[["flat_type"]])
encoded_flat_type = encoder.transform([[selected_flat_type]])
encoded_flat_type = int(encoded_flat_type[0][0])

#ORDINAL ENCODER FOR TOWN
encoder = OrdinalEncoder()
encoder.fit_transform(data[["town"]])
encoded_town = encoder.transform([[selected_town]])
encoded_town=int(encoded_town[0][0])+3

#ORDINAL ENCODER FOR FLAT MODEL
encoder = OrdinalEncoder()
encoder.fit_transform(data[["flat_model"]])
encoded_flat_model = encoder.transform([[selected_flat_model]])
encoded_flat_model=int(encoded_flat_model[0][0])+29

#BUTTON TO CONFIRM
predict_button=0
st.write('Predict HDB Resale Price for ', selected_yymm, "period, ", selected_town, " ", selected_fa, "SQM, ", selected_flat_type," ",selected_flat_model," ",selected_storey_range, "storey ", selected_lease, "lease?")
if st.button('Predict'):
    predict_button=1;
    

#############BUTTON PREDICT
if predict_button==1:
    estimated_price = my_prediction(encoded_month,encoded_flat_type,encoded_storey_range,encoded_town,encoded_flat_model,selected_fa,selected_lease)
    estimated_price = '{:,.2f}'.format(estimated_price)
    st.success('Success!')
    st.header('Predicted HDB Resale Price is **SGD$%s**' % estimated_price)
    predict_button=0