import pandas as pd
import streamlit as st
import pickle
from PIL import Image
image = Image.open('car-3189771_1920_e.jpg')
knn_model = pickle.load(open('models/knn_search.sav','rb'))
#knn_model = pickle.load(open('models/knn_search_var_low.sav','rb'))
lm_model = pickle.load(open('models/lm_pipeline.sav','rb'))
dtree_model = pickle.load(open('models/dtree_search.sav','rb'))
X_test = pd.read_csv('test_data/test_data_X_test.csv', sep= ";", index_col=None, header=0, engine ='python')
y_test = pd.read_csv('test_data/test_data_y_test.csv', sep= ";", index_col=None, header=0, engine ='python')

def main():
    st.subheader("Top Branded Cars Are Not A Dream Anymore !")
    st.write("Price Prediction for Used Cars")
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")    
    st.write('Different models have been trained to predict the price of used cars based on certain features such as brand, body type, fuel type, power, displacement, fuel efficiency, etc.')
    
    
    col1, col2 = st.columns(2)
    with col1:
        load_test_data = st.radio('Data Source', ('Test Data', 'User Entry'))  
        #test_data_idx = st.slider('Test Data Num: ',0,3000,0, key = 'test_data')
    with col2:    
        test_data_idx = st.number_input('Test Data Num: ',0,3000,0, key = 'test_data')
        #col1, col2 = st.columns(2)
    
    with st.sidebar:
        if(load_test_data == 'User Entry'):
            st.markdown('**USER ENTRY**')
            car_brand = st.selectbox('Brand', ('mercedes-benz', 'bmw', 'ford', 'volkswagen', 'audi', 'renault', 'porsche', 'toyota', 'opel', 'skoda'), key = 'car_brand')
            body_type = st.selectbox('Type', ('Kombi', 'SUV', 'Van', 'Cabrio', 'Smallcar', 'Coupe', 'Limousine', 'Super95', 'Transporter', 'opel'), key = 'body_type')    
            condition = st.selectbox('Condition',('Used_cars', 'Annual_cars', 'Demonstration_cars','One_day_registered_cars'),key='car_cond')    
            num_seats = st.number_input('Num of Seats', 0, 7, 5, step=1, key ='seats')    
            mileage = st.number_input('Mileage (in kms)', 0, 500000, 10000, step=500, key ='mileage')    
            power_kW = st.slider('Power (in kW)', 30, 400, 80, key ='power')    
            transmission = st.selectbox('Transmission', ('Automatic', 'Manual', 'Semi-automatic'), key = 'transmission')    
            displacement = st.slider('Displacement',500,7000,1800,key='displacement')    
            fuel = st.selectbox('Fuel Type', ('Super95', 'Diesel_PF', 'Super95_91', 'Super95_PF', 'Super91E10_PF', 'Super95_E10_PF', 'Super95_E10_91', 'Super98', 'Super95_91_PF', 'Super95_E10', 'Erdgas', 'Diesel','Super98_E10_PF', 'Biodiesel', 'Lpg', 'Super98_E10', 'Super98_PF'), key = 'fuel')    
            color = st.select_slider('Color', options=['Black','Grey','White','Blue','Silver','Red','Green','Orange','Brown','Yellow','Beige','Gold','Bronze','Purple'])    
            seller = st.radio('Seller', ('Retailer', 'Private'), key='owner')    
            year_built = st.radio('Year of Model', ('2021','2020'), key='year')    
            num_owners = st.slider('Num of pre-owners', 0,5,1, key='owner')    
            co2_emission_gpkm = st.slider('CO2 Emission', 0,900,120, key='co2_emission')    
            fuel_eff_combi_lpkm = st.slider('Fuel Efficiency Combined (in lpkm)',0,50,6, key = 'fuel_combi')    
            fuel_eff_city_lpkm = st.slider('Fuel Efficienty City) (in lpkm)',0,50,6, key = 'fuel_city')    
            fuel_eff_highway_lpkm = st.slider('Fuel Efficienty highway (in lpkm)',0,50,6, key = 'fuel_highway')
            
        if(load_test_data == 'Test Data'):
            user_car = X_test.iloc[test_data_idx:test_data_idx+1]
            price_actual = y_test.iloc[test_data_idx].price
            st.header('TEST DATA : ' + str(test_data_idx))
            st.markdown('Brand       :**'+ user_car.iloc[0].car_brand+'**')
            st.markdown('Type        :**'+ user_car.iloc[0].body_type+'**') 
            st.markdown('Condition   :**'+ user_car.iloc[0].condition+'**') 
            st.markdown('Num. Seats  :**'+ str(user_car.iloc[0].num_seats)+'**') 
            st.markdown('Mileage     :**'+ str(user_car.iloc[0].mileage)+'**') 
            st.markdown('Power \(kW\):**'+ str(user_car.iloc[0].power_kW)+'**') 
            st.markdown('Transmission:**'+ user_car.iloc[0].transmission+'**') 
            st.markdown('Displacement:**'+ str(user_car.iloc[0].displacement)+'**') 
            st.markdown('Fuel Type   :**'+ user_car.iloc[0].fuel+'**')                
            st.markdown('Color       :**'+ user_car.iloc[0].color+'**')
            st.markdown('Seller      :**'+ user_car.iloc[0].seller+'**')
            st.markdown('Model Year  :**'+ str(user_car.iloc[0].year_built)+'**')
            st.markdown('Num. Owners :**'+ str(user_car.iloc[0].num_owners)+'**')
            st.markdown('CO2 Emission:**'+ str(user_car.iloc[0].co2_emission_gpkm)+'**')
            st.markdown('FuelEff Combined.lpkm:**'+ str(user_car.iloc[0].fuel_eff_combi_lpkm)+'**')
            st.markdown('FuelEff     City.lpkm:**'+ str(user_car.iloc[0].fuel_eff_city_lpkm)+'**')
            st.markdown('FuelEff  Highway.lpkm:**'+ str(user_car.iloc[0].fuel_eff_highway_lpkm)+'**')
    
    if st.button("Estimate Price", key='predict'):
        try:            
            if(load_test_data == 'User Entry'):
                user_car = pd.DataFrame({'car_brand':[car_brand], 'body_type':[body_type], 'condition':[condition], 'num_seats':[num_seats], 'mileage':[mileage],'power_kW':[power_kW], 'transmission':[transmission], 'displacement':[displacement], 'fuel':[fuel], 'color':[color], 'seller':[seller],'year_built':[year_built], 'num_owners':[num_owners], 'co2_emission_gpkm':[co2_emission_gpkm], 'fuel_eff_combi_lpkm':[fuel_eff_combi_lpkm],'fuel_eff_city_lpkm':[fuel_eff_city_lpkm], 'fuel_eff_highway_lpkm':[fuel_eff_highway_lpkm]})
                price_actual = 0
            else:
                user_car = X_test.iloc[test_data_idx:test_data_idx+1]
                price_actual = y_test.iloc[test_data_idx].price

            #st.table(user_car[0])
            #col1, col2 = st.columns(2)
            prediction1 = knn_model.predict(user_car)
            output1 = round(prediction1[0])
            #prediction2 = lm_model.predict(user_car)
            #output2 = round(prediction2[0],2)

            #with col2:
            if(load_test_data == 'Test Data'):               
                st.subheader('Price Predicted by KNN_Model \(in Euros\): ' + str(output1))
                st.subheader('Actual Price from DataSet \(in Euros\):' + str(price_actual))
                #st.caption('*LM_Model : '+str(output2))
            else:                
                st.subheader('Price Predicted by KNN_Model \(in Euros\): ' + str(output1))
                #st.subheader('Price Predicted by LR_Model \(in Euros\) : ' + str(output2))
                #st.caption('*LM_Model : '+str(output2))
                
            
            #prediction2 = lm_model.predict(user_car)
            #output2 = round(prediction2[0],2)
            #st.write('LM_Model : ',output2)
            #prediction3 = dtree_model.predict(X_test)
            #output3 = round(prediction3[0],2)
            #st.write('DTREE_Model :',ouptput3)            
            
            #st.write('Actual Price:',y_test.iloc[0].price)
        except:
            st.warning("Opps!! Something went wrong\nTry again")          


if __name__ == "__main__":
    main()

