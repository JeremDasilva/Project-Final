#Libraries
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd
import numpy as np

#import psycopg
#from psycopg import sql

#import os
#from dotenv import load_dotenv, find_dotenv

import seaborn as sns
import matplotlib.pyplot as plt

import pickle

import openai

#Loading the dataset from the database or csv
'''try:
    load_dotenv()
    password = os.getenv('db_password')

    with psycopg.connect(f"dbname=Final-project user=postgres password={password}") as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM car_sales_cleaned;"
        cursor.execute(query)

        rows = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]
    sales_df = pd.DataFrame(rows, columns=columns)
except :''' 
sales_df = pd.read_csv('../Datasets/car_price_prediction_preprocess.csv')

#Formating the dataset
def model_cleaner(model):
    try:
        model = model.split(' ')
        return model[0]
    except :
        return model
sales_df['model'] = sales_df['model'].apply(lambda x : model_cleaner(x) )
sales_df['turbo'] = sales_df['turbo'].replace({0: 'No', 1: 'Yes'})

#Landing page and sidebar
st.title('Used cars market : analysis, prediction and search engine')

st.sidebar.title('Menu')

#Tool 1 : Prediction model
if st.sidebar.checkbox('Prediction Model'):
    
    st.markdown('## Prediction Model')
    
    #Preparing the dataset the same way as the prediction model
    sales_df_prediction = sales_df
    sales_df_prediction.drop(['turbo', 'wheel', 'cylinders', 'drive_wheels'], axis=1, inplace = True)
    sales_df_prediction = sales_df_prediction[sales_df_prediction['price'] < 50000]
    sales_df_prediction = sales_df_prediction[sales_df_prediction['price'] > 1000]
    sales_df_prediction = sales_df_prediction[sales_df_prediction['levy'] < 2000]
    sales_df_prediction = sales_df_prediction[sales_df_prediction['mileage'] < 200000]
    sales_df_prediction = sales_df_prediction[sales_df_prediction['production_year'] > 2000]
    sales_df_prediction = sales_df_prediction[sales_df_prediction['engine_size'] >= 1]
    
    column_order = list(sales_df_prediction.drop(['price'], axis = 1).columns)
    
    #User interface for input parameters
    X_pred = {}
    
    options_list_manufacturer = list(sorted(sales_df['manufacturer'].unique()))
    selected_manufacturer = st.sidebar.selectbox('manufacturer', options_list_manufacturer)
    X_pred['manufacturer'] = selected_manufacturer    
    filtered_df_manufacturer = sales_df[sales_df['manufacturer'] == selected_manufacturer]

    options_list_model = list(sorted(filtered_df_manufacturer['model'].unique()))
    X_pred['model'] = st.sidebar.selectbox('model', options_list_model)
    
    for col in sales_df_prediction.drop(['price', 'manufacturer', 'model'], axis=1).columns:
        if sales_df_prediction[col].dtype in [np.float64, np.int64]:
            col_value = st.sidebar.slider(col, int(sales_df_prediction[col].min()), int(sales_df_prediction[col].max()), int(sales_df_prediction[col].mean()))
            X_pred[col] = col_value
        else:
            col_value = st.sidebar.selectbox(col, sales_df_prediction[col].unique())
            X_pred[col] = col_value
            
    st.write('I want to estimate the following car :')
    X_pred_df = pd.DataFrame(X_pred, index = [0])
    st.write(X_pred_df)
            
    #Encoding the user input to feed the model
    manufacturer_dict = {manufacturer: i for i, manufacturer in enumerate(sales_df_prediction['manufacturer'].unique())}
    X_pred_df['manufacturer'] = X_pred_df['manufacturer'].map(manufacturer_dict)
    
    model_dict = {model: i for i, model in enumerate(sales_df_prediction['model'].unique())}
    X_pred_df['model'] = X_pred_df['model'].map(model_dict)

    X_pred_df['lether_interior'] = X_pred_df['lether_interior'].replace({'Yes': 1, 'No': 0})

    fuel_type_dict = {fuel_type: i for i, fuel_type in enumerate(sales_df_prediction['fuel_type'].unique())}
    X_pred_df['fuel_type'] = X_pred_df['fuel_type'].map(fuel_type_dict)

    gearbox_type_dict = {gearbox_type: i for i, gearbox_type in enumerate(sales_df_prediction['gearbox_type'].unique())}
    X_pred_df['gearbox_type'] = X_pred_df['gearbox_type'].map(gearbox_type_dict)

    color_dict = {color: i for i, color in enumerate(sales_df_prediction['color'].unique())}
    X_pred_df['color'] = X_pred_df['color'].map(color_dict)

    category_dict = {category: i for i, category in enumerate(sales_df_prediction['category'].unique())}
    X_pred_df['category'] = X_pred_df['category'].map(category_dict)
    
    #Reordering the columns to have them in the same order as the trained model
    X_pred_df = X_pred_df[column_order]

    #Running the model
    with open('../Model/ETR_car_sales.pkl', 'rb') as file:
        model = pickle.load(file)
    if st.button('Run'):
        pred = model.predict(X_pred_df)
        price_estimated = str(pred[0])
        st.write('This car is estimated $', price_estimated)

#Tool 2 : Data visualization 
elif st.sidebar.checkbox('Data visualization') :
    sales_df['turbo'] = sales_df['turbo'].replace({0: 'No', 1: 'Yes'})
    
    st.markdown('## Data visualization')
    
    options_list = ['Mean price by production year',
                    'Top manufacturer',
                    'Distribution of car categories',
                    'Distribution of gearbox types',
                    'Distribution of colors',
                    'Distribution of fuel type',
                    'Behavior of the model']

    selected_option = st.sidebar.selectbox('What do you want to visualize?', options_list)

    if selected_option == options_list[0]:
        plt.figure(figsize=(10, 6))
        sales_df_sorted = sales_df.groupby('production_year')['price'].mean().reset_index().sort_values(by='production_year')
        plt.barh(y=sales_df_sorted['production_year'], width=sales_df_sorted['price'])
        plt.xlabel('Mean Price')
        plt.ylabel('Production Year')
        plt.title(options_list[0])
        plt.tight_layout()
        st.pyplot()
        
    elif selected_option == options_list[1]:
        plt.figure(figsize=(10, 6))
        manufacturer_counts = sales_df['manufacturer'].value_counts().reset_index()
        ax = sns.barplot(x='index', y='manufacturer', data=manufacturer_counts[0:11])
        for bar in ax.patches:
            ax.annotate(str(int(bar.get_height())), 
                (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')
        plt.xlabel('Manufacturer')
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Count')
        plt.title(options_list[1])
        plt.tight_layout()
        st.pyplot()

    elif selected_option == options_list[2]:
        category_counts = sales_df['category'].value_counts().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='index', y='category', data=category_counts[0:6])
        plt.xlabel('Category')
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Count')
        plt.title(options_list[2])
        plt.tight_layout()
        st.pyplot()
        
    elif selected_option == options_list[3] :
        gearbox_count = sales_df['gearbox_type'].value_counts().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='index', y='gearbox_type', data=gearbox_count)
        plt.xlabel('Gear box')
        plt.ylabel('Count')
        plt.title(options_list[3])
        plt.tight_layout()
        st.pyplot() 
        
    elif selected_option == options_list[4] :
        color_count = sales_df['color'].value_counts().reset_index()
        plt.figure(figsize=(10, 6))
        unique_colors = color_count['index'].tolist()
        palette = sns.color_palette("husl", len(unique_colors))
        sns.barplot(x='index', y='color', data=color_count[0:11], palette=palette)
        plt.xlabel('Color')
        plt.ylabel('Count')
        plt.title(options_list[4])
        plt.tight_layout()
        st.pyplot() 
        
    elif selected_option == options_list[5] :
        fuel_count = sales_df['fuel_type'].value_counts().reset_index()
        plt.figure(figsize=(8, 8))
        sns.barplot(x='index', y='fuel_type', data=fuel_count)
        plt.xlabel('Count')
        plt.ylabel('Fuel')
        plt.title(options_list[5])
        st.pyplot()
        
    elif selected_option == options_list[6] :
        
        with open('../Model/ETR_car_sales.pkl', 'rb') as file:
            model = pickle.load(file)
        
        X_test = pd.read_pickle('../Model/test_data.pkl')
        predictions = model.predict(X_test)
        predictions = pd.DataFrame(predictions, columns=['predictions'])
        predictions = predictions.sort_values(by = 'predictions').reset_index()
        
        y_test = pd.read_pickle('../Model/test_label.pkl')
        y_test_values = y_test.values
        y_test_values = pd.DataFrame(y_test_values, columns=['y_test_values'])
        y_test_values = y_test_values.sort_values(by = 'y_test_values').reset_index()

        plt.figure(figsize=(10, 6))
        plt.scatter(predictions.index, predictions['predictions'], color='red', label='Predicted Values')
        plt.scatter(y_test_values.index, y_test_values['y_test_values'], color='blue', label='Test Values')
        plt.ylabel('Price')
        plt.title(options_list[6])
        plt.legend()
        st.pyplot()
        
#Tool 3 : Car finder 
elif st.sidebar.checkbox('Car Finder') :
    
    st.markdown('## Car Finder')
    
    options_list_manufacturer = list(sorted(sales_df['manufacturer'].unique()))
    selected_manufacturer = st.sidebar.selectbox('Which Manufacturer are you looking for?', options_list_manufacturer)
    filtered_df_manufacturer = sales_df[sales_df['manufacturer'] == selected_manufacturer]

    options_list_model = list(sorted(filtered_df_manufacturer['model'].unique()))
    selected_model = st.sidebar.selectbox('Which model are you looking for?', options_list_model)
    filtered_df_model = filtered_df_manufacturer[filtered_df_manufacturer['model'] == selected_model]

    st.write(filtered_df_model)
    
#Car insights with OpenAI
elif st.sidebar.checkbox('Car insights') :
    
    #Manufacturer
    options_list_manufacturer = list(sorted(sales_df['manufacturer'].unique()))
    selected_manufacturer = st.sidebar.selectbox('manufacturer', options_list_manufacturer)   
    filtered_df_manufacturer = sales_df[sales_df['manufacturer'] == selected_manufacturer]

    #Model
    options_list_model = list(sorted(filtered_df_manufacturer['model'].unique()))
    selected_model = st.sidebar.selectbox('model', options_list_model) 
    
    #Year
    year_selected = st.sidebar.slider('year manufactured', 1950, 2024)
    
    #Fuel type
    options_list_fuel = list(sorted(sales_df['fuel_type'].unique()))
    selected_fuel = st.sidebar.selectbox('fuel type', options_list_fuel) 
    
    #Creating the query
    load_dotenv()
    API_KEY = os.getenv('open_ai_api')
    client = openai
    client.api_key = API_KEY
    
    content = f"Can you tell me more about {selected_manufacturer} {selected_model} {year_selected} {selected_fuel} ?"
    
    st.write(content)
    if st.button("Yes!", key = 'insigths'):
        st.write("It may takes some time to load ...")        
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": content}
            ]
        )
        st.write(completion.choices[0].message.content)
        
    st.write('Click on the button below if you want to know more about this car')
    content_issues = content + ' And the most common issue this car has'
    if st.button("Let's find out!", key = 'issues'):
        st.write("It may takes some time to load ...")
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": content_issues}
            ]
        )
        st.write(completion.choices[0].message.content)