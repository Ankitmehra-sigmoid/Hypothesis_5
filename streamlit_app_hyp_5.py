#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px


# In[ ]:


# File: streamlit_app.py

data=pd.read_csv('/Users/ankit/Downloads/EDA_IB/New_Client_Data/streamlit/df_original_scenario.csv')
data2=pd.read_csv('/Users/ankit/Downloads/EDA_IB/New_Client_Data/streamlit/df_updated_scenario.csv')

data_orig_nitish=pd.read_csv('/Users/ankit/Downloads/EDA_IB/New_Client_Data/streamlit/df_original_scenario_nitish.csv')
data_up_nitish=pd.read_csv('/Users/ankit/Downloads/EDA_IB/New_Client_Data/streamlit/df_updated_scenario_nitish.csv')


# Filters
st.sidebar.header("Filters")
customers = st.sidebar.selectbox("Customer Clients data", options=data['Customer Clients data'].unique(), index=0)
filtered_data = data[data['Customer Clients data'] == customers]
filtered_data2=data2[data2['Customer Clients data']==customers]

filtered_data_n = data_orig_nitish[data_orig_nitish['Customer Clients data'] == customers]
filtered_data_n2=data_up_nitish[data_up_nitish['Customer Clients data']==customers]


postal_code = st.sidebar.selectbox("Postal Code clients data", options=filtered_data['Postal Code clients data'].unique())
filtered_data = filtered_data[filtered_data['Postal Code clients data'] == postal_code]
filtered_data2= filtered_data2[filtered_data2['Postal Code clients data'] == postal_code]

filtered_data_n = filtered_data_n[filtered_data_n['Postal Code clients data'] == postal_code]
filtered_data_n2=filtered_data_n2[filtered_data_n2['Postal Code clients data']==postal_code]

                    
street = st.sidebar.selectbox("Street", options=filtered_data['Street'].unique())
filtered_data = filtered_data[filtered_data['Street'] == street]
filtered_data2= filtered_data2[filtered_data2['Street'] == street]

filtered_data_n = filtered_data_n[filtered_data_n['Street'] == street]
filtered_data_n2=filtered_data_n2[filtered_data_n2['Street']==street]


                     
dc = st.sidebar.selectbox("DC", options=filtered_data['DC'].unique())
filtered_data = filtered_data[filtered_data['DC'] == dc]
filtered_data2= filtered_data2[filtered_data2['DC'] == dc]

filtered_data_n = filtered_data_n[filtered_data_n['DC'] == dc]
filtered_data_n2=filtered_data_n2[filtered_data_n2['DC']==dc]


savings=filtered_data['Total cost orig'].sum()-filtered_data2['Total cost updated'].sum()
savings_nitish=filtered_data_n['total_cost_cost_sheet_as_is_inflated'].sum()-filtered_data_n2['total_cost_cost_sheet_after_inflated'].sum()
                     
months = st.sidebar.multiselect(
    "Month_orig",
    options=filtered_data['Month_orig'].unique(),
    default=filtered_data['Month_orig'].unique()
)
# filtered_data = filtered_data[filtered_data['Month_orig'].isin(months)]

# filtered_data = filtered_data[filtered_data['Month_orig'] == month]

weeks = filtered_data[filtered_data['Month_orig'].isin(months)]['week_of_year'].unique().tolist() 

# weeks=filtered_data['week_of_year'].unique().tolist()
filtered_data = filtered_data[filtered_data['week_of_year'].isin(weeks)]

start_date=filtered_data['Lst.datum'].min()
last_date=filtered_data['Lst.datum'].max()

filtered_data2= filtered_data2[filtered_data2['updated_delivery_date']>=start_date]
filtered_data2= filtered_data2[filtered_data2['updated_delivery_date']<=last_date]

filtered_data_n2=filtered_data_n2[filtered_data_n2['updated_delivery_date']>=start_date]
filtered_data_n2= filtered_data_n2[filtered_data_n2['updated_delivery_date']<=last_date]

# df_savings_f=df_savings_f[df_savings_f['week_of_year'].isin(weeks)]


# Calculations for tiles
# total_cost_orig = filtered_data['Total cost orig'].sum()
# updated_cost = filtered_data2['Total cost updated'].sum() # Example: applying a 10% discount
# savings = total_cost_orig - updated_cost

# total_cost_orig = df_savings_f['Total cost orig'].sum()
# savings = df_savings_f['savings final'].sum()
# if savings==0:
#     updated_cost=total_cost_orig
# else:
#     updated_cost = df_savings_f['Total cost updated'].sum() # Example: applying a 10% discount



# total_cost_orig = filtered_data['total_cost_cost_sheet_as_is'].sum()
# updated_cost = filtered_data2['total_cost_cost_sheet_after'].sum() # Example: applying a 10% discount
# savings = total_cost_orig - updated_cost


# Display tiles
st.title("Cost Dashboard")


# # Bar plot function
# def create_bar_plot(df, date_col, qty_col):
#     df[date_col] = df[date_col].astype(str)
#     fig = px.bar(df, x=date_col, y=qty_col, 
#                  labels={date_col: date_col, qty_col: qty_col}, 
#                  title=f"{qty_col} by {date_col}") 
#     return fig

# # Graphs
# st.header("Graphs")
# graph1 = create_bar_plot(filtered_data, 'Lst.datum', 'TOTPAL')
# graph2 = create_bar_plot(filtered_data2, 'updated_delivery_date', 'TOTPAL')
# graph3 = create_bar_plot(filtered_data_n2, 'updated_delivery_date', 'TOTPAL')

# st.plotly_chart(graph1)

# col1, col2 = st.columns(2)
# col1.plotly_chart(graph2) 
# col2.metric("Total Savings in 2023", f"${savings:,.2f}")

# Bar plot function
def create_bar_plot(df, date_col, qty_col, title, width=900, height=350):
    df[date_col] = df[date_col].astype(str)
    fig = px.bar(
        df, 
        x=date_col, 
        y=qty_col, 
        labels={date_col: 'Delivery Date', qty_col: 'Total Pallets'}, 
        title=title,
        text=qty_col,  # Add text labels on the bars
        width=width,    # Set the width of the graph
        height=height   # Set the height of the graph
    )
    
    # Customize text position for better visibility
    fig.update_traces(textposition='outside')  # Places text above the bars
    
    return fig


# Graphs
st.header("Graphs")
graph1 = create_bar_plot(filtered_data, 'Lst.datum', 'TOTPAL',title='Shipment Profile Without Consolidation')
graph2 = create_bar_plot(filtered_data2, 'updated_delivery_date', 'TOTPAL',title='Shipment Profile After Consolidation', width=900, height=350)
# graph3 = create_bar_plot(filtered_data_3, 'updated_delivery_date', 'TOTPAL', width=800, height=350)

# Display the graphs
st.metric("Total Savings in 2023", f"${savings:,.2f}")
st.plotly_chart(graph1)
# col1, col2 = st.columns(2)
# col1.plotly_chart(graph2)
# col2.metric("Total Savings in 2023", f"${savings:,.2f}")

st.plotly_chart(graph2)

# col1, col2 = st.columns([7, 1])  # Set column proportions (3:1 for larger graph and metric alignment)

# # Graph in col1
# with col1:
#     st.plotly_chart(graph2, use_container_width=True)  # Make the graph responsive to column width

# Metric in col2
# with col2:
#     st.metric("Total Savings in 2023", f"${savings:,.2f}")

# Here is the updated code:
# Generate Graphs
# graph1 = create_bar_plot(filtered_data, 'Lst.datum', 'TOTPAL')
# graph2 = create_bar_plot(filtered_data2, 'updated_delivery_date', 'TOTPAL')
# graph3 = create_bar_plot(filtered_data_n2, 'updated_delivery_date', 'TOTPAL')

# graph_container = st.container()

# graph_container.header("Graphs")

# # Adjust height of graph
# graph1.update_layout(height=400)
# # Show graph
# graph_container.plotly_chart(graph1, use_container_width=True)

# c1, c2 = graph_container.columns([1, 5], vertical_alignment="center")
# c1.metric("Total Savings in 2023, Approach 2", f"${savings_nitish:,.2f}")

# # Adjust height of graph
# graph2.update_layout(height=400)
# # Show Graph
# c2.plotly_chart(graph2, use_container_width=True)



# col3, col4 = st.columns(2)
# col3.plotly_chart(graph2) 
# col4.metric("Total Savings in 2023 Approach 2", f"${savings_nitish:,.2f}")

