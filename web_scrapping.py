#!/usr/bin/env python
# coding: utf-8

# In[70]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By 
import re
import time

driver_path = "C:\\Users\\pulin\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

url = "https://www.magicbricks.com/property-for-sale/residential-commercial-agricultural-real-estate?bedroom=1,2,3,4,5,%3E5&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa,Residential-Plot,Commercial-Office-Space,Office-ITPark-SEZ,Commercial-Shop,Commercial-Showroom,Commercial-Land,Industrial-Land,Warehouse-Godown,Industrial-Building,Industrial-Shed,Agricultural-Land,Farm-House&cityName=Vadodara"
driver.get(url)

time.sleep(5)

last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

button = driver.find_elements(By.CLASS_NAME,"mb-srp__card__summary__action")

for i in range(len(button)):
    driver.execute_script("arguments[0].click();", button[i])

parent_element = []
price = []
carpet_area = []
name = []
Furnishing = []
bhk = []
house_type = []
address = []

parent_element = driver.find_elements(By.CLASS_NAME,"mb-srp__list")

for card in parent_element:
    try:
        price.append(card.find_element(By.CLASS_NAME, "mb-srp__card__price--amount").text)
    except:
        price.append("N/A")  # Handle missing price

    try:
        name.append(card.find_element(By.CLASS_NAME, "mb-srp__card--title").text)
    except:
        name.append("N/A")  # Handle missing name

    try:
        Furnishing.append(card.find_element(By.CSS_SELECTOR, 'div.mb-srp__card__summary__list--item[data-summary="furnishing"] .mb-srp__card__summary--value').text)
    except:
        Furnishing.append("N/A")  # Handle missing furnishing info
    
    try:
        try:
            carpet_area.append(card.find_element(By.CSS_SELECTOR, 'div.mb-srp__card__summary__list--item[data-summary="carpet-area"] .mb-srp__card__summary--value').text)
        except:
            carpet_area.append(card.find_element(By.CSS_SELECTOR, 'div.mb-srp__card__summary__list--item[data-summary="super-area"] .mb-srp__card__summary--value').text)
    except:
        carpet_area.append("N/A")  # Handle missing furnishing info

bhk_pattern = r'(\d+|\>\s*\d+)\s*BHK'
house_type_pattern = r'BHK\s*(Villa|House|Farm House|Apartment|Studio)'
address_pattern = r'for Sale in (.*)'

for item in name:

    # Extract BHK
    bhk_match = re.search(bhk_pattern, item)
    if bhk_match:
        bhk_value = bhk_match.group(0).replace('BHK', '').replace('>', '').strip()
        bhk.append(bhk_value)
    else:
        bhk.append("N/A")

    # Extract house type
    house_type_match = re.search(house_type_pattern, item)
    if house_type_match:
        house_type.append(house_type_match.group(1))
    else:
        house_type.append("N/A")

    # Extract address
    address_match = re.search(address_pattern, item)
    if address_match:
        address.append(address_match.group(1))
    else:
        address.append("N/A")

# Print the extracted values
print(f"Price: {price}")
print(f"Name: {name}")
print(f"Furnishing: {Furnishing}")
print(f"Carpet Area: {carpet_area}")
print(f"BHK: {bhk}")
print(f"House Type: {house_type}")
print(f"Address: {address}")

# driver.quit()


# In[76]:


print(f"Number of Cards: {len(parent_element)}")
print(f"Number of button: {len(button)}")
print(f"Number of Price: {len(price)}")
print(f"Number of Name: {len(name)}")
print(f"Number of Furnishing: {len(Furnishing)}")
print(f"Number of Carpet Area: {len(carpet_area)}")
print(f"Number of BHK: {len(bhk)}")
print(f"Number of House Type: {len(house_type)}")
print(f"Number of Address: {len(address)}")


# In[72]:


import pandas as pd

data = {
    "Carpet Area":carpet_area,
    "BHK":bhk,
    "House Type":house_type,
    "Furnishing":Furnishing,
    "Address":address,
    "Price":price
}

df = pd.DataFrame(data)
pd


# In[73]:


df.head()


# In[74]:


df.to_csv("Vadodara_housing.csv",index=False)


# In[75]:


urls=["https://www.magicbricks.com/property-for-sale/residential-commercial-agricultural-real-estate?bedroom=1,2,3,4,5,%3E5&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa,Residential-Plot,Commercial-Office-Space,Office-ITPark-SEZ,Commercial-Shop,Commercial-Showroom,Commercial-Land,Industrial-Land,Warehouse-Godown,Industrial-Building,Industrial-Shed,Agricultural-Land,Farm-House&cityName=Vadodara"
      ,"https://www.magicbricks.com/property-for-sale/residential-commercial-agricultural-real-estate?bedroom=1,2,3,4,5,%3E5&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa,Residential-Plot,Commercial-Office-Space,Office-ITPark-SEZ,Commercial-Shop,Commercial-Showroom,Commercial-Land,Industrial-Land,Warehouse-Godown,Industrial-Building,Industrial-Shed,Agricultural-Land,Farm-House&cityName=Anand"
      ,"https://www.magicbricks.com/property-for-sale/residential-commercial-agricultural-real-estate?bedroom=1,2,3,4,5,%3E5&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa,Residential-Plot,Commercial-Office-Space,Office-ITPark-SEZ,Commercial-Shop,Commercial-Showroom,Commercial-Land,Industrial-Land,Warehouse-Godown,Industrial-Building,Industrial-Shed,Agricultural-Land,Farm-House&cityName=Nadiad"
      ,"https://www.magicbricks.com/property-for-sale/residential-commercial-agricultural-real-estate?bedroom=1,2,3,4,5,%3E5&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa,Residential-Plot,Commercial-Office-Space,Office-ITPark-SEZ,Commercial-Shop,Commercial-Showroom,Commercial-Land,Industrial-Land,Warehouse-Godown,Industrial-Building,Industrial-Shed,Agricultural-Land,Farm-House&cityName=Surat"
      ]

