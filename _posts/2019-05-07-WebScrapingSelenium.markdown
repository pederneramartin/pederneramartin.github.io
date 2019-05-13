---
title: "Web Scraping with Python"
layout: post
date: 2019-05-07 20:00
tag: 
- webscraping
- python
- selenium
image:
headerImage: False
projects: true
hidden: false # don't count this post in blog pagination
description: "Using selenium library to scrape data from ecommerce sites."
category: project
author: pederneramartin
externalLink: false
---

Web Scraping is a technique employed to extract large amounts of data from websites whereby the data is extracted and saved to a local file in your computer or to a database.

Data displayed by most websites can only be viewed using a web browser. They do not offer the functionality to save a copy of this data for personal use. The only option then is to manually copy and paste the data - a very tedious job which can take many hours or sometimes days to complete. Web Scraping is the technique of automating this process, so that instead of manually copying the data from websites, the Web Scraping algorithm will perform the same task within a fraction of the time.

In this project I've used selenium to scrape data from ecommerce sites. Selenium is a Web Browser Automation Tool monstly for testing purpose, but is certainly not limited to just that. It allows you to open a browser of your choice & perform tasks as a human being would, such as:
* Clicking buttons
* Entering information in forms
* Searching for specific information on the web pages

### What did I build?

I built a web scraping program that scraped differents ecommerce sites and get data such as:
* Title
* Sales Price
* Original Price
* Availability (if it is in stock)
* Qty Reviews
* Stars
* Ranking

### What languages/packages did I use?

For this project I used Python3.x, and the following packages/libraries and drivers:

* selenium
* chromedriver
* json
* datetime

---
```python
from selenium import webdriver
import json
import datetime

#CHROME
chrome_path = r"path of webdriver"
driver = webdriver.Chrome(executable_path=chrome_path)

#EXTRACTION
def parse(url,i):
    driver.get(url)
    try:
        name_raw = driver.find_element_by_xpath('//h1[@id="title"]')
        NAME = name_raw.text
    except Exception as e:
            NAME = None
            pass  
    try:
        saleprice_raw = driver.find_element_by_xpath('//span[contains(@id,"ourprice") or contains(@id,"saleprice") or contains(@id,"priceblock_dealprice")]')
        SALE_PRICE = saleprice_raw.text
    except Exception as e:
            SALE_PRICE = None
            pass
    try:
        originalprice_raw = driver.find_element_by_xpath('//span[contains(@class,"a-text-strike") or contains(@id,"a-text-strike")]')
        ORIGINAL_PRICE = originalprice_raw.text
    except Exception as e:
            ORIGINAL_PRICE = None
            pass      
    try:
        availability_raw = driver.find_element_by_xpath('//div[@id="availability"]')
        AVAILABILITY = availability_raw.text
    except Exception as e:
            AVAILABILITY = None
            pass
    try:
        reviews_raw = driver.find_element_by_xpath('//span[@id="acrCustomerReviewText"]')
        REVIEWS = reviews_raw.text
    except Exception as e:
            REVIEWS = None
            pass
    try:
        stars_raw = driver.find_element_by_xpath('//span[@id="acrPopover"]')
        STARS = stars_raw.get_attribute('title')
    except Exception as e:
            STARS = None
            pass
    try:
        ranking_raw = driver.find_element_by_xpath('//*[@id="productDetails_detailBullets_sections1"]/tbody/tr[8]/td/span/span[2]')
        RANKING = ranking_raw.text
    except Exception as e:
        RANKING = None
        pass

    if SALE_PRICE is None:
        SALE_PRICE = ORIGINAL_PRICE
        
    data = {
            'DATE':datetime.date,
            'NAME':NAME,
            'SALE_PRICE':SALE_PRICE,
            'ORIGINAL_PRICE':ORIGINAL_PRICE,
            'AVAILABILITY':AVAILABILITY,
            'REVIEWS':REVIEWS,
            'STARS':STARS,
            'RANKING':RANKING,
            'URL':url,
            }
    return data
    
    

def ReadID():
    #URLs or IDs
    IDS = ['ID/URL',
    'ID/URL',
            ]
    
    #ITERATION
    extracted_data = []
    for i in IDS:
        url = "web page of the ecommerce site"+i
        extracted_data.append(parse(url,i))
    f=open('data.json','w')
    json.dump(extracted_data,f,indent=4)
    
if __name__ == "__main__":
    ReadID()
```
---
