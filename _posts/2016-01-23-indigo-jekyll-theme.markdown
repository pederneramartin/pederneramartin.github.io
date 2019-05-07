---
title: "Web Scraping"
layout: post
date: 2019-05-07 20:00
tag: webscraping
image:
headerImage: False
projects: true
hidden: false # don't count this post in blog pagination
description: "Web Scraping using Selenium."
category: project
author: johndoe
externalLink: false
---

Web Scraping Project for E-commerce Site - Selenium for browser automatization -

---
```python
	from selenium import webdriver
	import csv,os,json
	import time

	#CHROME
	chrome_path = r"PUT PATH OF WEBDRIVER"
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
	        seller_raw = driver.find_element_by_xpath('//div[@id="merchant-info"]')
	        SELLER = seller_raw.text
	    except Exception as e:
	            SELLER = None
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
	        ranking_raw = driver.find_element_by_xpath('//*[@id="productDetails_detailBullets_sections1"]')
	        RANKING = ranking_raw.text
	    except Exception as e:
	        RANKING = None
	        pass

	    pos = RANKING.find("Best Sellers Rank")
	    if pos is -1:
	        RANKING = None
	    if RANKING is not None:
	        RANKING = RANKING.replace("\n","")
	        pos = RANKING.find("Best Sellers Rank")
	        RANKING = RANKING[pos+17:]
	        RANKING = RANKING.strip(" ")
	    if ORIGINAL_PRICE is None:
	        ORIGINAL_PRICE = SALE_PRICE
	    if SALE_PRICE is None:
	        SALE_PRICE = ORIGINAL_PRICE
	        
	    data = {
	            'NAME':NAME,
	            'SALE_PRICE':SALE_PRICE,
	            'ORIGINAL_PRICE':ORIGINAL_PRICE,
	            'AVAILABILITY':AVAILABILITY,
	            'SELLER': SELLER,
	            'REVIEWS':REVIEWS,
	            'STARS':STARS,
	            'RANKING':RANKING,
	            'URL':url,
	            }
	    return data
	    
	    

	def ReadID():
	    #Put here your URLs or IDs
	    IDS = ['ID/URL',
	    'ID/URL',
	            ]
	    
	    #ITERATION
	    extracted_data = []
	    for i in IDS:
	        url = "PUT HERE YOUR WEBPAGE"+i
	        extracted_data.append(parse(url,i))
	    f=open('data.json','w')
	    json.dump(extracted_data,f,indent=4)
	    
	if __name__ == "__main__":
	    ReadID()
```
---
