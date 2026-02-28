Feature 0 Prompt:
I created the feature0.md basically to come up with the right infrastructure for supply chain demand forecasting. to do traditional, and ML based forecasting and lightning fast analysis in the UI and aggregations with item, location and customer attributes and should be able to handle 500 millions and more records. Please refine and make it simple. Iceberg, spark,trino that you suggested earlier are missing. refine and make a simple .md file with the right recommendation. 
Addiitonal prompts: MLFlow etc


Write feature1.md: Build an internal data structure for the supply chain demand forecasting that seemlessly goes well with the Iceberg, MlFLOW, postgres, and other technologies. The purpose is to store the data on item( multiple hierarchy/attributes) , storage locations(  multiple hierarchy/attirbutes), customer( multiple hierarchies/attributes). The sales history including qty sold, qty demanded, qty delievered etc. Pricing info promo information, external weather/events. forecasts generated, forecast archival at multiple lags, maintaing different forecasts from different algorithms, measuring KPIs, dashboarding etc. Also the best of the best forecasting internal data architecture that is robust and can be easily mapped to any customer data via mapping tables. Metadata layer and storage dat layer for easy retrievals. Can you refine, we need to add item,storage location level supersession information since this is critical. once superceded the new item should replace the archives for the old location. Secondly, the data structure should be able to do both monthly and weekly forecasting.

Now I have the fact table for the sales but this is a the same ALL customer lcoation and item grain dfu_lvl2_hist.txt is the file DMDUNIT is item, dmdgroup is customer at ALL level and LOC is location. startdate is date dimension but the grain is at the monthly level and the month is represented as the first of the month denoting the entire month and in the format YYYYMMDD and DD is always 01 for this monthly grain. You can ignore U_LVL. U_qty_shipped is the shipped quantity in cases, U_qty_ordered is quantiy demanded by the customer. use just the type 1 for now. 

forecast archive:
now do the same for the forecast fact that is coming from a different system as external forecasts. fcstdate is the date when the forecast was generated, the forecast is generated monthly which is denoted by the start of the month. startdate is the month for which the forecast was generated which is denoted by the start of the month. YYYY-MM-DD. basefcst_pref is the base stat forecast and tothist_dmd is the sales for that month. This file is an archived forecasts file used for measuring the accuracy. execution_lag refers to the lag at which the planners are interested on the accuracy of the forecast. the month difference between the startdate and fcstdate gives the lag of the forecast. we normally store up to 0,1,2,3,4 lags. 0 lag means startdate and fcstdate are same. if fcstdate is 2023-07-01 and the startdate is 2023-08-01 the lag is 1 and so on. Please pull this from ingestion to sink to UI.

I want to implement a clustering framework that can seperate the dfu based on the historical demand patterns. Generate the time series features, use item, and dfu features and come up with the optimal number of clusters. I need a valid definition for each cluster. like high volume, steady demand, seasonal cluster etc. add clustering as feature13.md. Need a strong clustering model so that LGBM global models can perform well.

Create timeframe 10 timeframes, A - J

For example the sales history has until Jan 2026.
The sales history starts with Feb 2023

We have 36 months

Go back 12 months.

First time A : Jan 2023 - Mar 2025 - training
April 2025 - Jan 2026 - Predictions

First time B : Jan 2023 - Apr 2025 - training
May 2025 - Jan 2026 - Predictions

First time C : Jan  2023 - May2025 - training
June 2025 - Jan 2026 - Predictions

First time D : Jan 2023 - June2025 - training
July 2025 - Jan 2026 - Predictions

First time E : Jan 2023 - July 2025 - training
Aug 2025 - Jan 2026 - Predictions

First time F : Jan 2023 - Aug 2025 - training
Sep 2025 - Jan 2026 - Predictions

First time G : Jan 2023 - Sep 2025 - training
Oct 2025 - Jan 2026 - Predictions

First time H: Jan 2023 - Oct 2025 - training
Nov 2025 - Jan 2026 - Predictions

First time I: Jan 2023 - Nov 2025 - training
Dec 2025 - Jan 2026 - Predictions

First time J: Jan 2023 - Dec 2025 - training
Jan 2026 - Predictions

With these Jan 2026 will have lag 0( Time Frame J)  to lag 9, but we care about only 
Lag 0 to lag 4 for all the months.

April 2025 will have only lag 0,
May 2025 will have lag 0 and lag 1
..
Aug 2025 to Jan 2025 will have all the 5 lags 0 to 4.

Measure the forecast at those lags, and store it. But for get the accuracy we need it only at execution lag the execution lag number is in the dfu.txt that is already loaded. Use that and load the data for each dfu for every month from Aug 2025 to Jan 2026 in to the forecast table with the right measure id. For example LGBM or different LGBM model or catboost

I need to implement the feature34.md. The inventory snapshots are in the dataafiles. therre are 14 files one for each month named that is Inventory_Snapshot_YYYY_MM.csv eg nventory_Snapshot_2025_01.csv. all needs to be loaded from datafiles allthe way similar to sales and external forecasts. Spin up 10 agents and manage this massive development.

The champion was supposed to be yet another forecastign algorithm that is nothing but assigning the best of the available algorithm to the DFU for forecasting the next n months. so in the case of backtesting it is more like chosing the best algorithm everymonth based on the performance of the  AVAILABLE algorithms in the previous months similar to ceiling but in Ceiling we do it after the fact best, here we choose before the fact. In simple words, ceiling the best case scenario where your pickbest for champion picked the best all the time with 100% accuracy but we all know that is not possible. The idea is to improve the champion selection to narrow the gap between champion and ceiling. Can you chekc the code to see how far away are from this logic and correct it.

check the code base with 5 judges 1. critic 2. technologist 3. UI/UX expert 4. Steve jobs 5. supply chain expert and provide me a very honest abrupt feedback so i can improve

There is problem with the champion select algorithm. selection of April 2025 based on the Jan - Mar 2025 execution lag is good. But selecting it for the execution lag for April 2025 will only be true if the execution lag is 0 is the execution lag is 1 then the selection of the champion should have been done in when we forecasted for APril 2025 in March 2025 when the sales for MArch wasn't available. So the selction of champion in this case will be for April lag 0, May lag 1 June lag 2. In short can you provide me an updated startegy that will mimic the real life. That is if I am forecasting in Feb 2026 then I choose the champion based on the data unitl Jan 2026 ( exec lags) but will assign it for Feb 2026 and above. 

run 5 agents and go through each spec and add an example for each of the feature and functionality. also add another 5 qa agents to check if all the examples are added if not send it back to the agent who created it to add the missing one. 

This is more than demand Studio it will emerge in to inventory optimization network optimization and replenishment. so name it appropriately using 10 creative agents suggest few names and then start 5 review agents one on each perception of brand, ease etc and suggest the final 5 .