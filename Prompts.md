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