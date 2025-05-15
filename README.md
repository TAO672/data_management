# data_management

##  Data Cleaning
-- Step 1:1: Make sure the data has been loaded into Hive correct
SELECT * FROM wine LIMIT 10;

-- Step 2: Data Cleaning
-- 2.1 Check if each column has missing values
SELECT COUNT(*) - COUNT(Alcohol) AS missing_Alcohol,
       COUNT(*) - COUNT(Malicacid) AS missing_Malicacid,
       COUNT(*) - COUNT(Ash) AS missing_Ash
FROM wine;

-- 2.2 Delete duplicate data
CREATE TABLE wine_cleaned AS 
SELECT DISTINCT * FROM wine;

## Data Analysis and Modeling
-- Step 1: Target variable analysis
SELECT class, COUNT(*) 
FROM wine
GROUP BY class;

-- Step 2: Calculate statistics
SELECT AVG(Alcohol) AS avg_Alcohol, 
       MAX(Alcohol) AS max_Alcohol,
       MIN(Alcohol) AS min_Alcohol,
       AVG(Malicacid) AS avg_Malicacid,
       MAX(Malicacid) AS max_Malicacid,
       MIN(Malicacid) AS min_Malicacid,
FROM wine;

-- Continue with statistics of other features
## Data Visualization (using Python)
