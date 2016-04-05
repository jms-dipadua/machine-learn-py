# File: outperform.r
# Author: James DiPadua
# Purpose: To Locate Out-Performing Stocks over a Given Time Period
# Date: 03.31.2016

# clear the enviornment when we start
rm(list=ls())

# Needed Packages: 
	# if the following are not already in your system, install (by uncommenting)
# install.packages("RSQLite") # deprecated from this impelmentation
# install.packages("doSNOW")
# install.packages("bigmemory")
# install.packages("foreach")
# install.packages("parallel")
# install.packages('doParallel')  


# load the libraries
#library(RSQLite) # not used in this particular implementation
library(bigmemory) # used for file-backed storage of 
library(foreach)  # used for our loop through each of the securities 
#library(doSNOW)  # not needed for this particular multi-core fork approach # i.e. for socket / non-POSIX systems
library(parallel) # technically packaged w/ snow BUT detectCores() MAY not run WITHOUT loading the library
library(doParallel)  # sets up a forked multi-core process, used with foreach

# general process
# 1. setup some initial variables such as the date range to evaluate for over performance
# 2. create a cluster (and initalize it)  
# 3. BIGMEMORY for file-backed storage
# 4. do a for-loop (using foreach & doParallel) to calculate the returns for each company (in the given date range)
# 5. (a) get the AVG results for the stocks and (b) select the outperformers
# 6. save the results (just company names) in a CSV 
# 7. do a happy dance cuz we're done. :)

# notes: 
# your data file (stocksNumeric.csv) MUST be in this working directory
# you MUST use the stocksNumeric.csv BECAUSE we're going to use a bigmemory matirx which relies on all the same data type
# stocks are organized all together -- so google is 90 rows together, apple's 90 days are all together, etc 

# 1. initial variables: working directory, date range, vector of ticker symbols
# SET YOUR WORKING DIRECTORY ! :)
#setwd()

# basic integer: 90 for full 90 days, 5 for previous 5 days, etc
date_range <- 90
# ticker symbols for our final file
stocks_clean <- c('AAPL', 'GOOG', 'ORCL', 'INTC', 'SYMC', 'FB', 'CSCO', 'XRX', 'IBM', 'MSFT')

# 2. CREATE cluster  
# get number of cores and create a cluster based on that number
registerDoParallel(detectCores())  # POSIX ONLY!! 
print(getDoParWorkers())  # outputs the number of workers
# socket / non-POSIX
#cluster <- makeCluster(detectCores())
# register cluster w/ snow
#registerDoSNOW(cluster) # 

# 3. GET DATA
read.big.matrix("stocksNumeric.csv",
	backingfile="stocksLarge.bin",
	descriptorfile="stocksLarge.desc",
	type="double", sep=",")

big_stock_desc <- dget("stocksLarge.desc")
bigSD <- attach.big.matrix(big_stock_desc)
#print(dim(bigSD))
#print(bigSD[1:10, 1:5])

stocks <- levels(as.factor(bigSD[,2]))

#4 parallelized work
stock_perf <- foreach( i= c(1:length(stocks)) ) %dopar% { 
		stock_data <- bigSD[bigSD[,2]==i,]
		avg_returns <- mean(stock_data[1:date_range,5])
	}

#stock_perf is an ugly, nested object
# SO, we're going to clean it up with a quick sapply
# it will make the mean() function more accessible
indexer <- 0
performance_clean <- sapply(1:length(stock_perf), FUN=function(x) {
	indexer <<- indexer + 1  # indexer is out of scope so we use <<- to access it
	indv_perform <- stock_perf[[indexer]]
})

# 5a. get the mean of all the returns
mean_stock_perf <- mean(performance_clean) # returns the mean

# 5b. set the top performers
# first we'll get the indicies (because those will be our stocks of interest)
positions <- which(stock_perf > mean_stock_perf)
# then we create a vector with the outperforming stocks 
outperformers <- stocks_clean[positions]
# print(outperformers)

# 6. save the file
file_name <- paste("outperformers-dipadua-FINAL-",toString(date_range),"day-range.csv", sep="")
write.table(outperformers, file = file_name, row.names=FALSE, col.names=FALSE, sep=",")

# 7. stop cluster
# GOTCHA! with doParallel's registerDoParallel(), it's automatically closed @ .onUnload()
# HOWEVER, if this was used via a "doSNOW()" route, then you can uncomment this line and it'll stop your cluster (ex: socket clusters)
# stopCluster(cluster)

