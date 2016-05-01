# make sure basic libraries are installed
# ggplot2, sqldf, plyr, gcookbook
# and loaded
rm(list=ls()) # restart workstation
# set working directory
library(sqldf)
library(plyr)
library(ggplot2)
# install.packages("corrplot")
library(corrplot)
library(maps)
library(Hmisc)

states_map <- map_data("state")
#states_map$region <- capitalize(states_map$region)

# library(gcookbook)

basic_d <- read.csv("data/regional-dems.csv", sep=",", header=TRUE)

#basic_d_1 <- basic_d[-1,]
#View(basic_d_1)

d_samp1 <- basic_d[, c('HC03_VC05','HC03_VC06','HC03_VC07','HC03_VC08','HC03_VC09','HC03_VC10','HC03_VC11','HC03_VC12','HC03_VC13','HC03_VC14','HC03_VC17','HC01_VC21','HC03_VC31','HC03_VC32','HC03_VC36','HC03_VC39','HC03_VC40','HC03_VC41','HC03_VC43','HC03_VC44','HC03_VC45','HC03_VC46','HC03_VC47','HC03_VC48','HC03_VC52','HC03_VC53','HC03_VC62','HC03_VC63','HC03_VC68','HC03_VC70','HC03_VC76','HC03_VC86','HC03_VC87','HC03_VC88','HC03_VC89','HC03_VC90','HC03_VC91','HC03_VC92','HC03_VC95','HC03_VC96','HC03_VC100','HC03_VC105','HC03_VC108','HC03_VC130')]

#rename cols to something half-way rememberable

colnames(d_samp1) <-c('no_un_18','married_only','married_with','male_only','male_with','female_only','female_with','roomies','singles','oldies','mo_1_18','avg_hs_size','rel_non_relatives','rel_non_rel_unmarried','males_married','males_sep','males_widowed','males_div','females_married','females_nv_mrd','females_sep','females_sep2','females_widows','females_div','wo_fertility','wo_fert_unmarried','grand_wit_kids','grand_wit_kids_res','grand_wit_kids_res_5yrs','grand_wit_kids_un18','per_in_school','ov25_less_9th','ov25_9th_12th','ov25_HS','ov25_smUni','ov25_assoc','ov25_bach','ov25_grad_deg','ov25_hs_plus','ov25_bach_plus','civs_ov18','per_disabled','per_disabled_kids','plc_o_birth')

# make sure state is on there! ergy-dergy
d_samp1$state <- basic_d$GEO.display.label


# get your reading and math scores

reading12thraw <- read.csv("data/reading-scores-12th.csv")

reading4thraw <- read.csv("data/reading-scores-4th.csv")

math4thraw <- read.csv("data/math-scores-4th.csv")

math12thraw <- read.csv("data/math-scores-12th.csv")

# drop the dumb error intervals. error intervals are for losers  ;-)
reading12th <- reading12thraw[,1:3]
reading4th <-  reading4thraw[,1:3]
math12th <-  math12thraw[,1:3]
math4th <-  math4thraw[,1:3]

# rename cols to "state" so that we can do a merge of the DFs
colnames(reading4th)[2] <- c('state')
colnames(reading12th)[2] <- c('state')
colnames(math4th)[2] <- c('state')
colnames(math12th)[2] <- c('state')

# rename cols to score-specific so that we know what they are later
colnames(reading4th)[3] <- c('reading4th')
colnames(reading12th)[3] <- c('reading12th')
colnames(math4th)[3] <- c('math4th')
colnames(math12th)[3] <- c('math12th')

reading4th$reading4th <- as.numeric(as.character(reading4th$reading4th))
reading12th$reading12 <- as.numeric(as.character(reading12th$reading12th))

math4th$math4th <- as.numeric(as.character(math4th$math4th))
math12th$math12th <- as.numeric(as.character(math12th$math12th))


# get everything into one DF. just easier that way
# reading
all_reading <- merge(reading4th, reading12th, by=c("state", "Year"))
# need to make some character conversions...
all_reading$reading4th <- as.numeric(as.character(all_reading$reading4th))
# math
all_math <- merge(math4th, math12th, by=c("state", "Year"))
all_math$math4th <- as.numeric(as.character(all_math$math4th))

#there's a gap in 12th year math & reading scores (weird!)
# so let's make a different subset that excludes those so we can have nice clean state data
read_math_ss2 <- merge (reading4th, math4th, by=c("state", "Year"))
#read_math_ss2 <- as.data.frame(read_math_ss2)

# range of test scores by year (as factor)

test_by_year <- qplot(factor(Year), reading4th, data=read_math_ss2, geom="boxplot", fill=as.factor(read_math_ss2$Year)) + labs(title="Reading Score Distributions: \n All Schools, 2007 - 2015", fill="Test Year", x="", y="Reading Scores, 4th Grade")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/test_by_year_read.png", plot = last_plot())

test_by_year <- qplot(factor(Year), math4th, data=read_math_ss2, geom="boxplot", fill=as.factor(read_math_ss2$Year)) + labs(title="Math Score Distributions: \n All Schools, 2007 - 2015", fill="Test Year", x="", y="Math Scores, 4th Grade")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/test_by_year_math.png", plot = last_plot())


# SO, let's make some subdata-set that we can then plot better
# and then maybe make some better YoY comparisons with
# (reading) years = c(2007, 2009, 2011, 2013, 2015)
reading4th_15 <- reading4th[reading4th$Year == 2015, ]
reading4th_13 <- reading4th[reading4th$Year == 2013, ]
reading4th_11 <- reading4th[reading4th$Year == 2011, ]
reading4th_09 <- reading4th[reading4th$Year == 2009, ]
reading4th_07 <- reading4th[reading4th$Year == 2007, ]

# (math) years = c(2007, 2009, 2011, 2013, 2015)
math4th_15 <- math4th[math4th$Year == 2015, ]
math4th_13 <- math4th[math4th$Year == 2013, ]
math4th_11 <- math4th[math4th$Year == 2011, ]
math4th_09 <- math4th[math4th$Year == 2009, ]
math4th_07 <- math4th[math4th$Year == 2007, ]

# maybe some ranks would be interesting...
# using rank > 50 (since there are 74 in the data set...which includes some cities)
# math - TOP
top20_m_15 <- subset(math4th_15, rank(math4th) > 50)
top20_m_13 <- subset(math4th_13, rank(math4th) > 50)
top20_m_11 <- subset(math4th_11, rank(math4th) > 50)
top20_m_09 <- subset(math4th_09, rank(math4th) > 50)
top20_m_07 <- subset(math4th_07, rank(math4th) > 50)

# reading - TOP
top20_r_15 <- subset(reading4th_15, rank(reading4th) > 50)
top20_r_13 <- subset(reading4th_13, rank(reading4th) > 50)
top20_r_11 <- subset(reading4th_11, rank(reading4th) > 50)
top20_r_09 <- subset(reading4th_09, rank(reading4th) > 50)
top20_r_07 <- subset(reading4th_07, rank(reading4th) > 50)

# maybe stack all these together so we can do some cool "dodge" stuff
top_performers_math <- rbind(top20_m_15, top20_m_13, top20_m_11, top20_m_09, top20_m_07)
top_performers_read <- rbind(top20_r_15, top20_r_13, top20_r_11, top20_r_09, top20_r_07)

# all around top_perfomers (common to both)
top_both <- merge (top_performers_math, top_performers_read, by=c("state", "Year"))

top_both$reading4th <- as.numeric(as.character(top_both$reading4th))
top_both$math4th <- as.numeric(as.character(top_both$math4th))

top_both <- na.omit(top_both)

# BOTTOM 
bot20_m_15 <- subset(math4th_15, rank(math4th) < 50)
bot20_m_13 <- subset(math4th_13, rank(math4th) < 50)
bot20_m_11 <- subset(math4th_11, rank(math4th) < 50)
bot20_m_09 <- subset(math4th_09, rank(math4th) < 50)
bot20_m_07 <- subset(math4th_07, rank(math4th) < 50)

# reading - BOTTOM
bot20_r_15 <- subset(reading4th_15, rank(reading4th) < 50)
bot20_r_13 <- subset(reading4th_13, rank(reading4th) < 50)
bot20_r_11 <- subset(reading4th_11, rank(reading4th) < 50)
bot20_r_09 <- subset(reading4th_09, rank(reading4th) < 50)
bot20_r_07 <- subset(reading4th_07, rank(reading4th) < 50)

# stack all these together
bot_performers_math <- rbind(bot20_m_15, bot20_m_13, bot20_m_11, bot20_m_09, bot20_m_07)
bot_performers_read <- rbind(bot20_r_15, bot20_r_13, bot20_r_11, bot20_r_09, bot20_r_07)

# all around bot_perfomers (common to both)
bot_both <- merge (bot_performers_math, bot_performers_read, by=c("state", "Year"))

bot_both$reading4th <- as.numeric(as.character(bot_both$reading4th))
bot_both$math4th <- as.numeric(as.character(bot_both$math4th))

bot_both <- na.omit(bot_both)

r_m_13 <- merge(reading4th_13, math4th_13, by=c("state", "Year"))
# drop the year
r_m_v2 <- r_m_13[,c(1,3, 4)]

# let's start merging some stuff up! 
d_samp2 <- merge(d_samp1, r_m_v2, by="state")

# merge in the income data 

income_raw <- read.csv("/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/data/income-data.csv")
income_v2 <- t(income_raw)
income_v3 <- income_v2
# clean up the col names
colnames(income_v3) <- income_v3[1,]
# drop useless rows
income_v3 <- income_v3[-c(1,2,3),]

# add state as a col
states <- as.factor(rownames(income_v3))
states_v <- as.vector(states) # combining these seems to be a problem. 

# version to new df
income_v4 <- cbind(income_v3, states_v)
colnames(income_v4)[colnames(income_v4) == 'states_v'] <- 'state'


# merge into d_samp3
d_samp3 <- merge(d_samp2, income_v4, by="state")
#d_samp3 <- as.data.frame(d_samp3)

# maybe this set should also have boxplots??  --> for highest and lowest performers

d_samp2b <- d_samp2
d_samp2b$state <- tolower(d_samp2$state)
d_samp2b$reading4th <- as.numeric(as.character(d_samp2b$reading4th))
d_samp2b$math4th <- as.numeric(as.character(d_samp2b$math4th))
d_samp4 <- merge(d_samp2b, states_map, by.x="state", by.y="region")


quant_r <- quantile(d_samp4$reading4th, c(0, .2, .4, .6, .8, 1.0))
d_samp4$quant_r <- cut(d_samp4$reading4th, quant_r, labels=c("0-20%", "20-40%", "40-60%", "60-80%", "80-100%"))
# detected a null for NM so going to put it at 0-20%
d_samp4$quant_r[is.na(d_samp4$quant_r)] <- "0-20%"
pal_r <- colorRampPalette(c("#663300", "#ff8000", "#ffe6cc"))(5)

map_read <- ggplot(d_samp4, aes(map_id = state, fill=quant_r)) + geom_map(map=states_map, color="white") + scale_fill_manual(values=pal_r) + expand_limits( x= states_map$long, y=states_map$lat)  + coord_map("polyconic") + labs(fill="Reading Scores \nPrecentile", title="Reading Scores: 2013") + theme(axis.title= element_blank(), axis.text=element_blank())

#map_read <- ggplot(d_samp4, aes(x=long, y=lat, group=group, fill=reading4th)) + geom_polygon(color="black") + coord_map("polyconic")
ggsave(file="images/map_read.png", plot = last_plot())

quant_m <- quantile(d_samp4$math4th, c(0, .2, .4, .6, .8, 1.0))
d_samp4$quant_m <- cut(d_samp4$math4th, quant_m, labels=c("0-20%", "20-40%", "40-60%", "60-80%", "80-100%"))


pal_m <- colorRampPalette(c("#141452", "#7070db", "#ebebfa"))(5)

map_math <- ggplot(d_samp4, aes(map_id = state, fill=quant_m)) + geom_map(map=states_map, color="white") + scale_fill_manual(values=pal_m) + expand_limits( x= states_map$long, y=states_map$lat)  + coord_map("polyconic") + labs(fill="Math Scores \nPrecentile", title="Math Scores: 2013") + theme(axis.title= element_blank(), axis.text=element_blank())
#map_math <- ggplot(d_samp4, aes(x=long, y=lat, group=group, fill=math4th)) + geom_polygon(color="black") + coord_map("polyconic")
ggsave(file="images/map_math.png", plot = last_plot())

