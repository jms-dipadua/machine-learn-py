# make sure basic libraries are installed
# ggplot2, sqldf, plyr, gcookbook
# and loaded
rm(list=ls()) # restart workstation
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

basic_d <- read.csv("/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/data/regional-dems.csv", sep=",", header=TRUE)

#basic_d_1 <- basic_d[-1,]
#View(basic_d_1)

d_samp1 <- basic_d[, c('HC03_VC05','HC03_VC06','HC03_VC07','HC03_VC08','HC03_VC09','HC03_VC10','HC03_VC11','HC03_VC12','HC03_VC13','HC03_VC14','HC03_VC17','HC01_VC21','HC03_VC31','HC03_VC32','HC03_VC36','HC03_VC39','HC03_VC40','HC03_VC41','HC03_VC43','HC03_VC44','HC03_VC45','HC03_VC46','HC03_VC47','HC03_VC48','HC03_VC52','HC03_VC53','HC03_VC62','HC03_VC63','HC03_VC68','HC03_VC70','HC03_VC76','HC03_VC86','HC03_VC87','HC03_VC88','HC03_VC89','HC03_VC90','HC03_VC91','HC03_VC92','HC03_VC95','HC03_VC96','HC03_VC100','HC03_VC105','HC03_VC108','HC03_VC130')]

#rename cols to something half-way rememberable

colnames(d_samp1) <-c('no_un_18','married_only','married_with','male_only','male_with','female_only','female_with','roomies','singles','oldies','mo_1_18','avg_hs_size','rel_non_relatives','rel_non_rel_unmarried','males_married','males_sep','males_widowed','males_div','females_married','females_nv_mrd','females_sep','females_sep2','females_widows','females_div','wo_fertility','wo_fert_unmarried','grand_wit_kids','grand_wit_kids_res','grand_wit_kids_res_5yrs','grand_wit_kids_un18','per_in_school','ov25_less_9th','ov25_9th_12th','ov25_HS','ov25_smUni','ov25_assoc','ov25_bach','ov25_grad_deg','ov25_hs_plus','ov25_bach_plus','civs_ov18','per_disabled','per_disabled_kids','plc_o_birth')

# make sure state is on there! ergy-dergy
d_samp1$state <- basic_d$GEO.display.label


# get your reading and math scores

reading12thraw <- read.csv("/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/data/reading-scores-12th.csv")

reading4thraw <- read.csv("/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/data/reading-scores-4th.csv")

math4thraw <- read.csv("/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/data/math-scores-4th.csv")

math12thraw <- read.csv("/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/data/math-scores-12th.csv")

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
# silly variation
#test_by_year2 <- qplot(factor(Year), as.numeric(reading4th), data=read_math_ss2, geom="boxplot", fill=Year)
#ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/test_by_year_2_dumb.png", plot = last_plot())

# basic bar graph of all reading across different years by state
# it's UGLY but a quick glance
#read_by_year <- ggplot(read_math_ss2, aes(x=as.factor(Year), y=reading4th, fill=state)) + geom_bar(position="dodge", stat="identity")
#ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/read_by_year.png", plot = last_plot())


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

# boxplot of performer score range vs all scores? 
# TOP PERFORMERS
top_math_perform_box <- qplot(as.factor(Year), math4th, data= top_both, geom="boxplot", fill=factor(Year))  + labs(title="Distribution of Top Perfomers: Math 2007 - 2013")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/top_math_perform_box.png", plot = last_plot())

top_read_perform_box <- qplot(as.factor(Year), reading4th, data= top_both, geom="boxplot", fill=factor(Year))  + labs(title="Distribution of Top Perfomers: Reading 2007 - 2013")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/top_read_perform_box.png", plot = last_plot())

# bargraphs of the top performers (math)
#top_perf_math <- ggplot(top20_m_15, aes(x=as.factor(state), y=math4th)) + geom_point(size=5, shape=21, fill="#7070db") + labs(title="Top Performing States: Math Score Distributions")  + coord_flip()
#ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/top_perf_math.png", plot = last_plot())

# bargraphs of all the top performers
#top_perf_bar <- ggplot(top_both, aes(x=as.factor(Year), y=math4th, fill=state)) + geom_bar(position="dodge", stat="identity")
#ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/top_perf_bar.png", plot = last_plot())

# BOTTOM PERFORMERS
bot_math_perform_box <- qplot(as.factor(Year), math4th, data= bot_both, geom="boxplot", fill=factor(Year))  + labs(title="Distribution of Bottom Perfomers: Math 2007 - 2013", x="", y="Math Scores, 4th Grade")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/bot_math_perform_box.png", plot = last_plot())

bot_read_perform_box <- qplot(as.factor(Year), reading4th, data= bot_both, geom="boxplot", fill=factor(Year))  + labs(title="Distribution of Bottom Perfomers: Reading 2007 - 2013",  x="", y="Reading Scores, 4th Grade")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/bot_read_perform_box.png", plot = last_plot())

# bargraphs of the bot performers (separately)
#bot_perf_math <- ggplot(bot20_m_15, aes(x=as.factor(state), y=math4th)) + geom_boxplot(fill="#7070db") + labs(title="Top Performing States: Math Score Distributions") + coord_flip()
#ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/bot_perf_math.png", plot = last_plot())

# another example merge (reading and math scores for 2013)
# we use 2013 because we have a ton of demographic data from 2013
r_m_13 <- merge(reading4th_13, math4th_13, by=c("state", "Year"))
# drop the year
r_m_v2 <- r_m_13[,c(1,3, 4)]

# let's start merging some stuff up! 
d_samp2 <- merge(d_samp1, r_m_v2, by="state")

# distribution of reading
read_hist <- ggplot(d_samp2, aes(x=reading4th)) + geom_histogram(color="darkgreen", fill="lightblue")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/read_hist.png", plot = last_plot())
# previous iterations
#ggplot(d_samp2, aes(x=reading4th)) + geom_histogram(color="darkgreen", fill="white")
#qplot(reading4th, data=d_samp2, geom="histogram")

# math
math_hist <- ggplot(d_samp2, aes(x=math4th)) + geom_histogram(color="darkgreen", fill="lightblue")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/math_hist.png", plot = last_plot())
# ggplot(d_samp2, aes(x=math4th)) + geom_histogram(color="darkgreen", fill="lightblue") + geom_density(fill="#FF6666") # shiat density! wtf??

#scatterplots be awesome
# females raising their kids (no daddy in house) vs reading
sing_mom_read <- ggplot(d_samp2, aes(x=reading4th, y=female_with)) + geom_point(size=5, shape=21, fill="lightblue", color="darkblue")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/sing_mom_read.png", plot = last_plot())
# and the correlations, obviously!
smr_cor <- cor(as.numeric(d_samp2$female_with), as.numeric(d_samp2$reading4th))
# same vs math
sing_mom_math <- ggplot(d_samp2, aes(x=math4th, y=female_with)) + geom_point(size=5, shape=21, fill="#7070db") + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Female-Only Housholds Raising Children", x="Reading Scores, 4th Grade", y="Percentage Housholds \n Female-Only Raising Children")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/sing_mom_math.png", plot = last_plot())
smm_cor <- cor(as.numeric(d_samp2$female_with), as.numeric(d_samp2$math4th))


# and with daddy
# reading
sing_d_read <- ggplot(d_samp2, aes(x=reading4th, y=male_with )) + geom_point(size=5, shape=21, fill="#e67300") + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Male-Only Housholds Raising Children", x="Reading Scores, 4th Grade", y="Percentage Housholds \n Male-Only Raising Children")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/sing_d_read.png", plot = last_plot())
sdr_core <- cor(as.numeric(d_samp2$male_with), as.numeric(d_samp2$reading4th))
# math
sing_d_math <- ggplot(d_samp2, aes(x=math4th, y=male_with )) + geom_point(size=5, shape=21, fill="#0066ff") + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Male-Only Households, Raising Children", x="Math Scores, 4th Grade", y="Percentage Households \n Male-Only Raising Own Children")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/sing_d_math.png", plot = last_plot())
sdr_cor <- cor(as.numeric(d_samp2$male_with), as.numeric(d_samp2$math4th))

# same for both but if the kids are staying with the grand parents
# just "with"
# reading
grand_wt_read <- ggplot(d_samp2, aes(x=reading4th, y=grand_wit_kids_res )) + geom_point(size=5, shape=21) +  stat_smooth(aes(group=1), method=lm)
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/grand_wt_read.png", plot = last_plot())
gwtr_cor <- cor(as.numeric(d_samp2$grand_wit_kids_res), as.numeric(d_samp2$reading4th))
# math
grand_wt_math <- ggplot(d_samp2, aes(x=math4th, y=grand_wit_kids_res )) + geom_point(size=5, shape=21, fill="#00b33c") + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Grandparents Resp. for Grandchildren, Math Scores", x="Math Scores, 4th Grade", y="Percentage of Housholds \n Grandparents Raising Own Grandchildren")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/grand_wt_math.png", plot = last_plot())
gwtm_cor <- cor(as.numeric(d_samp2$grand_wit_kids_res), as.numeric(d_samp2$math4th))

# 5 year rates 
# reading
grand_wt5_read <- ggplot(d_samp2, aes(x=reading4th, y=grand_wit_kids_res_5yrs )) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm)
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/grand_wt5_read.png", plot = last_plot())
gwtmr_cor <- cor(as.numeric(d_samp2$grand_wit_kids_res_5yrs), as.numeric(d_samp2$reading4th))

# math
grand_wt5_math <- ggplot(d_samp2, aes(x=math4th, y=grand_wit_kids_res_5yrs )) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Grandparents Resp. for Children, 5")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/grand_wt5_math.png", plot = last_plot())
gwt5m_cor <- cor(as.numeric(d_samp2$grand_wit_kids_res_5yrs), as.numeric(d_samp2$math4th))

# what's it look like for married couples
# reading
married_read <- ggplot(d_samp2, aes(x=reading4th, y=married_with )) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm)
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/married_read.png", plot = last_plot())
marriedr_cor <- cor(as.numeric(d_samp2$married_with), as.numeric(d_samp2$reading4th))

# math
married_math <- ggplot(d_samp2, aes(x=math4th, y=married_with )) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Married Couple and Math Scores", x= "4th Grade Math Scores", y="Percentage of Households: \nMarried Couples Raising Children", fill="#ff8000")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/married_math.png", plot = last_plot())
marriedm_cor <- cor(as.numeric(d_samp2$married_with), as.numeric(d_samp2$math4th))


# what about for general "percentage of people under 18"
 # DIDN'T FIND HELPFUL SO COMMENTED OUT
# reading
#ggplot(d_samp2, aes(x=as.numeric(reading4th), y=as.numeric(no_un_18))) + geom_point(size=5, shape=21) + stat_smooth(method=lm)
#cor(as.numeric(d_samp2$no_un_18), as.numeric(d_samp2$reading4th))
# math
#ggplot(d_samp2, aes(x=as.numeric(math4th), y=as.numeric(no_un_18))) + geom_point(size=5, shape=21) + stat_smooth(method=lm)
#cor(as.numeric(d_samp2$no_un_18), as.numeric(d_samp2$math4th))


# what about general education levels 
# reading
ov25_9less_read <- ggplot(d_samp2, aes(x=reading4th, y=ov25_less_9th )) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm)
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/ov25_9less_read.png", plot = last_plot())
ov25_9lessR_cor <- cor(as.numeric(d_samp2$ov25_less_9th), as.numeric(d_samp2$reading4th))
# math
ov25_9less_math <- ggplot(d_samp2, aes(x=math4th, y=ov25_less_9th )) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm)
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/ov25_9less_math.png", plot = last_plot())
ov25_9lessM_cor <- cor(as.numeric(d_samp2$ov25_less_9th), as.numeric(d_samp2$math4th))

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

# now let's look at income and education -- then we'll look at income and children reading

# low education  # APPEARS TO BE RIGHT . COME BACK TO . 
ls15k_ov25_l9 <- ggplot(d_samp3, aes(x=k10_15, y=ov25_less_9th)) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm)
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/ls15k_ov25_l9.png", plot = last_plot())
ls15ko25_cor <- cor(as.numeric(d_samp3$k10_15), as.numeric(d_samp3$ov25_less_9th))

# high salary and education
k200_bach_plus <- ggplot(d_samp3, aes(x=k200_p, y=ov25_bach_plus)) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm)
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k200_bach_plus.png", plot = last_plot())
k200_bachP_cor <- cor(as.numeric(d_samp3$k200_p), as.numeric(d_samp3$ov25_bach_plus))

k100_150_bach_plus <- ggplot(d_samp3, aes(x=k100_150, y=ov25_bach_plus)) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm) + labs(title="Ed. Bach. or More, Earnings $100 to $150k")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k100_150_bach_plus.png", plot = last_plot())
k100150_bachP_cor <- cor(as.numeric(d_samp3$k100_150), as.numeric(d_samp3$ov25_bach_plus))



# reading  # ALSO REDO **GRAPHS**  AS ::
ls10k_read <- ggplot(d_samp3, aes(x=reading4th, y=ls_10k )) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Earnings Less than 10k and Reading Score")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/ls10k_read.png", plot = last_plot())
# math
ls10k_math <- ggplot(d_samp3, aes(x=math4th, y=ls_10k )) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm)
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/ls10k_math.png", plot = last_plot())

ls10kr_cor <- cor(as.numeric(d_samp3$ls_10k), as.numeric(d_samp3$reading4th))
ls10km_cor <- cor(as.numeric(d_samp3$ls_10k), as.numeric(d_samp3$math4th))

# reading
ls15k_read <- ggplot(d_samp3, aes(x=reading4th, y=k10_15)) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Earnings Less than 10k and Reading Score")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/ls15k_read.png", plot = last_plot())
ls10km_cor <- cor(as.numeric(d_samp3$k10_15), as.numeric(d_samp3$reading4th))
# math
ls15k_math <- ggplot(d_samp3, aes(x=math4th, y=k10_15)) + geom_point(size=5, shape=21) + stat_smooth(aes(group=1), method=lm) + labs(fill="", title="Interactions: $10 to $15k and Math Scores", x="Math Score, 4th Grade", y="% $10 - $15K")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/ls15k_math.png", plot = last_plot())
ls15km_cor <- cor(as.numeric(d_samp3$k10_15), as.numeric(d_samp3$math4th))

# but let's look at more bins
k35_less_l9th <- ggplot(d_samp3, aes(x=k35_less, y=ov25_less_9th )) + geom_point(size=5, shape=21, fill="#7070db") + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Earnings Less than $35k, Less than 9th Grade Ed.", x="HH Earnings Less than 35k", y="Percentage: Over 25, Less than 9th Grade")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k35_less_l9th.png", plot = last_plot()) 
k35_less_l9th_cor <- cor(as.numeric(d_samp3$k35_less), as.numeric(d_samp3$ov25_less_9th))

# reading and math 
k35_less_read <- ggplot(d_samp3, aes(x=reading4th, y=k35_less )) + geom_point(size=5, shape=21, , fill="#e67300", color="darkblue") + stat_smooth(aes(group=1), method=lm) + labs(title="Interactions: Earnings $35k and Less,Reading Scores", x="Reading Scores, 4th Grade", y="Percentage Household Earnings \n $35k and Less") 
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k35_less_read.png", plot = last_plot())
k35_less_read_cor <- cor(as.numeric(d_samp3$k35_less), as.numeric(d_samp3$reading4th))

k35_less_math <- ggplot(d_samp3, aes(x=math4th, y=k35_less )) + geom_point(size=5, shape=21, fill="#7070db") + stat_smooth(aes(group=1), method=lm) + labs(title="Earning 35k and less and Math",x="Math Scores", y="Percentage Income: Less than $35k")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k35_less_math.png", plot = last_plot())
k35_less_math_cor <- cor(as.numeric(d_samp3$k35_less), as.numeric(d_samp3$math4th))
		
#ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k150_plus_read.png", plot = last_plot())
k150_plus_read_cor <- cor(as.numeric(d_samp3$k150_plus), as.numeric(d_samp3$reading4th))

k150_plus_math <- ggplot(d_samp3, aes(x=math4th, y=k150_plus )) + geom_point(size=5, shape=21, fill="#7070db") + stat_smooth(aes(group=1), method=lm)  + labs(title="Earning 150k and more and Math")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k150_plus_math.png", plot = last_plot())
k150_plus_math_cor <- cor(as.numeric(d_samp3$k150_plus), as.numeric(d_samp3$math4th))

k50_less_read <- ggplot(d_samp3, aes(x=reading4th, y=k50_less )) + geom_point(size=5, shape=21, fill="#e67300") + stat_smooth(aes(group=1), method=lm) + labs(title="Earning 50 to 100k and Reading", x="Reading Scores", y="Percentage Income: Less than $50k")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k50_less_read.png", plot = last_plot())
k50_less_read_cor <- cor(as.numeric(d_samp3$k50_less), as.numeric(d_samp3$reading4th))

k50_less_math <- ggplot(d_samp3, aes(x=math4th, y=k50_less )) + geom_point(size=5, shape=21, fill="#7070db") + stat_smooth(aes(group=1), method=lm) + labs(title="Earning 50 to 100k and Math", x="Math Scores", y="Percentage Income: Less than $50k")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k50_less_math.png", plot = last_plot())
k50_less_math_cor <- cor(as.numeric(d_samp3$k50_less), as.numeric(d_samp3$math4th))


k50_150_read <- ggplot(d_samp3, aes(x=reading4th, y=k50_150 )) + geom_point(size=5, shape=21, fill="#e67300") + stat_smooth(aes(group=1), method=lm) 
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k50_150_read.png", plot = last_plot())
k50_150_read_cor <- cor(as.numeric(d_samp3$k50_150), as.numeric(d_samp3$reading4th))

k50_150_math <- ggplot(d_samp3, aes(x=math4th, y=k50_150 )) + geom_point(size=5, shape=21, fill="#7070db") + stat_smooth(aes(group=1), method=lm)
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k50_150_math.png", plot = last_plot())
k50_150_math_cor <- cor(as.numeric(d_samp3$k50_150), as.numeric(d_samp3$math4th))


k50_100_read <- ggplot(d_samp3, aes(x=reading4th, y=k50_100 )) + geom_point(size=5, shape=21, fill="#e67300") + stat_smooth(aes(group=1), method=lm) + labs(title="Earning 50 to 100k and Reading", x="Reading Scores", y="Percentage Income: $50 to $100k")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k50_100_read.png", plot = last_plot())
k50_100_read_cor <- cor(as.numeric(d_samp3$k50_100), as.numeric(d_samp3$reading4th))

k50_100_math <- ggplot(d_samp3, aes(x=math4th, y=k50_150 )) + geom_point(size=5, shape=21, fill="#7070db") + stat_smooth(aes(group=1), method=lm) + labs(title="Earning 50 to 100k and Math", x="Math Scores", y="Percentage Income: $50 to $100k")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k50_100_math.png", plot = last_plot())
k50_100_math_cor <- cor(as.numeric(d_samp3$k50_150), as.numeric(d_samp3$math4th))

k10_35_read <- ggplot(d_samp3, aes(x=reading4th, y=k10_35 )) + geom_point(size=5, shape=21, fill="#e67300") + stat_smooth(aes(group=1), method=lm) + labs(title="Earning 10 to 35k and Reading", x="Reading Scores", y="Percentage Income: 10 to $35k")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k10_35_read.png", plot = last_plot())
k10_35_read_cor <- cor(as.numeric(d_samp3$k10_35), as.numeric(d_samp3$reading4th))

k10_35_math <- ggplot(d_samp3, aes(x=math4th, y=k10_35 )) + geom_point(size=5, shape=21, fill="#7070db") + stat_smooth(aes(group=1), method=lm) + labs(title="Earning 10 to 35k and Math", x="Math Scores", y="Percentage Income: 10 to $35k")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/k10_35_math.png", plot = last_plot())
k10_35_math_cor <- cor(as.numeric(d_samp3$k10_35), as.numeric(d_samp3$math4th))


all_correlations <- data.frame(smr_cor, smm_cor, sdr_core, sdr_cor, gwtr_cor, gwtm_cor, gwtmr_cor, gwt5m_cor, marriedr_cor, marriedm_cor, ov25_9lessR_cor, ov25_9lessM_cor, ls15ko25_cor, k200_bachP_cor, k100150_bachP_cor, ls10kr_cor, ls10km_cor, ls10km_cor, ls15km_cor, k35_less_read_cor, k35_less_math_cor, k150_plus_read_cor, k150_plus_math_cor, k50_less_read_cor, k50_less_math_cor, k50_150_read_cor, k50_150_math_cor, k50_100_read_cor, k50_100_math_cor, k10_35_read_cor, k10_35_math_cor)
all_cors_bar <- ggplot(all_correlations, aes(x=as.factor(colnames(all_correlations)), y=as.numeric(all_correlations), fill=as.numeric(all_correlations))) + geom_bar(stat="identity", position="identity") + labs(fill="Correlation Intensity", title="Correlations: Important Factors") + coord_flip()
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/all_cors_bar.png", plot = last_plot())

#d_samp3b <- d_samp3
#d_samp3b$state <- tolower(d_samp3b$state
#ggplot(d_samp3, aes(x=k15_25, y=reading4th, fill=grand_wit_kids_res_5yrs)) + geom_tile() + scale_fill_gradient2(midpoint=10, mid="grey70", limits=c(0,50))
#ggplot(d_samp3, aes(x=grand_wit_kids_res_5yrs, y=k15_25, fill=reading4th)) + geom_tile()  + stat_smooth(aes(group=1), method=lm)
#heatMap_grand5_read <- ggplot(d_samp3, aes(x=k15_25, y=reading4th, fill=grand_wit_kids_res_5yrs)) + geom_tile()  + stat_smooth(aes(group=1), method=lm)
#heatMap_ed_r1 <- ggplot(d_samp3, aes(x=ls_10k, y=reading4th, fill=as.numeric(ov25_less_9th))) + geom_raster()
	# ggplot(d_samp3, aes(x=as.numeric(ov25_less_9th), y=reading4th, fill=ls_10k)) + geom_raster()
#  + stat_smooth(aes(group=1), method=lm) + scale_fill_gradient2(midpoint=10, mid="grey70", limits=c(0,50))

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
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/map_read.png", plot = last_plot())

quant_m <- quantile(d_samp4$math4th, c(0, .2, .4, .6, .8, 1.0))
d_samp4$quant_m <- cut(d_samp4$math4th, quant_m, labels=c("0-20%", "20-40%", "40-60%", "60-80%", "80-100%"))


pal_m <- colorRampPalette(c("#141452", "#7070db", "#ebebfa"))(5)

map_math <- ggplot(d_samp4, aes(map_id = state, fill=quant_m)) + geom_map(map=states_map, color="gray") + scale_fill_manual(values=pal_m) + expand_limits( x= states_map$long, y=states_map$lat)  + coord_map("polyconic") + labs(fill="Reading Scores \nPrecentile", title="Math Scores: 2013") + theme(axis.title= element_blank(), axis.text=element_blank())

#map_math <- ggplot(d_samp4, aes(x=long, y=lat, group=group, fill=math4th)) + geom_polygon(color="black") + coord_map("polyconic")
ggsave(file="/Users/jadalm/Desktop/everything/JAMES_STUFF/Classes/data-vis/assignment_2/images/map_math.png", plot = last_plot())

