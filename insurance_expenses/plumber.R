
# load plumber and tidyverse 

library(plumber)
library(tidyverse)

# load the model 

model <- read_rds("insurance_model.rds")


# plumber essentials 

#* @apiTitle Insurance expenses prediction

#* @get /expenses

#* @apiDescription This API serves up predictions of insurance expenses based on age, gender, no of children, region, BMI, smoking status

#* @param  age  age 

#* @param  sex gender 

#* @param bmi bmi

#* @param children children

#* @param smoker smoker 

#* @region region


function(age, sex, bmi, children, smoker, region){
  
  to_predict <- data.frame(age= as.numeric(age),
                           sex= as.factor(sex),
                           bmi= as.numeric(bmi),
                           children = as.numeric(chidren),
                           smoker = as.factor(smoker),
                           region = as.factor(region))
  
  predict(model,to_predict )
}
