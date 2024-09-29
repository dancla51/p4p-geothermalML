library("readxl")
library("openxlsx")

T = as.numeric(read_excel("DataFormattedSep1.xlsx", sheet="T"))
n = as.numeric(read_excel("DataFormattedSep1.xlsx", sheet="n"))

TFTtable = matrix(0, T, n-1)  # n-1 because no sheet exists for Well6



# Manipulate
for (well in 1:5) {
  df = read_excel("Well_Output_Test_TFT_Summary.xlsx", sheet=well, skip=1)
  colnames(df)[6] = "ReviewedResEng"
  df$Date = as.Date(df$Date)
  
  #df = df[!is.na(df$`Test Type`), ]
  map = df$`ReviewedResEng`=="Y" | df$`ReviewedResEng`=="y"
  map[is.na(map)] = FALSE
  df = df[map,]
  
  df$MassKgS = df$`Total Mass (t/hr)` * 0.2777777778
  df$Date = as.numeric(df$Date - as.Date("2012-12-31"))
  
  if (nrow(df)>0) {
    for (i in 1:nrow(df)) {
      if (df$Date[i] <= T & df$Date[i] > 0) {
        TFTtable[df$Date[i], well] = df$MassKgS[i]
      }
    }
  }
}

# Output
write.xlsx(as.data.frame(TFTtable), "YonlyFormattedSep1.xlsx", sheetName="TFTs")


