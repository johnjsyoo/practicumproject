### Practicum - Part 1 - Data Cleanup

#Prior to running this code, please read the attached project report. After downloading the dataset the UCI website, please proceed with the following data cleanup steps.

#loading data frames
sub_1_1 <- read.table("PAMAP2_Dataset/Protocol/subject101.dat",header=FALSE)
sub_1_2 <- read.table("PAMAP2_Dataset/Optional/subject101.dat",header=FALSE)
sub_2 <- read.table("PAMAP2_Dataset/Protocol/subject102.dat",header=FALSE)
sub_3 <- read.table("PAMAP2_Dataset/Protocol/subject103.dat",header=FALSE)
sub_4 <- read.table("PAMAP2_Dataset/Protocol/subject104.dat",header=FALSE)
sub_5_1 <- read.table("PAMAP2_Dataset/Protocol/subject105.dat",header=FALSE)
sub_5_2 <- read.table("PAMAP2_Dataset/Optional/subject105.dat",header=FALSE)
sub_6_1 <- read.table("PAMAP2_Dataset/Protocol/subject106.dat",header=FALSE)
sub_6_2 <- read.table("PAMAP2_Dataset/Optional/subject106.dat",header=FALSE)
sub_7 <- read.table("PAMAP2_Dataset/Protocol/subject107.dat",header=FALSE)
sub_8_1 <- read.table("PAMAP2_Dataset/Protocol/subject108.dat",header=FALSE)
sub_8_2 <- read.table("PAMAP2_Dataset/Optional/subject108.dat",header=FALSE)
sub_9_1 <- read.table("PAMAP2_Dataset/Protocol/subject109.dat",header=FALSE)
sub_9_2 <- read.table("PAMAP2_Dataset/Optional/subject109.dat",header=FALSE)

#creating the sub_id column for each df
sub_1_1$sub_id <- 1
sub_1_2$sub_id <- 1.1
sub_2$sub_id <- 2
sub_3$sub_id <- 3
sub_4$sub_id <- 4
sub_5_1$sub_id <- 5
sub_5_2$sub_id <- 5.1
sub_6_1$sub_id <- 6
sub_6_2$sub_id <- 6.1
sub_7$sub_id <- 7
sub_8_1$sub_id <- 8
sub_8_2$sub_id <- 8.1
sub_9_1$sub_id <- 9
sub_9_2$sub_id <- 9.1


#here we replace the null values with the last valid non-null value for each variable and invidual

#change variable "a" to each of the 14 datasets that were previously created and run the following code
a <- sub_9_2

a <- transform(a, V1 = na.locf(V1))
a <- transform(a, V2 = na.locf(V2))
a <- transform(a, V3 = na.locf(V3))
a <- transform(a, V4 = na.locf(V4))
a <- transform(a, V5 = na.locf(V5))
a <- transform(a, V6 = na.locf(V6))
a <- transform(a, V7 = na.locf(V7))
a <- transform(a, V8 = na.locf(V8))
a <- transform(a, V9 = na.locf(V9))
a <- transform(a, V10 = na.locf(V10))
a <- transform(a, V11 = na.locf(V11))
a <- transform(a, V12 = na.locf(V12))
a <- transform(a, V13 = na.locf(V13))
a <- transform(a, V14 = na.locf(V14))
a <- transform(a, V15 = na.locf(V15))
a <- transform(a, V16 = na.locf(V16))
a <- transform(a, V17 = na.locf(V17))
a <- transform(a, V18 = na.locf(V18))
a <- transform(a, V19 = na.locf(V19))
a <- transform(a, V20 = na.locf(V20))
a <- transform(a, V21 = na.locf(V21))
a <- transform(a, V22 = na.locf(V22))
a <- transform(a, V23 = na.locf(V23))
a <- transform(a, V24 = na.locf(V24))
a <- transform(a, V25 = na.locf(V25))
a <- transform(a, V26 = na.locf(V26))
a <- transform(a, V27 = na.locf(V27))
a <- transform(a, V28 = na.locf(V28))
a <- transform(a, V29 = na.locf(V29))
a <- transform(a, V30 = na.locf(V30))
a <- transform(a, V31 = na.locf(V31))
a <- transform(a, V32 = na.locf(V32))
a <- transform(a, V33 = na.locf(V33))
a <- transform(a, V34 = na.locf(V34))
a <- transform(a, V35 = na.locf(V35))
a <- transform(a, V36 = na.locf(V36))
a <- transform(a, V37 = na.locf(V37))
a <- transform(a, V38 = na.locf(V38))
a <- transform(a, V39 = na.locf(V39))
a <- transform(a, V40 = na.locf(V40))
a <- transform(a, V41 = na.locf(V41))
a <- transform(a, V42 = na.locf(V42))
a <- transform(a, V43 = na.locf(V43))
a <- transform(a, V44 = na.locf(V44))
a <- transform(a, V45 = na.locf(V45))
a <- transform(a, V46 = na.locf(V46))
a <- transform(a, V47 = na.locf(V47))
a <- transform(a, V48 = na.locf(V48))
a <- transform(a, V49 = na.locf(V49))
a <- transform(a, V50 = na.locf(V50))
a <- transform(a, V51 = na.locf(V51))
a <- transform(a, V52 = na.locf(V52))
a <- transform(a, V53 = na.locf(V53))
a <- transform(a, V54 = na.locf(V54))
a <- transform(a, V54 = na.locf(V54))
a <- transform(a, sub_id = na.locf(sub_id))

#replace individual with "a". Now you have to go back and do the same for the remaining 13 subjects
sub_9_2 <- a

#creating master raw_data
raw_data <- rbind(sub_1_1, sub_1_2, sub_2, sub_3, sub_4, sub_5_1, sub_5_2, sub_6_1, sub_6_2, sub_7, sub_8_1, sub_8_2, sub_9_1, sub_9_2)

#dropping columns on acceleration at 6g and orientation (invalid data)
raw_data$V8 <- NULL
raw_data$V9 <- NULL
raw_data$V10 <- NULL
raw_data$V17 <- NULL
raw_data$V18 <- NULL
raw_data$V19 <- NULL
raw_data$V20 <- NULL
raw_data$V25 <- NULL
raw_data$V26 <- NULL
raw_data$V27 <- NULL
raw_data$V34 <- NULL
raw_data$V35 <- NULL
raw_data$V36 <- NULL
raw_data$V37 <- NULL
raw_data$V42 <- NULL
raw_data$V43 <- NULL
raw_data$V44 <- NULL
raw_data$V51 <- NULL
raw_data$V52 <- NULL
raw_data$V53 <- NULL
raw_data$V54 <- NULL

# Activity names
num <- c(1,2,3,4,5,6,7,9,10,11,12,13,16,17,18,19,20,24,0)
act <- c('lying','sitting','standing','walking','running',
         'cycling','nordic_walking','watching_TV','computer_work',
         'car_driving','ascending_stairs','descending_stairs',
         'vacuum cleaning','ironing','folding_laundry','house_cleaning',
         'playing_soccer','rope_jumping','transient_activity')
act_df <- data.frame(num, act)

# Headers

headers <- c(
  "activity_id", #see pdf for list of activities
  "timestamp",
  "heart_rate", #heart rate in beats per minute
  "IMU_hand_temp",  #temperature in deg C
  "IMU_hand_acc_x", #3D-acceleration at 16g x
  "IMU_hand_acc_y", #3D-acceleration at 16g y
  "IMU_hand_acc_z", #3D-acceleration at 16g z
  "IMU_hand_gyro_x", #3D-gyroscope (rad/s) x
  "IMU_hand_gyro_y", #3D-gyroscope (rad/s) y
  "IMU_hand_gyro_z", #3D-gyroscope (rad/s) z
  "IMU_hand_mag_x", #3D-magnetometer (mu T) x
  "IMU_hand_mag_y", #3D-magnetometer (mu T) y
  "IMU_hand_mag_z", #3D-magnetometer (mu T) z
  "IMU_chest_temp",  #temperature in deg C
  "IMU_chest_acc_x", #3D-acceleration at 16g x
  "IMU_chest_acc_y", #3D-acceleration at 16g y
  "IMU_chest_acc_z", #3D-acceleration at 16g z
  "IMU_chest_gyro_x", #3D-gyroscope (rad/s) x
  "IMU_chest_gyro_y", #3D-gyroscope (rad/s) y
  "IMU_chest_gyro_z", #3D-gyroscope (rad/s) z
  "IMU_chest_mag_x", #3D-magnetometer (mu T) x
  "IMU_chest_mag_y", #3D-magnetometer (mu T) y
  "IMU_chest_mag_z", #3D-magnetometer (mu T) z
  "IMU_ankle_temp",  #temperature in deg C
  "IMU_ankle_acc_x", #3D-acceleration at 16g x
  "IMU_ankle_acc_y", #3D-acceleration at 16g y
  "IMU_ankle_acc_z", #3D-acceleration at 16g z
  "IMU_ankle_gyro_x", #3D-gyroscope (rad/s) x
  "IMU_ankle_gyro_y", #3D-gyroscope (rad/s) y
  "IMU_ankle_gyro_z", #3D-gyroscope (rad/s) z
  "IMU_ankle_mag_x", #3D-magnetometer (mu T) x
  "IMU_ankle_mag_y", #3D-magnetometer (mu T) y
  "IMU_ankle_mag_z", #3D-magnetometer (mu T) z
  "subject_id", # subject id
  "activity"
)


# Add activity names
raw_data2 <- merge(raw_data, act_df, by.x = "V2", by.y = "num", all.x = TRUE)

# Add headers
colnames(raw_data2) = headers

# Sanity print check
raw_data2[1:100,3]

# Save clean data to csv
write.csv(raw_data2, "PAMAP2_Dataset/Protocol/data.csv", row.names = TRUE)