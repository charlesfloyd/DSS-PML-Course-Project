# Practical Machine Learning Course Project
Charles Floyd  
August 21, 2014  

Predicting the Manner of Exercise

The task is to build a model to predict how well exercises are performed based on data collected by activity monitors worn during the exercise. The first step was to download and read in the training data.

```r
library(caret)
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', 
              method = 'curl', destfile = '/tmp/pml-training.csv')
training.full <- read.csv('/tmp/pml-training.csv')
```
Splitting the available data into test and training sets provides the opportunity to estimate a given model's out of sample error rate.

```r
set.seed(2004)
intrain <- createDataPartition(training.full$classe, p = .7, list = F)
training <- training.full[intrain,]
testing <- training.full[-intrain,]
nrow(training) ; nrow(testing)
```

```
## [1] 13737
```

```
## [1] 5885
```

```r
ncol(training)
```

```
## [1] 160
```
The data has many covariates, but some may not be add value to a model. Showing a summary may provide clues as to which variables can be removed as predictors

```r
summary(training)
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    2   adelmo  :2724   Min.   :1.32e+09     Min.   :   294      
##  1st Qu.: 4916   carlitos:2133   1st Qu.:1.32e+09     1st Qu.:254713      
##  Median : 9789   charles :2462   Median :1.32e+09     Median :502639      
##  Mean   : 9812   eurico  :2149   Mean   :1.32e+09     Mean   :502359      
##  3rd Qu.:14721   jeremy  :2420   3rd Qu.:1.32e+09     3rd Qu.:752367      
##  Max.   :19622   pedro   :1849   Max.   :1.32e+09     Max.   :998750      
##                                                                           
##           cvtd_timestamp new_window    num_window    roll_belt     
##  28/11/2011 14:14:1050   no :13453   Min.   :  1   Min.   :-28.90  
##  05/12/2011 11:24:1044   yes:  284   1st Qu.:222   1st Qu.:  1.09  
##  30/11/2011 17:11:1020               Median :424   Median :113.00  
##  05/12/2011 14:23: 980               Mean   :430   Mean   : 64.48  
##  02/12/2011 14:57: 967               3rd Qu.:643   3rd Qu.:123.00  
##  02/12/2011 13:34: 962               Max.   :864   Max.   :162.00  
##  (Other)         :7714                                             
##    pitch_belt        yaw_belt      total_accel_belt kurtosis_roll_belt
##  Min.   :-55.80   Min.   :-179.0   Min.   : 1.0              :13453   
##  1st Qu.:  1.80   1st Qu.: -88.3   1st Qu.: 3.0     #DIV/0!  :    8   
##  Median :  5.28   Median : -13.0   Median :17.0     -1.908453:    2   
##  Mean   :  0.33   Mean   : -11.2   Mean   :11.3     -0.016850:    1   
##  3rd Qu.: 15.20   3rd Qu.:  13.1   3rd Qu.:18.0     -0.021024:    1   
##  Max.   : 60.30   Max.   : 179.0   Max.   :28.0     -0.033935:    1   
##                                                     (Other)  :  271   
##  kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
##           :13453            :13453              :13453   
##  #DIV/0!  :   23     #DIV/0!:  284     #DIV/0!  :    7   
##  -2.060105:    3                       0.422463 :    2   
##  11.094417:    3                       -0.003095:    1   
##  47.000000:    3                       -0.010002:    1   
##  7.770402 :    3                       -0.014020:    1   
##  (Other)  :  249                       (Other)  :  272   
##  skewness_roll_belt.1 skewness_yaw_belt max_roll_belt   max_picth_belt 
##           :13453             :13453     Min.   :-94     Min.   : 3     
##  #DIV/0!  :   23      #DIV/0!:  284     1st Qu.:-88     1st Qu.: 5     
##  -3.072669:    3                        Median : -5     Median :18     
##  -0.189082:    2                        Mean   : -5     Mean   :13     
##  -0.587156:    2                        3rd Qu.: 15     3rd Qu.:19     
##  -0.733570:    2                        Max.   :180     Max.   :30     
##  (Other)  :  252                        NA's   :13453   NA's   :13453  
##   max_yaw_belt   min_roll_belt   min_pitch_belt   min_yaw_belt  
##         :13453   Min.   :-180    Min.   : 0             :13453  
##  -1.4   :   23   1st Qu.: -88    1st Qu.: 3      -1.4   :   23  
##  -1.1   :   21   Median :  -7    Median :17      -1.1   :   21  
##  -1.2   :   19   Mean   : -10    Mean   :11      -1.2   :   19  
##  -0.7   :   16   3rd Qu.:   4    3rd Qu.:17      -0.7   :   16  
##  -1.3   :   15   Max.   : 173    Max.   :23      -1.3   :   15  
##  (Other):  190   NA's   :13453   NA's   :13453   (Other):  190  
##  amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
##  Min.   :  0         Min.   : 0                  :13453     
##  1st Qu.:  0         1st Qu.: 1           #DIV/0!:    8     
##  Median :  1         Median : 1           0.00   :    6     
##  Mean   :  4         Mean   : 2           0.0000 :  270     
##  3rd Qu.:  2         3rd Qu.: 2                             
##  Max.   :360         Max.   :12                             
##  NA's   :13453       NA's   :13453                          
##  var_total_accel_belt avg_roll_belt   stddev_roll_belt var_roll_belt  
##  Min.   : 0           Min.   :-21     Min.   : 0       Min.   :  0    
##  1st Qu.: 0           1st Qu.:  1     1st Qu.: 0       1st Qu.:  0    
##  Median : 0           Median :117     Median : 0       Median :  0    
##  Mean   : 1           Mean   : 70     Mean   : 1       Mean   :  8    
##  3rd Qu.: 0           3rd Qu.:123     3rd Qu.: 1       3rd Qu.:  0    
##  Max.   :16           Max.   :154     Max.   :14       Max.   :201    
##  NA's   :13453        NA's   :13453   NA's   :13453    NA's   :13453  
##  avg_pitch_belt  stddev_pitch_belt var_pitch_belt   avg_yaw_belt  
##  Min.   :-51     Min.   :0         Min.   : 0      Min.   :-138   
##  1st Qu.:  2     1st Qu.:0         1st Qu.: 0      1st Qu.: -88   
##  Median :  5     Median :0         Median : 0      Median :  -6   
##  Mean   :  1     Mean   :1         Mean   : 1      Mean   :  -8   
##  3rd Qu.: 16     3rd Qu.:1         3rd Qu.: 0      3rd Qu.:  11   
##  Max.   : 41     Max.   :3         Max.   :10      Max.   : 173   
##  NA's   :13453   NA's   :13453     NA's   :13453   NA's   :13453  
##  stddev_yaw_belt  var_yaw_belt    gyros_belt_x      gyros_belt_y    
##  Min.   :  0     Min.   :    0   Min.   :-1.0400   Min.   :-0.6400  
##  1st Qu.:  0     1st Qu.:    0   1st Qu.:-0.0300   1st Qu.: 0.0000  
##  Median :  0     Median :    0   Median : 0.0300   Median : 0.0200  
##  Mean   :  2     Mean   :  153   Mean   :-0.0066   Mean   : 0.0391  
##  3rd Qu.:  1     3rd Qu.:    0   3rd Qu.: 0.1100   3rd Qu.: 0.1100  
##  Max.   :177     Max.   :31183   Max.   : 2.2200   Max.   : 0.6400  
##  NA's   :13453   NA's   :13453                                      
##   gyros_belt_z     accel_belt_x      accel_belt_y    accel_belt_z   
##  Min.   :-1.460   Min.   :-120.00   Min.   :-69.0   Min.   :-269.0  
##  1st Qu.:-0.200   1st Qu.: -21.00   1st Qu.:  3.0   1st Qu.:-162.0  
##  Median :-0.100   Median : -15.00   Median : 35.0   Median :-152.0  
##  Mean   :-0.132   Mean   :  -5.64   Mean   : 30.2   Mean   : -72.6  
##  3rd Qu.:-0.020   3rd Qu.:  -5.00   3rd Qu.: 61.0   3rd Qu.:  28.0  
##  Max.   : 1.620   Max.   :  83.00   Max.   :164.0   Max.   : 105.0  
##                                                                     
##  magnet_belt_x   magnet_belt_y magnet_belt_z     roll_arm     
##  Min.   :-52.0   Min.   :359   Min.   :-623   Min.   :-180.0  
##  1st Qu.:  9.0   1st Qu.:581   1st Qu.:-375   1st Qu.: -32.0  
##  Median : 35.0   Median :601   Median :-320   Median :   0.0  
##  Mean   : 55.6   Mean   :594   Mean   :-346   Mean   :  17.6  
##  3rd Qu.: 59.0   3rd Qu.:610   3rd Qu.:-306   3rd Qu.:  77.0  
##  Max.   :485.0   Max.   :673   Max.   : 293   Max.   : 179.0  
##                                                               
##    pitch_arm         yaw_arm        total_accel_arm var_accel_arm  
##  Min.   :-88.80   Min.   :-180.00   Min.   : 1.0    Min.   :  0    
##  1st Qu.:-25.60   1st Qu.: -43.10   1st Qu.:17.0    1st Qu.:  9    
##  Median :  0.00   Median :   0.00   Median :27.0    Median : 42    
##  Mean   : -4.59   Mean   :  -0.65   Mean   :25.6    Mean   : 56    
##  3rd Qu.: 11.10   3rd Qu.:  46.30   3rd Qu.:33.0    3rd Qu.: 81    
##  Max.   : 88.50   Max.   : 180.00   Max.   :66.0    Max.   :332    
##                                                     NA's   :13453  
##   avg_roll_arm   stddev_roll_arm  var_roll_arm   avg_pitch_arm  
##  Min.   :-167    Min.   :  0     Min.   :    0   Min.   :-82    
##  1st Qu.: -39    1st Qu.:  2     1st Qu.:    2   1st Qu.:-24    
##  Median :   0    Median :  6     Median :   40   Median :  0    
##  Mean   :  16    Mean   : 12     Mean   :  422   Mean   : -4    
##  3rd Qu.:  80    3rd Qu.: 17     3rd Qu.:  295   3rd Qu.: 11    
##  Max.   : 161    Max.   :161     Max.   :26067   Max.   : 76    
##  NA's   :13453   NA's   :13453   NA's   :13453   NA's   :13453  
##  stddev_pitch_arm var_pitch_arm    avg_yaw_arm    stddev_yaw_arm 
##  Min.   : 0       Min.   :   0    Min.   :-173    Min.   :  0    
##  1st Qu.: 2       1st Qu.:   6    1st Qu.: -27    1st Qu.:  4    
##  Median : 8       Median :  65    Median :   0    Median : 18    
##  Mean   :10       Mean   : 182    Mean   :   4    Mean   : 23    
##  3rd Qu.:15       3rd Qu.: 232    3rd Qu.:  40    3rd Qu.: 37    
##  Max.   :43       Max.   :1885    Max.   : 152    Max.   :177    
##  NA's   :13453    NA's   :13453   NA's   :13453   NA's   :13453  
##   var_yaw_arm     gyros_arm_x      gyros_arm_y      gyros_arm_z    
##  Min.   :    0   Min.   :-6.370   Min.   :-3.440   Min.   :-2.330  
##  1st Qu.:   15   1st Qu.:-1.330   1st Qu.:-0.800   1st Qu.:-0.070  
##  Median :  314   Median : 0.080   Median :-0.240   Median : 0.230  
##  Mean   : 1117   Mean   : 0.047   Mean   :-0.259   Mean   : 0.269  
##  3rd Qu.: 1362   3rd Qu.: 1.560   3rd Qu.: 0.140   3rd Qu.: 0.720  
##  Max.   :31345   Max.   : 4.870   Max.   : 2.840   Max.   : 3.020  
##  NA's   :13453                                                     
##   accel_arm_x      accel_arm_y      accel_arm_z      magnet_arm_x 
##  Min.   :-404.0   Min.   :-315.0   Min.   :-636.0   Min.   :-584  
##  1st Qu.:-242.0   1st Qu.: -54.0   1st Qu.:-144.0   1st Qu.:-304  
##  Median : -42.0   Median :  14.0   Median : -47.0   Median : 283  
##  Mean   : -60.3   Mean   :  32.6   Mean   : -71.7   Mean   : 190  
##  3rd Qu.:  83.0   3rd Qu.: 138.0   3rd Qu.:  23.0   3rd Qu.: 636  
##  Max.   : 437.0   Max.   : 308.0   Max.   : 271.0   Max.   : 782  
##                                                                   
##   magnet_arm_y   magnet_arm_z  kurtosis_roll_arm kurtosis_picth_arm
##  Min.   :-386   Min.   :-595           :13453            :13453    
##  1st Qu.:  -8   1st Qu.: 129   #DIV/0! :   50    #DIV/0! :   51    
##  Median : 202   Median : 444   -0.02438:    1    -0.01311:    1    
##  Mean   : 157   Mean   : 306   -0.04190:    1    -0.07394:    1    
##  3rd Qu.: 324   3rd Qu.: 545   -0.05051:    1    -0.10385:    1    
##  Max.   : 582   Max.   : 694   -0.05695:    1    -0.11279:    1    
##                                (Other) :  230    (Other) :  229    
##  kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
##          :13453           :13453            :13453             :13453  
##  #DIV/0! :    8   #DIV/0! :   49    #DIV/0! :   51     #DIV/0! :    8  
##  0.55844 :    2   -0.00696:    1    -0.01185:    1     0.55053 :    2  
##  -0.01749:    1   -0.01884:    1    -0.02652:    1     -0.00800:    1  
##  -0.02101:    1   -0.03484:    1    -0.03065:    1     -0.01697:    1  
##  -0.04059:    1   -0.04186:    1    -0.04528:    1     -0.03455:    1  
##  (Other) :  271   (Other) :  231    (Other) :  229     (Other) :  271  
##   max_roll_arm   max_picth_arm    max_yaw_arm     min_roll_arm  
##  Min.   :-73     Min.   :-173    Min.   : 4      Min.   :-89    
##  1st Qu.: -2     1st Qu.:   0    1st Qu.:29      1st Qu.:-40    
##  Median :  4     Median :  33    Median :34      Median :-22    
##  Mean   : 12     Mean   :  40    Mean   :35      Mean   :-20    
##  3rd Qu.: 27     3rd Qu.: 100    3rd Qu.:42      3rd Qu.:  0    
##  Max.   : 86     Max.   : 180    Max.   :62      Max.   : 66    
##  NA's   :13453   NA's   :13453   NA's   :13453   NA's   :13453  
##  min_pitch_arm    min_yaw_arm    amplitude_roll_arm amplitude_pitch_arm
##  Min.   :-180    Min.   : 1      Min.   :  0        Min.   :  0        
##  1st Qu.: -71    1st Qu.: 7      1st Qu.: 10        1st Qu.: 14        
##  Median : -33    Median :12      Median : 28        Median : 57        
##  Mean   : -33    Mean   :14      Mean   : 32        Mean   : 73        
##  3rd Qu.:   0    3rd Qu.:19      3rd Qu.: 47        3rd Qu.:122        
##  Max.   : 152    Max.   :38      Max.   :120        Max.   :360        
##  NA's   :13453   NA's   :13453   NA's   :13453      NA's   :13453      
##  amplitude_yaw_arm roll_dumbbell    pitch_dumbbell    yaw_dumbbell    
##  Min.   : 0        Min.   :-153.5   Min.   :-148.5   Min.   :-147.11  
##  1st Qu.:13        1st Qu.: -18.6   1st Qu.: -41.5   1st Qu.: -77.54  
##  Median :22        Median :  47.9   Median : -21.2   Median :  -5.69  
##  Mean   :21        Mean   :  23.7   Mean   : -11.0   Mean   :   1.40  
##  3rd Qu.:29        3rd Qu.:  67.6   3rd Qu.:  17.2   3rd Qu.:  79.25  
##  Max.   :51        Max.   : 153.6   Max.   : 149.4   Max.   : 154.95  
##  NA's   :13453                                                        
##  kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
##         :13453                 :13453                  :13453        
##  #DIV/0!:    4          -0.5464:    2           #DIV/0!:  284        
##  -0.2583:    2          -2.0851:    2                                
##  -2.0851:    2          #DIV/0!:    2                                
##  -0.0035:    1          -0.0233:    1                                
##  -0.0115:    1          -0.0280:    1                                
##  (Other):  274          (Other):  276                                
##  skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
##         :13453                 :13453                  :13453        
##  #DIV/0!:    3          -0.2328:    2           #DIV/0!:  284        
##  0.1110 :    2          0.1090 :    2                                
##  1.0312 :    2          -0.0053:    1                                
##  -0.0082:    1          -0.0084:    1                                
##  -0.0096:    1          -0.0166:    1                                
##  (Other):  275          (Other):  277                                
##  max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
##  Min.   :-70       Min.   :-113              :13453    Min.   :-150     
##  1st Qu.:-26       1st Qu.: -65       -0.3   :   14    1st Qu.: -59     
##  Median : 18       Median :  54       0.0    :   13    Median : -43     
##  Mean   : 16       Mean   :  39       -0.5   :   12    Mean   : -39     
##  3rd Qu.: 51       3rd Qu.: 134       -0.6   :   12    3rd Qu.: -18     
##  Max.   :137       Max.   : 155       0.2    :   12    Max.   :  73     
##  NA's   :13453     NA's   :13453      (Other):  221    NA's   :13453    
##  min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
##  Min.   :-146              :13453    Min.   :  0            
##  1st Qu.: -91       -0.3   :   14    1st Qu.: 16            
##  Median : -49       0.0    :   13    Median : 37            
##  Mean   : -28       -0.5   :   12    Mean   : 56            
##  3rd Qu.:  33       -0.6   :   12    3rd Qu.: 85            
##  Max.   : 121       0.2    :   12    Max.   :256            
##  NA's   :13453      (Other):  221    NA's   :13453          
##  amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
##  Min.   :  0                     :13453          Min.   : 0.0        
##  1st Qu.: 17              #DIV/0!:    4          1st Qu.: 4.0        
##  Median : 42              0.00   :  280          Median :11.0        
##  Mean   : 67                                     Mean   :13.8        
##  3rd Qu.:101                                     3rd Qu.:20.0        
##  Max.   :274                                     Max.   :58.0        
##  NA's   :13453                                                       
##  var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
##  Min.   : 0         Min.   :-129      Min.   :  0         
##  1st Qu.: 0         1st Qu.: -12      1st Qu.:  5         
##  Median : 1         Median :  47      Median : 13         
##  Mean   : 4         Mean   :  23      Mean   : 21         
##  3rd Qu.: 4         3rd Qu.:  63      3rd Qu.: 26         
##  Max.   :45         Max.   : 126      Max.   :108         
##  NA's   :13453      NA's   :13453     NA's   :13453       
##  var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
##  Min.   :    0     Min.   :-71        Min.   : 0           
##  1st Qu.:   21     1st Qu.:-38        1st Qu.: 4           
##  Median :  170     Median :-16        Median : 8           
##  Mean   : 1049     Mean   :-10        Mean   :13           
##  3rd Qu.:  693     3rd Qu.: 17        3rd Qu.:19           
##  Max.   :11612     Max.   : 94        Max.   :83           
##  NA's   :13453     NA's   :13453      NA's   :13453        
##  var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell
##  Min.   :   0       Min.   :-113     Min.   :  0         Min.   :    0   
##  1st Qu.:  13       1st Qu.: -74     1st Qu.:  4         1st Qu.:   16   
##  Median :  70       Median :  14     Median : 10         Median :  101   
##  Mean   : 358       Mean   :   6     Mean   : 17         Mean   :  625   
##  3rd Qu.: 367       3rd Qu.:  76     3rd Qu.: 26         3rd Qu.:  667   
##  Max.   :6836       Max.   : 135     Max.   :107         Max.   :11468   
##  NA's   :13453      NA's   :13453    NA's   :13453       NA's   :13453   
##  gyros_dumbbell_x  gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
##  Min.   :-204.00   Min.   :-2.07    Min.   : -2.4    Min.   :-419    
##  1st Qu.:  -0.03   1st Qu.:-0.14    1st Qu.: -0.3    1st Qu.: -51    
##  Median :   0.14   Median : 0.05    Median : -0.1    Median :  -9    
##  Mean   :   0.16   Mean   : 0.05    Mean   : -0.1    Mean   : -29    
##  3rd Qu.:   0.37   3rd Qu.: 0.21    3rd Qu.:  0.0    3rd Qu.:  10    
##  Max.   :   2.22   Max.   :52.00    Max.   :317.0    Max.   : 235    
##                                                                      
##  accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
##  Min.   :-189     Min.   :-334.0   Min.   :-643      Min.   :-3600    
##  1st Qu.:  -8     1st Qu.:-142.0   1st Qu.:-535      1st Qu.:  231    
##  Median :  41     Median :  -2.0   Median :-478      Median :  311    
##  Mean   :  53     Mean   : -38.9   Mean   :-327      Mean   :  219    
##  3rd Qu.: 113     3rd Qu.:  38.0   3rd Qu.:-299      3rd Qu.:  391    
##  Max.   : 315     Max.   : 318.0   Max.   : 592      Max.   :  633    
##                                                                       
##  magnet_dumbbell_z  roll_forearm    pitch_forearm     yaw_forearm    
##  Min.   :-262.0    Min.   :-180.0   Min.   :-72.50   Min.   :-180.0  
##  1st Qu.: -45.0    1st Qu.:   0.0   1st Qu.:  0.00   1st Qu.: -67.9  
##  Median :  13.0    Median :  23.1   Median :  9.06   Median :   0.0  
##  Mean   :  46.1    Mean   :  34.8   Mean   : 10.45   Mean   :  19.6  
##  3rd Qu.:  96.0    3rd Qu.: 140.0   3rd Qu.: 28.10   3rd Qu.: 110.0  
##  Max.   : 451.0    Max.   : 180.0   Max.   : 89.80   Max.   : 180.0  
##                                                                      
##  kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
##         :13453                :13453                 :13453       
##  #DIV/0!:   57         #DIV/0!:   57          #DIV/0!:  284       
##  -0.9169:    2         -0.0073:    1                              
##  -0.0227:    1         -0.0489:    1                              
##  -0.0359:    1         -0.0523:    1                              
##  -0.0567:    1         -0.0891:    1                              
##  (Other):  222         (Other):  223                              
##  skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
##         :13453                :13453                 :13453       
##  #DIV/0!:   56         #DIV/0!:   57          #DIV/0!:  284       
##  -0.0004:    1         -0.0113:    1                              
##  -0.0013:    1         -0.0131:    1                              
##  -0.0063:    1         -0.0405:    1                              
##  -0.0090:    1         -0.0478:    1                              
##  (Other):  224         (Other):  223                              
##  max_roll_forearm max_picth_forearm max_yaw_forearm min_roll_forearm
##  Min.   :-67      Min.   :-151             :13453   Min.   :-72     
##  1st Qu.:  0      1st Qu.:   0      #DIV/0!:   57   1st Qu.: -3     
##  Median : 29      Median : 111      -1.2   :   23   Median :  0     
##  Mean   : 27      Mean   :  83      -1.6   :   22   Mean   :  1     
##  3rd Qu.: 49      3rd Qu.: 175      -1.3   :   20   3rd Qu.: 12     
##  Max.   : 90      Max.   : 180      -1.0   :   17   Max.   : 62     
##  NA's   :13453    NA's   :13453     (Other):  145   NA's   :13453   
##  min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
##  Min.   :-180             :13453   Min.   :  0           
##  1st Qu.:-175      #DIV/0!:   57   1st Qu.:  2           
##  Median : -77      -1.2   :   23   Median : 19           
##  Mean   : -61      -1.6   :   22   Mean   : 25           
##  3rd Qu.:   0      -1.3   :   20   3rd Qu.: 40           
##  Max.   : 167      -1.0   :   17   Max.   :126           
##  NA's   :13453     (Other):  145   NA's   :13453         
##  amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
##  Min.   :  0                    :13453         Min.   :  0.0      
##  1st Qu.:  2             #DIV/0!:   57         1st Qu.: 29.0      
##  Median : 88             0.00   :  227         Median : 36.0      
##  Mean   :144                                   Mean   : 34.8      
##  3rd Qu.:350                                   3rd Qu.: 41.0      
##  Max.   :360                                   Max.   :108.0      
##  NA's   :13453                                                    
##  var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
##  Min.   :  0       Min.   :-177     Min.   :  0         Min.   :    0   
##  1st Qu.:  7       1st Qu.:   0     1st Qu.:  0         1st Qu.:    0   
##  Median : 25       Median :  14     Median :  9         Median :   73   
##  Mean   : 34       Mean   :  35     Mean   : 42         Mean   : 5177   
##  3rd Qu.: 53       3rd Qu.: 113     3rd Qu.: 83         3rd Qu.: 6857   
##  Max.   :173       Max.   : 177     Max.   :175         Max.   :30602   
##  NA's   :13453     NA's   :13453    NA's   :13453       NA's   :13453   
##  avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
##  Min.   :-68       Min.   : 0           Min.   :   0      Min.   :-155   
##  1st Qu.:  0       1st Qu.: 0           1st Qu.:   0      1st Qu.: -28   
##  Median : 13       Median : 6           Median :  34      Median :   0   
##  Mean   : 14       Mean   : 8           Mean   : 142      Mean   :  17   
##  3rd Qu.: 29       3rd Qu.:13           3rd Qu.: 166      3rd Qu.:  85   
##  Max.   : 72       Max.   :48           Max.   :2280      Max.   : 169   
##  NA's   :13453     NA's   :13453        NA's   :13453     NA's   :13453  
##  stddev_yaw_forearm var_yaw_forearm gyros_forearm_x   gyros_forearm_y 
##  Min.   :  0        Min.   :    0   Min.   :-22.000   Min.   : -7.02  
##  1st Qu.:  1        1st Qu.:    1   1st Qu.: -0.220   1st Qu.: -1.46  
##  Median : 26        Median :  696   Median :  0.050   Median :  0.03  
##  Mean   : 46        Mean   : 4619   Mean   :  0.158   Mean   :  0.07  
##  3rd Qu.: 87        3rd Qu.: 7606   3rd Qu.:  0.560   3rd Qu.:  1.62  
##  Max.   :170        Max.   :29060   Max.   :  3.520   Max.   :311.00  
##  NA's   :13453      NA's   :13453                                     
##  gyros_forearm_z  accel_forearm_x  accel_forearm_y accel_forearm_z 
##  Min.   : -7.94   Min.   :-498.0   Min.   :-632    Min.   :-446.0  
##  1st Qu.: -0.18   1st Qu.:-177.0   1st Qu.:  60    1st Qu.:-182.0  
##  Median :  0.07   Median : -56.0   Median : 201    Median : -40.0  
##  Mean   :  0.15   Mean   : -60.5   Mean   : 164    Mean   : -55.4  
##  3rd Qu.:  0.49   3rd Qu.:  79.0   3rd Qu.: 312    3rd Qu.:  26.0  
##  Max.   :231.00   Max.   : 477.0   Max.   : 923    Max.   : 287.0  
##                                                                    
##  magnet_forearm_x magnet_forearm_y magnet_forearm_z classe  
##  Min.   :-1280    Min.   :-892     Min.   :-966     A:3906  
##  1st Qu.: -615    1st Qu.:   7     1st Qu.: 198     B:2658  
##  Median : -372    Median : 591     Median : 513     C:2396  
##  Mean   : -311    Mean   : 382     Mean   : 396     D:2252  
##  3rd Qu.:  -72    3rd Qu.: 737     3rd Qu.: 654     E:2525  
##  Max.   :  672    Max.   :1460     Max.   :1080             
## 
```
The summary does reveal some columns that could be removed. The first 5 variables identify the user and the activity time, which don't seem relevant for modeling activity quality. They can be removed.

```r
training <- training[,6:ncol(training)]
testing <- testing[,6:ncol(testing)]
ncol(training)
```

```
## [1] 155
```
Other variables have na for more than 90% of their values.  Those also seem to be candidates for removal.

```r
napcts <- sapply(1:ncol(training), 
         function(i) length(which(is.na(training[,i]))) / nrow(training))
training <- training[,-which(napcts > 0.9, arr.ind = T)]
testing <- testing[,-which(napcts > 0.9, arr.ind = T)]
ncol(training)
```

```
## [1] 88
```
After removing those variables, it's also worthwhile to see which of the remining variables have zero or close to zero variance.  They are also unlikely to add much to a model, and can be removed.

```r
nzvdata <- nearZeroVar(training, saveMetrics = T)
nzvdata.nonzv<- nzvdata[!nzvdata$nzv,]
training <- training[, rownames(nzvdata.nonzv)]
testing <- testing[, rownames(nzvdata.nonzv)]
ncol(training)
```

```
## [1] 54
```
Having been restricted to numerical, varying variables, the data is now ready for model building.  The first attempt will use trees.

```r
system.time(model.rpart <- train(classe ~ ., data = training, method = 'rpart'))
```

```
##    user  system elapsed 
##  100.95    1.91  103.59
```

```r
confusionMatrix(predict(model.rpart, testing), testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1478  471  494  432  143
##          B   42  396   33  186  148
##          C  128  272  499  346  291
##          D    0    0    0    0    0
##          E   26    0    0    0  500
## 
## Overall Statistics
##                                         
##                Accuracy : 0.488         
##                  95% CI : (0.475, 0.501)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.332         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.883   0.3477   0.4864    0.000   0.4621
## Specificity             0.634   0.9138   0.7866    1.000   0.9946
## Pos Pred Value          0.490   0.4919   0.3249      NaN   0.9506
## Neg Pred Value          0.932   0.8537   0.8788    0.836   0.8914
## Prevalence              0.284   0.1935   0.1743    0.164   0.1839
## Detection Rate          0.251   0.0673   0.0848    0.000   0.0850
## Detection Prevalence    0.513   0.1368   0.2610    0.000   0.0894
## Balanced Accuracy       0.759   0.6307   0.6365    0.500   0.7283
```
The runtime was short, but the out of sample accuracy is very low, under 50%, and there was an entire class unrepresented in its predictions, class D.  Compare this model to a model using lda.

```r
system.time(model.lda <- train(classe ~ ., data = training, method = 'lda'))
```

```
##    user  system elapsed 
##  27.974   1.801  30.021
```

```r
confusionMatrix(predict(model.lda, testing), testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1371  140   87   46   36
##          B   43  745   91   43  158
##          C  103  156  712  108  108
##          D  152   42  112  735  101
##          E    5   56   24   32  679
## 
## Overall Statistics
##                                         
##                Accuracy : 0.721         
##                  95% CI : (0.709, 0.732)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.647         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.819    0.654    0.694    0.762    0.628
## Specificity             0.927    0.929    0.902    0.917    0.976
## Pos Pred Value          0.816    0.690    0.600    0.644    0.853
## Neg Pred Value          0.928    0.918    0.933    0.952    0.921
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.233    0.127    0.121    0.125    0.115
## Detection Prevalence    0.285    0.184    0.202    0.194    0.135
## Balanced Accuracy       0.873    0.792    0.798    0.840    0.802
```
Faster and with better out of sample accuracy. If there are highly correlated variables, it's possible that preprocessing them with pca could yield better accuracy.

```r
M <- abs(cor(training[,-ncol(training)]))
diag(M) <- 0
which(M > 0.8, arr.ind = T)
```

```
##                  row col
## yaw_belt           4   2
## total_accel_belt   5   2
## accel_belt_y      10   2
## accel_belt_z      11   2
## accel_belt_x       9   3
## magnet_belt_x     12   3
## roll_belt          2   4
## roll_belt          2   5
## accel_belt_y      10   5
## accel_belt_z      11   5
## pitch_belt         3   9
## magnet_belt_x     12   9
## roll_belt          2  10
## total_accel_belt   5  10
## accel_belt_z      11  10
## roll_belt          2  11
## total_accel_belt   5  11
## accel_belt_y      10  11
## pitch_belt         3  12
## accel_belt_x       9  12
## gyros_arm_y       20  19
## gyros_arm_x       19  20
## magnet_arm_x      25  22
## accel_arm_x       22  25
## magnet_arm_z      27  26
## magnet_arm_y      26  27
## accel_dumbbell_x  35  29
## accel_dumbbell_z  37  30
## gyros_dumbbell_z  34  32
## gyros_forearm_z   47  32
## gyros_dumbbell_x  32  34
## gyros_forearm_z   47  34
## pitch_dumbbell    29  35
## yaw_dumbbell      30  37
## gyros_forearm_z   47  46
## gyros_dumbbell_x  32  47
## gyros_dumbbell_z  34  47
## gyros_forearm_y   46  47
```
There are such correlated variables, so next pca is used to create new covariates on which to train models with rpart and lda.

```r
training.pp <- preProcess(training[,-ncol(training)], method = 'pca', thresh = 0.8)
training.pc <- predict(training.pp, training[,-ncol(training)])
system.time(model.rpart.pca <- train(training$classe ~ ., method = 'rpart', data = training.pc))
```

```
##    user  system elapsed 
##  43.329   0.816  44.757
```

```r
system.time(model.lda.pca <- train(training$classe ~ ., method = 'lda', data = training.pc))
```

```
##    user  system elapsed 
##   4.987   0.939   5.955
```

```r
testing.pc <- predict(training.pp, testing[,-ncol(training)])
confusionMatrix(predict(model.rpart.pca, testing.pc), testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1246  414  621  252  379
##          B  185  393  280  173  183
##          C    0    0    0    0    0
##          D  142  300   64  409  200
##          E  101   32   61  130  320
## 
## Overall Statistics
##                                        
##                Accuracy : 0.402        
##                  95% CI : (0.39, 0.415)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.222        
##  Mcnemar's Test P-Value : <2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.744   0.3450    0.000   0.4243   0.2957
## Specificity             0.604   0.8270    1.000   0.8565   0.9325
## Pos Pred Value          0.428   0.3237      NaN   0.3668   0.4969
## Neg Pred Value          0.856   0.8403    0.826   0.8836   0.8546
## Prevalence              0.284   0.1935    0.174   0.1638   0.1839
## Detection Rate          0.212   0.0668    0.000   0.0695   0.0544
## Detection Prevalence    0.495   0.2063    0.000   0.1895   0.1094
## Balanced Accuracy       0.674   0.5860    0.500   0.6404   0.6141
```

```r
confusionMatrix(predict(model.lda.pca, testing.pc), testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1031  214  364  102  159
##          B   92  512  107  179  181
##          C  199  174  386   98  126
##          D  214  156   64  464  128
##          E  138   83  105  121  488
## 
## Overall Statistics
##                                         
##                Accuracy : 0.49          
##                  95% CI : (0.477, 0.502)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.352         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.616    0.450   0.3762   0.4813   0.4510
## Specificity             0.801    0.882   0.8771   0.8858   0.9069
## Pos Pred Value          0.551    0.478   0.3927   0.4522   0.5219
## Neg Pred Value          0.840    0.870   0.8694   0.8971   0.8800
## Prevalence              0.284    0.194   0.1743   0.1638   0.1839
## Detection Rate          0.175    0.087   0.0656   0.0788   0.0829
## Detection Prevalence    0.318    0.182   0.1670   0.1743   0.1589
## Balanced Accuracy       0.708    0.666   0.6267   0.6836   0.6790
```
Using pca didn't improve the quality of the rpart and lda models.  Interestingly, using pca, the rpart model still left an entire class unaccounted for in its predictions, this time class C.  Having tried faster alternatives to this point, the next attempt will use random forest on a small subset of the data. This could demonstrate a lower bound on its accuracy and an idea of its runtime, both of which could help determine whether it's worthwhile for building the final model. To reduce the runtime as much as possible, a custom trainControl is passed to minimize iterations and repetitions for random forest.

```r
minitrain <- training[createDataPartition(training$classe, p = .05, list = F),]
system.time (model.rf.trial0 <- train(classe ~ ., data = minitrain, method = 'rf', trainControl = trainControl(method = 'cv', number = 1, repeats = 0)))
```

```
##    user  system elapsed 
## 242.773   2.076 246.793
```

```r
confusionMatrix(predict(model.rf.trial0, testing), testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1643  113    1   11    8
##          B   11  907   37   16   56
##          C    6  108  976   73   32
##          D   11    8   10  839   44
##          E    3    3    2   25  942
## 
## Overall Statistics
##                                         
##                Accuracy : 0.902         
##                  95% CI : (0.894, 0.909)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.876         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.981    0.796    0.951    0.870    0.871
## Specificity             0.968    0.975    0.955    0.985    0.993
## Pos Pred Value          0.925    0.883    0.817    0.920    0.966
## Neg Pred Value          0.992    0.952    0.989    0.975    0.971
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.279    0.154    0.166    0.143    0.160
## Detection Prevalence    0.302    0.175    0.203    0.155    0.166
## Balanced Accuracy       0.975    0.886    0.953    0.928    0.932
```
On much less data, the model generated from rf had higher out of sample accuracy than the others, albeit in longer time. Before alotting the time for training on the entire set, a larger subset of the data will be used to determine if a combination of factors can yield a highly accurate model with rf in more reasonable time.

```r
minitrain <- training[createDataPartition(training$classe, p = .25, list = F),]
system.time (model.rf.trial <- train(classe ~ ., data = minitrain, method = 'rf', trainControl = trainControl(method = 'cv', number = 1, repeats = 0)))
```

```
##    user  system elapsed 
## 1589.93   10.84 1611.25
```

```r
confusionMatrix(predict(model.rf.trial, testing), testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670   20    0    0    0
##          B    0 1105   11    3   10
##          C    3   14 1012   17    2
##          D    1    0    3  942    9
##          E    0    0    0    2 1061
## 
## Overall Statistics
##                                        
##                Accuracy : 0.984        
##                  95% CI : (0.98, 0.987)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.98         
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.970    0.986    0.977    0.981
## Specificity             0.995    0.995    0.993    0.997    1.000
## Pos Pred Value          0.988    0.979    0.966    0.986    0.998
## Neg Pred Value          0.999    0.993    0.997    0.996    0.996
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.188    0.172    0.160    0.180
## Detection Prevalence    0.287    0.192    0.178    0.162    0.181
## Balanced Accuracy       0.996    0.983    0.989    0.987    0.990
```
Out of sample accuracy of 98% and runtime of just under 30 minutes represent an appropriate compromise between speed and accuracy.  This will constitute the final model of exercise quality based on activity monitoring data.

```r
model.rf.trial$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, trainControl = ..1) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 1.43%
## Confusion matrix:
##     A   B   C   D   E class.error
## A 974   2   0   0   1    0.003071
## B   9 652   3   1   0    0.019549
## C   0  13 586   0   0    0.021703
## D   0   3   8 551   1    0.021314
## E   0   1   2   5 624    0.012658
```
