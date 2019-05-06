# FIFA-2019-Data-Analysis-and-Machine-Learning
The purpose of this project is to practice and showcase Machine Learning Algorithms and Data Visualization through the 
Jupyter Notebook. I chose to work with the 2019 FIFA dataset available as a CSV file which was obtained from Kaggle.com since I could approach the project with my unique twist and because I have zero knowledge of soccer, I would be able to learn the rules of the sport through programming. 

# Part 1: Initialization 
To begin doing any work, I needed to modify the dataset to a form that would be useful. I imported the data visualization libraries and from looking at the dataframe, there were 89 columns available for over 18000+ different players. Asking my brother who has experience with 
the FIFA games, these columns contained information on each player such as value, statistics for various mechanics in game and potential scores if these players were assigned to play other roles. The purpose of these numbers is for managerial purposes when it comes to selecting player positions on the field and being in charge of unique teams and formations. Generally, it would not be advised to have a player deviate from their real life role so this information already made it easier to choose relevant columns to work with in the project. 

I decided that I would create a classification model to predict a player's position using mechanical stats and physiology as predictors. 
However, it is to be noted that some of the columns are not in a format acceptable for machine learning; here I made functions that would change the string objects into float objects. I also chose to convert the Value column using a log transformation since the numbers would be very large to work with later on and to remove any Null values so that NA errors would not occur. At the end, the modified dataframe 
contained information on 17900 players and contained 40 columns to work with. 

# Part 2A: Machine Learning Algorithms 
Now that the data has been modified into a form ready for classification, I conducted a Principal Component Analysis to make sense of the information available and possibly use Feature Engineering to reduce the number of predictors. To run these MLAs, the sklearn library is used which contains various algorithms to run classification, regression and clustering methods. Because it is impossible to visually make sense of 30+ dimensions, I ran PCA using two components; the analysis attempts to explain the reasons for variation in the data and so some predictors have more of an effect than other features in classification. I also made a correlation matrix to visualize which features were important. 

# Insert Image 1 and 2 

From just 2 components, there is a lot of noise on the left of the figure and one position that is clearly distinct from the rest. This position would be the Goalkeeper and it shows since these players have much higher values in the GK stats. From the position codes, there are 27 possible positions which is another reason for the noisy mess and a change will be necessary for the model. A note to make is that there are 11 players on the field and the reason for so many classes is due to the different formations in soccer. These allow teams to play a particular style whether it be more offensive or defensive and so the extra classes are roles that fit a respect style of soccer. Here I decided to write a function that would convert these roles to fit the original 11 positions! Another note to take into account is that since the Goalkeeper is clearly distinct from the rest of the players, we can remove this role from the model and drop the GK columns so that fewer features are used. 

# Part 2B: Model Selection

When selecting a model, I took into consideration the results of the PCA. Because there are 34 features to work with, my first choice was the SVC since it is able to run a multi-class classification and work with vectors of many dimensions. Following the procedure on Jupyter, I created a train_test_split on the dataset, fitted and predicted the positions and ran metrics using a classification report and confusion matrix. (Note that due to the complexity of this dataset, this process may take a while so be patient)

# Insert first report 

From first glance, it appears that the model isn't successful since there is a 35% error rate based on F1-scores however this is not the fault of the algorithm. For example, observe the LW and RW positions which are the left and right wings respectively. The wings are responsible for receiving the ball from the mid players and will either set the ball for the striker or score for the team directly depending the team formation. Being on the left or right makes little difference for a wing player and as such the model can't differentiate between these two positions so that is one reason for the inaccuracy of the model. Also given the formation, the wings may interchange between being a mid or forward role and this provides another explanation for the low F1-scores. Two options to consider are to either generalize the positions even further to three roles or running a GridSearch in hopes of finding better grid parameter values.

Running the Gridsearch with 16 candidate parameters took approximately 20 minutes and even using the best parameters the result was no better than the SVC model so from now on we can ignore running a GridSearchCV. So now we turn to further classify the players by role (forward, mid or defender respectively). One issue to note is that I created the function assuming a 4-2-4 team formation and overlooking that some roles are hybrid roles for instance a CAM plays both mid and forward roles and is expected to interchange during the game. Following the same procedure, we obtain the following results: 

# Insert second report 

There is a significant improvement in the F1 scores however this is only due to sacrificing accuracy to eliminating the conflicting noise that was present in the previous model. Checking the PCA figure, we note that while there is a clear distinction between a forward and defender, the model still has issue with differentiation mid players. This goes back to the function that was tailored towards a particular team formation which focuses heavily on offense. It is to be noted that there are many different formations each with their strategies. 

# Insert third image 

## Still being Updated! 



