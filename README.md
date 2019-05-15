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

# Part 2B: Model Selection (SVM)

When selecting a model, I took into consideration the results of the PCA. Because there are 34 features to work with, my first choice was the (Support Vector Machine) SVC since it is able to run a multi-class classification and work with vectors of many dimensions. Following the procedure on Jupyter, I created a train_test_split on the dataset, fitted and predicted the positions and ran metrics using a classification report and confusion matrix. (Note that due to the complexity of this dataset, this process may take a while so be patient)

# Insert first report 

From first glance, it appears that the model isn't successful since there is a 35% error rate based on F1-scores however this is not the fault of the algorithm. For example, observe the LW and RW positions which are the left and right wings respectively. The wings are responsible for receiving the ball from the mid players and will either set the ball for the striker or score for the team directly depending the team formation. Being on the left or right makes little difference for a wing player and as such the model can't differentiate between these two positions so that is one reason for the inaccuracy of the model. Also given the formation, the wings may interchange between being a mid or forward role and this provides another explanation for the low F1-scores. Two options to consider are to either generalize the positions even further to three roles or running a GridSearch in hopes of finding better grid parameter values.

Running the Gridsearch with 16 candidate parameters took approximately 20 minutes and even using the best parameters the result was no better than the SVC model so from now on we can ignore running a GridSearchCV. So now we turn to further classify the players by role (forward, mid or defender respectively). One issue to note is that I created the function assuming a 4-2-4 team formation and overlooking that some roles are hybrid roles for instance a CAM plays both mid and forward roles and is expected to interchange during the game. Following the same procedure, we obtain the following results: 

# Insert second report 

There is a significant improvement in the F1 scores however this is only due to sacrificing accuracy thus eliminating the conflicting noise that was present in the previous model. Checking the PCA figure, we note that while there is a clear distinction between a forward and defender, the model still has issues with differentiation of mid players. This goes back to the function that was tailored towards a particular team formation which focuses heavily on offense. It is to be noted that there are many different formations each with their strategies. 

# Insert third image 

To not have such a simplified model, I also decided to modify the dataset so that playing on the left or right could just be grouped as a side (in this case LW and RW become SW as an example). Once again following the same procedure, I ran a linear SVC model and obtained the following information: 

# Insert third report 

The results are not as good as those of the more generalized roles in soccers but there is no longer the issue of the model confusing the wing and back positions and therefore is independent of the side the player is on. One note to make is that the CAM and CDM positions have the worst F1-scores of all positions and this is due to these two positions being hybrid position which was previously mentioned in the limitations of the first test model. 

# Part 2C: Model Selection (Alternatives)

One of the bigger issues in the selection of the SVM model was the run time. In the instance of classifying 11 positions, it took the system approximately 5 minutes to obtain the classification results. Another possible algorithm to use is the Random Forest Classifier (RFC) which separates classes based on randomly selected features and repeats the process for a desired number of decision trees; that way there is no bias (only one feature/variable being the sole indicator of a split in the tree). As a example, I conducted a RFC on the third test and obtained the following results: 

# Insert RFC report 

A significant difference to note is the time elapsed to complete this task which was about 30 seconds. Even though the scores are lower this is a much greater improvement if time taken is a metric of importance. So in order to figure out which Machine Learning Algorithm would be better to use I conducted a Cross Validation score analysis for each model and I got the following results:

# Insert CV scores 

The CV score method ran each classification model with a chose k value of 5 which would split the dataset into combinations of train and test data groups performed over five iterations. The scores range between 0 and 1 and the closer to 1, the better the model performed in classifying the data. Although the RFC takes a shorter time to evaluate the data, the scores are varying and depending on the metrics of importance may not prove to be reliable whereas the SVC model is consistant with its performance however the classification takes a much longer time.

# Part 3: Data Visualization

Here we can showcase the visualization capabilities of Jupyter as well as find interesting information within the FIFA dataset. Of the possible 18000+ players, I wanted to learn more about these individuals so one attribute was their country of origin just to figure out their distribution. Using choromap, I was able to create an interactive visualization of the counts of players from all around the world.

# Insert choromap 

Just from the map, Western Europe and South America are producing the most players followed by a decent amount in North America and Northern Europe. Africa as a continent does not have many players in the FIFA league. While this is strange at first impression, delving further and doing some research it turns out that the European countries are pushing towards importings these players who are talented in their respective countries as well as being descendents of past players who played for the European national teams respectively. This is leading to a competition between the European and African countries to obtain players but as for why they do not play for their home country is still unclear. It is most likely that whoever is offering more salary wise and the reputation the player can potentially receive based on team affiliation that is explaning the emigration of these African players. 

I also ran some other basic statistical analysis on the dataset; in particular using a countplot, I created a function to figure out which roles ranked the highest given a particular statistic for a number of chosen players. This is helpful to note since each position has its specific role and as a manager one should be able to allocate the appropriate players. As an example, I decided to evaluate the top 100 players in Marking, Aggression, and Ball Control and obtained the following results:

# Insert countplot images 

If one were interested in training for a particular position or trying to figure out the most important attributes for a player this countplot would give enough information. Other parameters of interest include the overall score and value of a soccer player which I presented as a histogram with its respective KDE. 

# Insert KDE plots 

Using seaborn, I also created a jointplot involving the two features and set the plot as a KDE plot; the darker the region, the more heavily populated the number of players. 

# Insert jointplot 

Lastly using Matplotlib, I made a pie chart that showed the distribution of clubs the top 100 performing players belonged in. 



# Part 4: Discussion







