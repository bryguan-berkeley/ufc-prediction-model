# UFC Prediction Model

This project builds a logistic regression machine learning model to predict the winner of UFC fights by analyzing historical fighter and fight data, achieving a final accuracy of 66%.

## Introduction

The Ultimate Fighting Championship (UFC) has grown into one of the most popular sports globally, captivating millions of fans and bettors alike with its blend of martial arts, strategy, and high-stakes drama. The sport’s massive fanbase, combined with the rise of sports analytics, has driven a growing trend toward data-driven decision-making to enhance strategies and predictions. With a significant betting market surrounding UFC fights, accurate predictions offer not only potential financial gains but also deeper fan engagement. Understanding the factors influencing fighter performance—such as physical attributes, fighting style, and historical data—can provide a competitive edge for coaches, analysts, and bettors alike.

This UFC fight winner prediction model addresses these needs by leveraging machine learning techniques to analyze historical fight data, offering a data-driven approach to predict outcomes. By uncovering key trends and features associated with victory, the model provides valuable insights into the sport, aiding stakeholders in making informed decisions while enhancing the overall fan experience.

While there is a significant body of literature on the topic, much of it is gate-kept due to its profitability. This project, while fundamental and relatively simple, serves as a starting point for those looking to explore this space. It provides a foundation that can be learned from and built upon, paving the way for more complex and sophisticated models in the future.

## Table of Contents

- [Notebooks Introduction](#notebooks-introduction)
- [Notebook 1: Scraping](#notebook-1-scraping)
- [Notebook 2: Loading](#notebook-2-loading)
- [Notebook 3: Preprocessing](#notebook-3-preprocessing)
- [Notebook 4: Analyzing](#notebook-4-analyzing)
- [Notebook 5: Modeling](#notebook-5-exploring)
- [Results](#results)
- [Conclusion](#conclusion)

## Notebooks Introduction

The workflow is divided into five key Jupyter notebooks, each handling a distinct stage of the process. First, the scraping notebook collects the raw fight data from the UFC website. Next, the loading notebook organizes and stores this data in a structured format. The preprocessing notebook cleans and prepares the data, ensuring it is ready for analysis. In the exploration phase, key trends and relationships in the data are identified. Finally, the modeling notebook applies machine learning techniques to predict the winner of UFC fights based on the processed data.

## Notebook 1: Scraping

In the Scraping notebook, I decided to collect my own data rather than rely on existing datasets available online. Although I found some similar datasets, they were often outdated, and I couldn't be certain that the methods used by other authors were rigorous enough for my purposes. Since the UFC doesn't offer a public API, I chose to use Beautiful Soup to scrape the data directly from the UFC website.

The UFC website has two primary sections that were relevant for my project: fighter details and fight details. For fighter data, I started with the main fighters page, which lists all UFC fighters. For fight data, I scraped the main events page, which contains every event, each of which lists all the fights associated with that event. The challenge here was making sure I looped through all these nested pages to capture the complete data for both fighters and fights. Throughout the process, I had to do spot checks because some pages didn’t follow the typical structure, especially with older data going back to 1994. This made the task even more challenging, given the scale: 4,252 fighters and 7,836 fights in total.

After collecting the data, I exported it into a CSV format for easy loading in the next stages of the project. If you're looking for the most up-to-date UFC data, you can run this notebook.

## Notebook 2: Loading

In the Loading notebook, I started by reading the scraped CSV files and thoroughly investigating the structure and content of the data. The fighter dataset consists of 4,252 rows and 15 columns, capturing key attributes such as name, record, height, weight, reach, stance, birthday, and various performance metrics. These metrics include Significant Strikes Landed per Minute (SLPM), Significant Strike Accuracy (SA), and Average Takedowns Landed per 15 minutes (TDAVG), providing a detailed view of each fighter's historical performance, which is crucial for analyzing their fighting style and success.

The fight dataset contains 7,836 rows and 42 columns, with information such as Fighter A and Fighter B names, the winner, fight date, location, and specific fight statistics. Key metrics include A_KD (knockdowns by Fighter A), A_SS (significant strikes by Fighter A), and A_TD (takedowns by Fighter A), along with corresponding statistics for Fighter B. Additional details such as takedown percentage (TD%), significant strikes to various areas (head, body, legs), and control time (distance, clinch, ground) are also recorded. These in-depth stats are critical for building a model that accurately predicts fight outcomes.

To ensure data quality, I thoroughly examined the datasets for missing values, duplicates, unique values, and verified the data types of each column. After completing this quality check, I exported the cleaned dataframes as pickle files for use in the subsequent notebooks focused on preprocessing, exploration, and modeling.

## Notebook 3: Preprocessing

In the Preprocessing notebook, I tackled extensive cleaning of the data, as the raw data from the UFC website was fairly rough. First, I removed columns from the fights dataset that described in-fight statistics such as significant strikes, takedowns, and control time, keeping only the outcome (winner). Since this model is designed to be forward-looking, predicting who will win a fight, it’s crucial not to include information that comes from the actual fight itself.

Next, I dropped rows for fights that ended in a draw, as there were only 58 draws out of over 7,836 fights, and our focus is on predicting a winner. I also randomized the winners since the UFC website always listed Fighter B as the winner, and I didn’t want the model to learn any bias that Fighter B always wins. After that, I merged the fighter and fight datasets to ensure that each fight was matched with the relevant fighter details.

In terms of data cleaning, I converted the fighters' height, weight, and reach from strings to floats in the metric system for consistency. I also calculated each fighter’s age by subtracting their birthdate from the fight date. Since reach was the most commonly missing value, I imputed missing reach values by assuming that a fighter’s reach is, on average, equal to their height—this allowed me to retain more rows in the dataset without sacrificing too much accuracy. Lastly, I ensured that all columns were set to the appropriate data types.

After cleaning the data, I exported the resulting dataframes as pickle files to be used in the next notebooks for exploration and modeling.

## Notebook 4: Analyzing

In the Exploring notebook, I performed exploratory data analysis (EDA) to gain a deeper understanding of the dataset. I began by calculating the summary statistics for key fighter attributes such as age, height, weight, and reach, and visualized their distributions using histograms to observe how these variables are spread across the dataset. Additionally, I examined the distributions for categorical variables like win method, fighter stance, weight division, and fight dates, which provided valuable insights into trends within the UFC data. Finally, I created a correlation matrix to explore relationships between numerical features, helping to identify potential predictors for modeling fight outcomes. This exploratory analysis laid the groundwork for feature selection and model development in later stages.

## Notebook 5: Modeling

In the Modeling notebook, I began by selecting the features and outcome variable that would be used to train the machine learning models. The target outcome was the fight winner, and the features included each fighter's physical attributes—height, weight, reach, and age—as well as performance statistics. After selecting the relevant features, I shuffled and split the data into training and testing sets, ensuring that the numerical data was standardized to ensure consistent scaling across features. This preprocessing step prepared the dataset so that any model could easily use this "modeling-ready" data for training and evaluation.

As a starting point, I defined a baseline model that always predicts the majority class (the fighter with the most wins), which returned a test accuracy of 51%. This provided a benchmark to measure the performance of the machine learning models against. Each of the models I built was a binary logistic regression model, with no hidden layers, using a sigmoid activation function and a bias parameter. Both the weight and bias were initialized at one, and I used binary cross-entropy as the loss function. I employed Stochastic Gradient Descent (SGD) as the optimizer, and loss and accuracy were used as the evaluation metrics.

Next, I worked on building and evaluating several machine learning models. Each model followed the same process: dropping missing values, building the model, fitting the data, assessing and plotting performance, and finally, performing hyperparameter tuning (adjusting learning rate, batch size, and epochs) to optimize accuracy. The first model used only the physical attributes (height, weight, reach, and age) to predict the winner, resulting in a test accuracy of 48%, which was lower than the baseline.

For the second model, I expanded the feature set by adding each fighter's performance statistics, such as significant strikes and takedowns, to the physical attributes. After hyperparameter tuning, this model achieved a test accuracy of 66%, representing a 15% improvement over the baseline. Finally, in the third model, I introduced one-hot encoded fighter names to Model 2 in an attempt to capture additional fighter-specific characteristics. However, after tuning, this model's performance decreased, yielding a test accuracy of 53%, slightly above the baseline but lower than the second model.

## Results

The results of the modeling indicate that Model 2 is the most effective, showing a 15% improvement over the baseline, with a test accuracy of 66%. In hindsight, Model 1, which only used the fighter's age, height, weight, and reach, was likely too simplistic. These physical attributes are more innate characteristics and less reflective of a fighter's actual skill level, especially since fighters tend to optimize their weight to the maximum allowed in their weight class. This may have made weight a less meaningful predictor.

With Model 3, adding one-hot encoded fighter names introduced unnecessary complexity. Since many fighters don't compete frequently—some only once—there was insufficient data to properly train these name variables, which led to a decrease in accuracy. While 66% accuracy may not seem impressive on the surface, in contexts like sports betting, even a modest edge over the long term can lead to significant payoffs. This suggests that while there is room for improvement, Model 2’s performance offers meaningful potential in real-world applications.

## Conclusion

In conclusion, Model 2, with its 66% accuracy, demonstrates that even a basic model can make a difference in real-world applications, particularly in areas like sports betting, where a small edge can yield significant payoffs over time. However, there is still much to be desired in terms of improving the model. The machine learning techniques used in this project are quite rudimentary, and more advanced methods could lead to better results. We can enhance the model by adding hidden layers, using learning rate scheduling, experimenting with different optimizers, and incorporating regularization techniques like dropout or batch normalization. Beyond logistic regression, other models such as K-Nearest Neighbors (KNN), Recurrent Neural Networks (RNNs), Transformers, or even reinforcement learning could provide additional insights.

Moreover, rather than relying on overall fighter statistics from their entire careers, we could refine the model by calculating each fighter's statistics at the point in time when the fight occurs, using data from their previous fights. Incorporating additional data and modeling fighter interactions—such as how different fighting styles or stances match up—could also add complexity and improve predictions. These kinds of models are inherently more complex because they involve predicting the outcome of two entities interacting with each other.

Despite the potential for improvement, it’s important to acknowledge that sports are inherently human and unpredictable, with a level of randomness that no model can fully capture. Nonetheless, I hope this notebook serves as a starting point for more advanced modeling techniques and inspires further work in this space.

