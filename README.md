# Predicting Students' Dropout and Academic Success

![istockphoto-1127339203-612x612](https://github.com/user-attachments/assets/3d1fa993-7850-4b90-b2dd-c280d168e44c)

![Language](https://img.shields.io/badge/language-Python-orange.svg)  
![Last Updated](https://img.shields.io/badge/last%20updated-May%202025-brightgreen)  
![Status](https://img.shields.io/badge/status-completed-green)


**Authors**: Andres Machado, Bret Geyer, Jackson Small, Jackson Colon, and Thomas Tibbetts<br>
*Spring 2025 | Data Science II*

#### Dataset Citation:
Realinho, Valentim, et al. "Predict Students' Dropout and Academic Success." UCI Machine Learning Repository, 2021, https://doi.org/10.24432/C5MC89.

## Project Statement
‭Student retention is a major problem for institutes of higher education, with many‬ resources devoted to analyzing what causes students to drop out or graduate late. When the‬ graduation rate falls, so does a school’s ranking and reputation. For this reason it’s desirable to‬ reach out to these students early in their academic path, so they can receive extra counseling or‬ attention that may improve their situation. For this project, we attempt to use data collected from‬ ‭university students to predict whether they will be graduates, dropouts, or still enrolled at the‬ end of a typical program term. Some variables of particular interest are the student’s age at‬ ‭enrollment, scholarship status, and early grades. We will use classification algorithms with‬ considerations for reducing class imbalance. The objective is to identify students at risk of failing‬ to graduate, so that they can be prioritized for assistance.‬
‭
## Description of Data:‬
‭Broadly speaking, this project's focus is predicting the academic success of university‬ students. We are working with a tabular dataset which lists 37 attributes for each of 4,424‬ students at the Polytechnic Institute of Portalegre, and contains contains 18 categorical features‬ which encode information such as the student’s program, marital status, application mode, and‬ their parents’ level of education. Some of the quantitative variables encode the student’s age‬ and numerical evaluations (0-100) of their admission profile and previous qualifications, and‬ other quantitative variables which detail the number of curricular units for which the student was‬ enrolled, approved, and credited during their first two semesters. The dataset’s target variable is‬ a categorical column which designates each student as one of the following:‬

* **Enrolled**‬‭: Indicating that the student was still enrolled‬‭ (not graduated) at the end of the‬ normal term of the program. Where 794 students classify as Enrolled.‬
* **Graduated‬‭**: Indicating the student graduated by the‬‭ end of the normal term of the‬ program. Where 2,209 students classify as Graduated.‬
* **Dropout**‬‭: Indicating that the student dropped out by‬‭ the end of the program’s normal‬ term. Where 1421 students classify as Dropouts.‬

More information about feature definitions and dataset origins can be found here:‬
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success‬

‭## Data Exploration:
One of the dataset’s issues is containing many categorical features, each of which‬ consists of many different levels. For example, the “Application mode” variable alone has 18‬ levels, some of which apply to fewer than 10 students. If all of these categorical features were‬ one-hot encoded, they would add hundreds of sparse columns to the dataset and create‬ unnecessary bulk during the data exploration. For this reason, it was decided that all categorical‬ feature levels representing less than 2% of the dataset would be binned into one level called‬ “Other”.This significantly reduced the unnecessary complexity in the dataset.‬

‭
‭
‭
‬
