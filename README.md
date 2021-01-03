# UrineAnalysis
Master Thesis

## Usage
### For Training
1-Create /test, /train folder in main directory.<br/>
2-Categorize your train and test data with same subfolder name. If you have 3 different cell type your folders look like this:<br/>
  * /test/rbc
  * /test/wbc
  * /test/rbc
  * /train/rbc
  * /train/wbc
  * /train/rbc<br/>
  
3-Insert you train and test images inside the subfolders.
4-In Run.py script, uncomment these functions in __main__ methot. Comment other functions.
  * bovw_train_and_save()

5-End of the training, visual words.csv and train.pkl files will appear on the root.
  
### For Testing Accuracy
1-In Run.py script, uncomment these functions in __main__ methot. Comment other functions.
  * visual_words, train_bovw = bovw_read_train_data()
  * bovw_train_test(visual_words, train_bovw)
  
### For Testing Image
1-If you want to see marked image you have to create folder in main direction and put your image in it. 
  * /single test/random/
  
2-After that in Run.py script, uncomment these functions in __main__ methot. Comment other functions.
  * visual_words, train_bovw = bovw_read_train_data()
  * bovw_single_test(visual_words, train_bovw)
