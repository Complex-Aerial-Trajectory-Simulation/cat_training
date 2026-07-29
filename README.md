# cat_training
trained an lstm model from keras lib


results
**val_loss 6.20** for 64 units with 115 on epochs with _sigma = 0.3_   
**val_loss 6.15** for 128 units with 39 with _sigma = 0.3_ epochs but but train_loss 4.14 was below val_loss  
**128 units is overfitting because both sigmas are 0.3(the same)**  
but I still need to have a loop for sigma for eliza to do the interpretation. 
sigmas are diff so the diff btw tr_loss and val_loss has no meaning  
on _varied sigma_ **val_loss: 6.38**  


**ELIZA: download** varsigma_64.keras, predictor_lstm.py with the instructions:  
#cand il folosesti o sa ai:  model = Predictor("varsigma_64.keras")  
#!!!trebuie sa instalezi tensorflow in terminal ca sa ai in enviornmentul tau ca altfel nu o sa mearga, nu am facut cu Pytorch ca  
#am facut cu codu meu care folosea tf nu pt!!!  
