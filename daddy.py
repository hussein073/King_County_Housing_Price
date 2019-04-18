def custom_k_fold(predictors, target, num_folds, verbose=False, log=False):
    """
    Leverage sklearn's KFold to run custom K-Fold validation.
 
    Parameters
    ----------
    predictors: pandas dataframe of predictors
    target: pandas series of target variable
    num_folds: integer of number of desired k-folds
    verbose: True if ya want print statements
    log: True if target variable was log transformed and want to
        apply np.exp

    Returns
    -------
    Return list of dictionaries of mse, rmse from train & test
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Get KFold generator
    kf = KFold(n_splits=num_folds) 
    
    counter = 1
    
    #Initialize list to hold metrics
    metrics = []
    
    # Get metric for every train/test combo
    for train_index, test_index in kf.split(predictors):
        
        #allocate train
        X_train = predictors.iloc[train_index,:]
        y_train = target.iloc[train_index]
        
        #allocate test
        X_test = predictors.iloc[test_index,:]
        y_test = target.iloc[test_index]

        #run regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        #get predictions
        y_hat_train = lr.predict(X_train)
        y_hat_test = lr.predict(X_test)

        #get MSEs
        if log:
            pred_train = np.exp(y_hat_train)
            act_train = np.exp(y_train)
            pred_test = np.exp(y_hat_test)
            act_test = np.exp(y_test)
        else:
            pred_train = y_hat_train
            act_train = y_train
            pred_test = y_hat_test
            act_test = y_test
            
        train_res = pred_train - act_train
        test_res = pred_test - act_test

        # Calculate Metrics
        mse_train = (train_res**2).mean()
        rmse_train = np.sqrt(mse_train)
        mape_train = mape(pred_train, act_train)
        
        mse_test = (test_res**2).mean()
        rmse_test = np.sqrt(mse_test)
        mape_test = mape(pred_test, act_test)

        metrics.append({'mse_train':mse_train
                   ,'rmse_train':rmse_train
                   ,'mape_train':mape_train
                   ,'mse_test':mse_test
                   ,'rmse_test':rmse_test
                   ,'mape_test':mape_test})
    
        if verbose:
            # Verbose results
            print('Fold ' + str(counter) + ':')
            print('********')
            print('Train Mean Squared Error (MSE):', mse_train)
            print('Train Root Mean Squared Error (RMSE):', rmse_train)
            print('Train Mean Absolute Percentage Error (MAPE):', mape_train)
            print('--------')
            print('Test Mean Squared Error (MSE):', mse_test)
            print('Test Root Mean Squared Error (RMSE):', rmse_test)
            print('Test Mean Absolute Percentage Error (MAPE):', mape_train)
            print('\n')
        
        counter += 1

    return metrics
  
    
def mape(predictions, actuals):
    """
    Calcuate Mean Absoute Percentage Error (MAPE)
    Reported as percentage
    
    Parameters
    ----------
    predictions:  numpy array of predicted values
    actuals: numpy array of actual values

    Returns
    -------
    Returns float of calculated MAPE
    """
    import numpy as np
    
    result = np.mean(np.abs((actuals - predictions)/actuals)) * 100
    return result
