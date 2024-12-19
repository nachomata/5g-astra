import joblib
class Functions:
    
    @staticmethod
    def LoadScaler(name: str):
        '''
        This method loads a pkl file and returns the correspondand scaler.

        Inputs:
        name: str, name of the file to load without the extension

        Returns:
        scaler: StandardScaler, StandardScaler of the pkl file
        '''
        
        fileName = '{}.pkl'.format(name)
        
        scaler = joblib.load(fileName)
            
        return scaler
    

    @staticmethod
    def LoadModel(filepath: str):
        '''
        This method loads a model from a file in format joblib.

        Inputs:
        filepath: str, path to the file to load

        Returns:
        model: model, model loaded from the file
        '''
        
        # Import the necessary packages
        from joblib import load

        # Load the model from the file
        model = load(filepath)

        # Return the model
        return model