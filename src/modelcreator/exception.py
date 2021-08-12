class ModelCreatorError(Exception):
    """Exception base class for modelcreator."""
    pass
    
class CorruptRecommenderError(ModelCreatorError):
    """Raised when required recommender model files are missing."""
    
    def __init__(self):
        self.message = 'Please call .initialize_recommender() and rebuild the recommender.'
        super().__init__(self.message)

class ModelNotFoundError(ModelCreatorError):
    """Raised when a model cannot be found on a specified path."""
    
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class NoTrainedModelError(ModelCreatorError):
    """Raised when a model was not trained yet but prediction-based functions are called."""
    
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class IncompatibleDataError(ModelCreatorError):
    """Raised when errors occur because of wrong data."""
    
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)