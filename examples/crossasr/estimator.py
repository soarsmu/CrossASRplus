class Estimator:
    def __init__(self, name:str):
        self.name = name

    def getName(self) -> str :
        return self.name
    
    def setName(self, name:str):
        self.name = name
    
    def fit(self, X:[str], y:[int]):
        raise NotImplementedError()

    def predict(self, X:[str]):
        raise NotImplementedError()
