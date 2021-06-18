import logging


from split_data import DataSpliter
from train import train
from config import Config



if __name__ == "__main__":
    
    # logging.basicConfig(filename='debug.log',level=logging.INFO)
    # try:
    if True:
        
        
        data_split = DataSpliter.split_data(data_type=DataSpliter.DATA_TYPE_VESSELS,seed=42)
        
        config = Config()
        
        
        train(config,data_split)
      

        
            
    # except Exception as e:
    #     logging.critical(e, exc_info=True)