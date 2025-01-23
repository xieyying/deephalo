#import相关模块
from .methods_main import flow_base
import importlib_resources

class MyMzml:
    def __init__(self,file,para) -> None:
        self.file = file
        self.para = para
        # self.model_path = r'./trained_models/pick_halo_ann.h5' # for training process
        self.model_path = importlib_resources.files('HaloAnalyzer') / 'models/deephalo_ann_model.h5' 
    def work_flow(self,blank=None,ms2=None):
        return flow_base(self.file,self.model_path,self.para,blank=blank,ms2=ms2) 
    
if __name__ == "__main__":
    pass