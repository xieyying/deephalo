#import相关模块
from .methods_main import flow_base

class my_mzml:
    def __init__(self,file,para) -> None:
        self.file = file
        self.para = para
        self.model_path = r'./trained_models/pick_halo_ann.h5'
    def work_flow(self):
        return flow_base(self.file,self.model_path,self.para) 
    
if __name__ == "__main__":
    pass