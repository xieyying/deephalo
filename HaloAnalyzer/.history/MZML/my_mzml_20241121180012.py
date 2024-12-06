#import相关模块
from .methods_main import flow_base

class MyMzml:
    def __init__(self,file,para) -> None:
        self.file = file
        self.para = para
        self.EPM = None
        self.EPM_dense_output_model = None
        self.ADM = None

    def work_flow(self,blank=None,ms2=None):
        return flow_base(self.file,self.para, self.EPM, self.EPM_dense_output_model, self.ADM, blank=blank,ms2=ms2)   
    
if __name__ == "__main__":
    pass