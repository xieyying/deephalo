import pyopenms as oms

#read mzML file
file = r'C:\Users\xyy\Desktop\xcms_test\ChloroDBPFinder\demo_data\test.mzML'
exp = oms.MSExperiment()
oms.MzMLFile().load("test.mzML", exp)