import pyopenms as oms

#read mzML file
file = r'C:\Users\xyy\Desktop\xcms_test\ChloroDBPFinder\demo_data\data1.mzXML'
exp = oms.MSExperiment()
oms.MzMLFile().load(file, exp)