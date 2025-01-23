import pyopenms as oms

# Read mzXML file
file = r'C:\Users\xyy\Desktop\xcms_test\ChloroDBPFinder\demo_data\data1.mzXML'
exp = oms.MSExperiment()
oms.MzXMLFile().load(file, exp)

mtd = oms.MassTraceDetection()
mtd_par = mtd.getDefaults()
mass_traces = []
mtd.run(exp, mass_traces, 0)