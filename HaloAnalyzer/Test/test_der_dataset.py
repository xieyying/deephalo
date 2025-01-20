from HaloAnalyzer.Dereplication.Database_processing import DereplicationDataset

if __name__ == '__main__':
    # Create a Dataset object and execute the workflow
    database_file = r'E:\github\renew\HaloAnalyzer\HaloAnalyzer\Test\Resoure\Demo_dereplication_database.csv'
    test = DereplicationDataset(database_file, 'Formula')
    test.work_flow()
