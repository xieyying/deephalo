#创建统计函数运行时间的修饰器
import time
def timeit(method):
    """
    统计函数运行时间的修饰器
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r %2.2f sec' % \
              (method.__name__, te-ts))
        return result
    return timed

import os

def path_check(path):
    if not os.path.exists(path):
        os.mkdir(path)