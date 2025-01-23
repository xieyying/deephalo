import time
import os
#创建统计函数运行时间的修饰器
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

#创建路径检查函数
def path_check(path):
    if not os.path.exists(path):
        os.makedirs(path)