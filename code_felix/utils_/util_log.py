
import logging
import pandas as pd
format_str = '%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s'
format = logging.Formatter(format_str)
logging.basicConfig(level=logging.DEBUG, format=format_str)

logger = logging.getLogger()

handler = logging.FileHandler('./log/forecast.log', 'a')
handler.setFormatter(format)
logger.addHandler(handler)

def is_mini_args(item):
    if '__len__' in dir(item) and len(item) >=20:
        return False
    elif (type(item) in (tuple, list, dict) and len(item) <= 20) :
        return True
    elif type(item) in (tuple, list, dict, pd.DataFrame, pd.SparseDataFrame):
        return False
    else:
        return True




import functools
import time
def timed(logger=logger, level=None, format='%s: %s ms', paras=True):
    if level is None:
        level = logging.DEBUG


    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            start = time.time()
            import pandas as pd
            args_mini = [item  if is_mini_args(item) else type(item).__name__ for item in args  ]

            kwargs_mini = [ (k, v ) if is_mini_args(v) else (k, type(v).__name__) for k, v in kwargs.items()]
            arg_count = len(args) + len(kwargs)
            if paras:
                logger.info("Begin to run %s(%s paras) with:%r, %r" % (fn.__name__, arg_count, args_mini, kwargs_mini))
            else:
                logger.info(f"Begin to run {fn.__name__} with {arg_count} paras")
            result = fn(*args, **kwargs)
            duration = time.time() - start
            logging.info('cost:%7.2f sec: ===%r(%s paras), end (%r, %r) '
                         % (duration, fn.__name__, arg_count, args_mini, kwargs_mini, ))
            #logger.log(level, format, repr(fn), duration * 1000)
            return result
        return inner

    return decorator

def logger_begin_paras(paras):
    import socket
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    logger.debug(f'Start the program at:{host_name}, {host_ip}, with:{paras}')


logger_begin_paras("Load module")