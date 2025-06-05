
# import loguru
# from loguru import logger, _logger
# from rich.console import Console
# from functools import wraps
# from contextlib import contextmanager
# from typing import Callable
# from importlib import reload

# def default(a, b):
#     return a if a is not None else b

# class LoguruLogger:
#     _logger = logger
#     console = None
#     handler = []
    
#     _first_import = True
#     _default_file_format = "<green>[{time:MM-DD HH:mm:ss}]</green> <cyan>{name}</cyan>:<cyan>{line}</cyan> <level>[{level}]</level> - <level>{message}</level>"
#     _default_console_format = "[{time:MM-DD HH:mm:ss}] <cyan>{name}</cyan>:<cyan>{line}</cyan> <level>[{level}]</level> - <level>{message}</level>"
    
#     @classmethod
#     def logger(cls,
#                sink=None,
#                format=None,
#                filter=None,
#                **kwargs) -> "_logger.Logger":
#         reload(loguru)
        
#         if cls._first_import:
#             cls._logger.remove()  # the first time import
#             cls.console = Console(color_system=None)
#             handler = cls._logger.add(
#                 default(sink, lambda *x: cls.console.print(*x, end='')),
#                 colorize=True,
#                 format=default(format, cls._default_console_format),
#                 **kwargs
#             )
#             cls.handler.append(handler)
            
#             cls._first_import = False
        
#         else:
#             if sink is not None:
#                 handler = cls._logger.add(sink, format=default(format, cls._default_file_format), filter=filter, **kwargs)
                
#                 cls.handler.append(handler)
                
#         return cls._logger
    
#     @classmethod
#     def add(cls, *args, **kwargs):
#         handler = cls._logger.add(*args, **kwargs)
#         cls.handler.append(handler)
        
#     @classmethod
#     def remove_all(cls):
#         for h in cls.handler:
#             cls._logger.remove(h)
#         cls.handler = []
        
#     @classmethod
#     def remove_id(cls, id):
#         cls._logger.remove(id)
        
#     @classmethod
#     def bind(cls, *args, **kwargs):
#         # modify the handlers
#         # for hld in cls.handler:
#         #     cls._logger.
#         return cls._logger.bind(*args, **kwargs)


# def catch_any_error(func: "Callable | None" =None):
#     @contextmanager
#     def error_catcher():
#         try:
#             logger = LoguruLogger.logger()
#             yield logger
#         except Exception as e:
#             logger.error(f"catch error: {e}", raise_error=True)
#             logger.exception(e)
#         finally:
#             LoguruLogger.remove_all()

#     if func is None:
#         return error_catcher()
    
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         with error_catcher() as logger:
#             return func(*args, **kwargs)
    
#     return wrapper
        
        
# logger = LoguruLogger.logger()
# logger.add('test.log')
# logger.info('info')

# logger.warning('this is a warning')


# from test_log import may_beartype_raise


# may_beartype_raise([1])

import accelerate

accelerator = accelerate.Accelerator()

d = {'a': 1, 'b': 2}
print(accelerate.utils.gather_object(d))