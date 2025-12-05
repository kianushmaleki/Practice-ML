import sys, importlib, traceback
print('EXE:', sys.executable)
print('find_spec keras:', importlib.util.find_spec('keras'))
print('find_spec tensorflow:', importlib.util.find_spec('tensorflow'))
try:
    import keras
    print('keras version', getattr(keras,'__version__','no-version'))
except Exception:
    print('keras import failed')
    traceback.print_exc()
try:
    import tensorflow as tf
    print('tf version', getattr(tf,'__version__','no-version'))
except Exception:
    print('tf import failed')
    traceback.print_exc()
try:
    import tensorflow.keras as tkeras
    print('tf.keras available', getattr(tkeras,'__name__','ok'))
except Exception:
    print('tf.keras failed')
    traceback.print_exc()
