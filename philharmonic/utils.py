import os, errno
import warnings

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""
    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def input_loc(filepath):
    """get the file path based on the configured input folder"""
    from philharmonic import conf
    if conf.add_date_to_folders:
        return loc_date(os.path.join(filepath),
                        input_loc=True)
    else:
        return loc_normal(os.path.join(filepath),
                        input_loc=True)

def output_loc(filepath):
    """get the file path based on the configured output folder"""
    from philharmonic import conf
    if conf.add_date_to_folders:
        return loc_date(filepath, input_loc=False)
    else:
        return loc_normal(filepath, input_loc=False)

def loc_normal(filepath, input_loc):
    from philharmonic import conf
    if input_loc:
        rel_folder = conf.rel_input_folder(conf.add_date_to_folders)
    else:
        rel_folder = conf.rel_output_folder(conf.add_date_to_folders)
    path = conf.output_folder + rel_folder
    mkdir_p(path)
    return os.path.join(path, filepath)

def loc_date(filepath, input_loc):
    from philharmonic import conf
    from time import strftime
    from time import mktime
    date = strftime("%Y-%m-%d", conf.current_time)
    time = strftime("%H%M%S", conf.current_time)
    filepath = os.path.basename(filepath)

    if input_loc:
        type = "input"
        rel_folder = conf.rel_input_folder(conf.add_date_to_folders)
    else:
        type = "output"
        rel_folder = conf.rel_output_folder(conf.add_date_to_folders)
    filepath = time + "_" + type + "_" + filepath # save the time as part of the name of the output file
    # filepath = date + "/" + filepath # put the file inside a folder of the current date
    path = conf.output_folder + rel_folder + date + "/"
    mkdir_p(path)
    return os.path.join(path, filepath)

def common_loc(filepath):
    from philharmonic import conf
    new_path = os.path.join(conf.common_output_folder,
                            os.path.dirname(filepath))
    mkdir_p(new_path)
    return os.path.join(conf.common_output_folder, filepath)

class CommonEqualityMixin(object):

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """Override the default hash behavior
        (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))
