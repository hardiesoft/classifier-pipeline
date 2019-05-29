import logging
import multiprocessing
import os
import sys
import time
import traceback
from memory_profiler import profile
import objgraph
import ml_tools.globals as globs


def process_job(job):
    """ Just a wrapper to pass tupple containing (extractor, *params) to the process_file method. """
    processor = job[0]
    path = job[1]
    params = job[2]

    try:
        processor.process_file(path, **params)
    except Exception:
        logging.exception("Warning - error processing job")

    time.sleep(0.001)  # apparently gives me a chance to catch the control-c


class CPTVFileProcessor:
    """
    Base class for processing a collection of CPTV video files.
    Supports a worker pool to process multiple files at once.
    """

    def __init__(self, config, tracker_config):

        self.config = config
        self.tracker_config = tracker_config

        """
        A base class for processing large sets of CPTV video files.
        """

        # folder to output files to
        self.output_folder = None

        # base folder for source files
        self.source_folder = None

        # number of threads to use when processing jobs.
        self.workers_threads = config.worker_threads

        # optional initializer for worker threads
        self.worker_pool_init = None

        os.makedirs(config.classify.classify_folder, mode=0o775, exist_ok=True)

    def process_file(self, filename, **kwargs):
        """ The function to process an individual file. """
        raise Exception("Process file method must be overwritten in sub class.")

    def needs_processing(self, filename):
        """ Checks if source file needs processing. """
        return True

    def process_folder(self, folder_path, worker_pool_args=None, **kwargs):
        """Processes all files within a folder."""
        if not os.path.exists(folder_path):
            logging.exception(
                "Warning - folder {} does not exist anymore".format(folder_path)
            )
            return
        logging.info("processing %s", folder_path)
        jobs = []
        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            if (
                os.path.isfile(full_path)
                and os.path.splitext(full_path)[1].lower() == ".cptv"
            ):
                if self.needs_processing(full_path):
                    jobs.append((self, full_path, kwargs))

        self.process_job_list(jobs, worker_pool_args)
        
    @profile
    def process_job_list(self, jobs, worker_pool_args=None):
        """
        Processes a list of jobs. Supports worker threads.
        :param jobs: List of jobs to process
        :param worker_pool_args: optional arguments to worker pool initializer
        """

        if self.workers_threads == 0:
            # just process the jobs in the main thread
            for job in jobs:
                process_job(job)
        else:
            # send the jobs to a worker pool
            pool = multiprocessing.Pool(
                self.workers_threads,
                initializer=self.worker_pool_init,
                initargs=worker_pool_args,
            )
            try:
                # see https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
                pool.map(process_job, jobs, chunksize=1)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                pool.terminate()
                exit()
            except Exception:
                logging.exception("Error processing files")
            else:
                pool.close()


    def log_message(self, message):
        """ Record message in stdout.  Will be printed if verbose is enabled. """
        # note, python has really good logging... I should probably make use of this.
        if self.tracker_config.verbose:
            logging.info(message)

    def log_warning(self, message):
        """ Record warning message in stdout."""
        # note, python has really good logging... I should probably make use of this.
        logging.warning("Warning: %s", message)


if __name__ == "__main__":
    # for some reason the fork method seems to memory leak, and unix defaults to this so we
    # stick to spawn.  Also, form might be a problem as some numpy commands have multiple threads?
    multiprocessing.set_start_method("spawn")

def get_size(name, obj, seen=None, depth = 0, parent_name=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(key, value, seen, depth+1, name) for key, value in obj.items()])
        size += sum([get_size(key, key, seen, depth+1, name) for key, value in obj.items()])
    if hasattr(obj, '__dict__'):
        seen.add(id(obj.__dict__))
        size += sum([get_size(key, value, seen, depth+1, name) for key, value in obj.__dict__.items()])
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(name, i, seen, depth+1) for i in obj])

    size_in_mb = size/1000000

    if size_in_mb > 0 and depth <= 2:
        print("size of {} is {}MB parent {}".format(name,size_in_mb,parent_name))
    return size