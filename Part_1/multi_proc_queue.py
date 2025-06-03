import multiprocessing
from multiprocessing import Pool, Queue, Process
from tqdm import tqdm

SENTINEL_VALUE = None
ERROR_STATUS = "Proc Error: 123047"

def pbar_queue_task(pbar_queue, pbar):
    num_errors = 0
    for item in iter(pbar_queue.get, None):
        process_item, error_status = item
        # Updates as it gets values
        if process_item is SENTINEL_VALUE:
            break

        if error_status == ERROR_STATUS:
            num_errors += 1
        pbar.update()
        pbar.set_description("Processing " + str(item[0]) + " Errors: " + str(num_errors))


class MultiProcQueueProcessing:
    def __init__(self, args_global, task, num_workers):
        # task:
        # f(process_item, *args_global):
        #     if error: return ERROR_STATUS
        #     else: return None or desired output
        # by convention, the process_item is a string or a number. A more complex item can be extracted from args_global
        # execute the useful task to parallelize. Takes in the item to be processed and the constant global arguments.
        # args_global: a tuple of global arguments (garg1, garg2, ...)
        self.args_global = args_global
        self.task = task
        self.num_workers = num_workers

    def _queue_remesh_task(self, queue, proc_num, outputs, args_global, pbar_queue: Queue):
        task_outputs = []
        while True:
            args_item = queue.get()

            if args_item is SENTINEL_VALUE:
                pbar_queue.put(SENTINEL_VALUE)
                outputs[proc_num] = task_outputs
                break

            status = self.task(*((args_item, ) + args_global))
            if status != ERROR_STATUS:
                task_outputs.append(status)

            # This signals the progress bar
            pbar_queue.put((args_item, status))

    def process(self, process_list, timeout=None):
        # We create 2 sets of queues. One is for the mesh processing. The other is a surrogate for the tqdm bar (which would get copied across tasks otherwise)
        pbar = tqdm(process_list, smoothing=0.01)
        # pbar.set_description("Folder " + folder)
        pbar_queue = Queue()

        # set up output
        manager = multiprocessing.Manager()
        outputs = manager.dict()

        # Set up queue for multiprocessing
        q = Queue()
        processes = []
        # Create processes
        for i in range(self.num_workers):
            p = Process(target=self._queue_remesh_task, args=(q, i, outputs, self.args_global, pbar_queue), name=f"Process-{i}")
            processes.append(p)
            p.start()
        p_pbar = Process(target=pbar_queue_task, args=(pbar_queue, pbar), name=f"pbar")
        p_pbar.start()

        # Create queue to parse through
        for i in process_list:
            q.put(i)

        # Sentinel values: Tells workers when task is finished
        for i in range(self.num_workers+2):
            q.put(None)

        for p in processes:
            p.join(timeout=timeout)
        p_pbar.join()

        # print(outputs)
        outputs = [ent for sublist in outputs.values() for ent in sublist]
        return outputs

if __name__=="__main__":
    process_list = [1, 2, 3, 4, 5, 6]

    scale = 12
    center = 1

    num_workers = 3

    def task(x, scale, center):
        # print(x * scale + center)
        return (x * scale + center)

    processor = MultiProcQueueProcessing(args_global=(scale, center), task=task, num_workers=num_workers)
    output = processor.process(process_list)
    print(output)
