import threading, time

# Performs 3x as good as multiprocessing.pool.ThreadPool and multiprocessing.Pool
# Bypasses GIL for the most part, limited by memory due to loading images.
class CustomPool():
    def __init__(self, threads=1000):
        self.max_threads = threads
        self.threads = []
        threading.Thread(target=self.activeThreadUpdater, daemon=True).start()

    def activeThreadUpdater(self):
        while True:
            [self.threads.remove(t) for t in list(self.threads) if not t.is_alive()] # Deletes threads that are completed from the local list
            time.sleep(0.001)

    def map(self, target, args):
        while len(self.threads) >= self.max_threads: 
            time.sleep(0.001)
            continue
        self.threads.append(t := threading.Thread(target=target, args=args))
        t.start()

    def is_running(self): return any([t.is_alive() for t in self.threads])