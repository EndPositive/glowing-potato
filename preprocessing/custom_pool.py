import threading
import time

# Performs 3x as good as multiprocessing.pool.ThreadPool and multiprocessing.Pool
# Bypasses GIL for the most part, limited by memory due to loading images.
class CustomPool:
    def __init__(self, threads=1000):
        self.max_threads = threads
        self.threads = []
        threading.Thread(target=self.active_thread_updater, daemon=True).start()

    def active_thread_updater(self):
        while True:
            dead_threads = (
                thread for thread in list(self.threads) if not thread.is_alive()
            )
            for dead_thread in dead_threads:
                self.threads.remove(dead_thread)
            time.sleep(0.001)

    def map(self, target, args):
        while len(self.threads) >= self.max_threads:
            time.sleep(0.001)
            continue
        self.threads.append(thread := threading.Thread(target=target, args=args))
        thread.start()

    def is_running(self):
        return any(t.is_alive() for t in self.threads)
