import os
import sys
import ray
import json
import tempfile
import shutil


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self):
        self.state = {"num_tasks_executed": 0}

    def execute_task(self, crash=False):
        if crash:
            sys.exit(1)

        # Execute the task
        # ...

        # Update the internal state
        self.state["num_tasks_executed"] = self.state["num_tasks_executed"] + 1

    def checkpoint(self):
        return self.state

    def restore(self, state):
        self.state = state


class Controller:
    def __init__(self):
        self.worker = Worker.remote()
        self.worker_state = ray.get(self.worker.checkpoint.remote())

    def execute_task_with_fault_tolerance(self):
        i = 0
        while True:
            i = i + 1
            try:
                ray.get(self.worker.execute_task.remote(crash=(i % 2 == 1)))
                # Checkpoint the latest worker state
                self.worker_state = ray.get(self.worker.checkpoint.remote())
                return
            except ray.exceptions.RayActorError:
                print("Actor crashes, restarting...")
                # Restart the actor and restore the state
                self.worker = Worker.remote()
                ray.get(self.worker.restore.remote(self.worker_state))


controller = Controller()
controller.execute_task_with_fault_tolerance()
controller.execute_task_with_fault_tolerance()
assert ray.get(controller.worker.checkpoint.remote())["num_tasks_executed"] == 2