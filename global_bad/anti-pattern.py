import ray

ray.init()

global global_var

global_var = 3


@ray.remote
class Actor:
    def update(self, var):
        global global_var
        global_var += var
        print(global_var)
        return global_var


actor = Actor.remote()

# This returns 6, not 7. It is because the value change of global_var
# inside a driver is not reflected to the actor
# because they are running in different processes.
assert ray.get(actor.update.remote(10)) == 13

assert global_var == 3
