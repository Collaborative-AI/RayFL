import ray

ray.init()


@ray.remote
class GlobalVarActor:
    def __init__(self):
        self.global_var = 3

    def set_global_var(self, var):
        self.global_var = var
        
    def update_global_var(self, var):
        self.global_var += var

    def get_global_var(self):
        return self.global_var


@ray.remote
class Actor:
    def __init__(self, global_var_actor):
        self.global_var_actor = global_var_actor

    def f(self):
        return ray.get(self.global_var_actor.get_global_var.remote()) + 3


global_var_actor = GlobalVarActor.remote()
actor = Actor.remote(global_var_actor)
ray.get(global_var_actor.set_global_var.remote(4))
# This returns 7 correctly.
assert ray.get(actor.f.remote()) == 7