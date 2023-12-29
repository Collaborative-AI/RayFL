
1. Ray implements a unified interface that can express both task-parallel and actor-based computations. 

    - Actors enable Ray to efficiently support stateful computations, such as model training, and expose shared mutable state to clients
    - Tasks enable Ray to efficiently and dynamically load balance simulations, process large in- puts and state spaces

2. Stateful and Stateless computation in Ray

    Stateful computation in Ray involves using actors to maintain persistent state across method invocations, while stateless computation uses remote functions for independent, parallelizable tasks with no retained state.

    Stateless Computation in Ray

    Remote Functions:
        Ray remote functions are typically stateless.
        When you decorate a Python function with @ray.remote, it can be executed as a task in the Ray cluster.
        Each invocation of a remote function is independent, and the function does not retain any internal state between calls.
        This statelessness aligns with Ray's goal of enabling easy parallelization of computation.
    Scalability: 
        Stateless remote functions are easy to scale in Ray. You can invoke multiple instances of these functions in parallel without worrying about shared state.
    Use Cases: 
        Ideal for data processing tasks, mathematical computations, and any operation where the output depends solely on the input.

    Stateful Computation in Ray
    Actors:
        Ray supports stateful computation through its actor model.
        By decorating a Python class with @ray.remote, you create a Ray actor.
        Each actor instance is a stateful worker that retains its state across method invocations.
    State Management:
        The state (attributes) of the actor is maintained as long as the actor is alive.
        Actors can be used to maintain stateful services, manage resources, or handle workflows where state persistence is necessary.
    Resource Allocation:
        Ray allows fine-grained control over resources (like CPUs, GPUs) for actors, enabling efficient management of stateful computations.
    Fault Tolerance:
        Ray provides mechanisms for actor reconstruction in case of failures, although the internal state may be lost unless explicitly checkpointed.
    Use Cases:
        Suitable for applications requiring persistent state, such as user sessions, databases, or any long-running service.


3. fault tolerence in ray
    Task Fault Tolerance:
        1. 
        try:
            ray.get(f.remote())
        except ray.exceptions.RayTaskError as e:
                print(e)
        2.specifying the num of tries
        @ray.remote(max_retries=1, retry_exceptions=True)
        3. Specifying -1 allows infinite retries, and 0 disables retries.
        @ray.remote(max_retries=-1, retry_exceptions=True)

    Actor Fault Tolerance:
        in the worker define the checkpoint and restore for fault tolerance
            def checkpoint(self):
                return self.state

            def restore(self, state):
                self.state = state
    
        checkpoint is very useful for long running stateful process/computation


4. schedule in ray
    memory management
    
