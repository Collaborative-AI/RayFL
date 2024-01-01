# Ray
## idea tests on aws
So far we tested it on aws, following the tutorial on https://saturncloud.io/blog/getting-started-with-ray-clusters/, ray cluster on aws part in Scaling Python with Ray by Holden Karau & Boris Lublinsky https://github.com/scalingpythonml/scaling-python-with-ray/blob/main/appB.asciidoc, and ray cluster documentation https://docs.ray.io/en/latest/cluster/vms/getting-started.html#vm-cluster-quick-start. 

There are two files under fed_ideas_test, fed_block and fed_not_block. There is a simple Model class acts as ray stateful actor, that weights init as [0], and train() that sleeps 20s first then increase weights by 3. We create 40 actors, and train for 1 round. If these 40 actors are not running in parallel, it's going to take 40*20 = 800s, around 13.3 minutes. However, on ray with 3 nodes(workers), each nodes has 2 core cpu and 4g ram, the running time is 36s on both fed_block and fed_not_block. Due to ray autoscaler, ray head might add more nodes(workers) automatically to make running time even less. 

One difference is, the fed_block uses ray.put() that will get all ray object(task) reference result back which is a blocking operation, and ray_not_block uses ray.wait(), which can get the first finished task reference and do some operations on them without waiting those still running tasks. However, since it's a really easy demo, there's no difference. You can learn more the differnece on https://docs.ray.io/en/latest/ray-core/tips-for-first-time.html on ray.wait() part. 


## finished RayFL under src
Run the experiment src/train_model_fl.py, which is based on the [FLPipe](https://github.com/diaoenmao/FLPipe) code. Major changes to adopt FLPipe to Ray were made in module/distrib/controller.py, where the controller manages the `Ray-actor` clients stored in the list `self.worker['client']`. The Client class is decorated with @Ray.remote to make it a Ray actor.

## docker
- Added `dockerfile`
- Changed `src/train_model_fl.py`
- Changed `init_clusters.yaml`

```
FROM rayproject/ray:latest-gpu

# install requirements.txt
COPY RayFL/requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . /RayFL
WORKDIR /RayFL/src
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

# set root user
USER root
```


Set up `aws-cli` locally, open terminal, run `ray up init_clusters.yaml`
After successful launch, run `ray submit init_clusters.yaml src/train_model_fl.py
You can view ray dashboard by `ray dashboard init_clusters.yaml`

Note: now the `init_clusters.yaml` is using [image: "tardism/rayfl"](https://hub.docker.com/repository/docker/tardism/rayfl/general). 
```
# uncomment the following 2 lines to run it locally
import sys
sys.path.insert(0, '/RayFL/src')
```
This is added to `train_model_fl.py` to add `src` to python search path for modules. 
