# Ray
## ideas test on aws
So far we tested it on aws, following the tutorial on https://saturncloud.io/blog/getting-started-with-ray-clusters/, Scaling Python with Ray by Holden Karau & Boris Lublinsky https://github.com/scalingpythonml/scaling-python-with-ray/blob/main/appB.asciidoc, aws setting part, and ray cluster documentation https://docs.ray.io/en/latest/cluster/vms/getting-started.html#vm-cluster-quick-start. 

fed_ideas_test, there are two files, fed_block and fed_not_block. There is a simple model, that weights init as [0], and training is sleep 20s, and then increase weights by 3. We create 40 actors, and train for 1 round, if these 40 actors are not running in parallel, it's going to take 40*20 = 800s, around 13.3 minutes. However, on ray with 3 nodes(workers), each nodes has 2 core cpu and 4g ram, the running time is 36s on both fed_block and fed_not_block. Due to ray autoscaler, ray head might add more nodes(workers) automatically to make running time even less. 

One difference is, the fed_block uses ray.put() that will get all ray object(task) reference result back which is a blocking operation, and ray_not_block uses ray.wait(), which can get the first finished task reference and do some operations on them without waiting those still running tasks. However, since it's a really easy demo, there's no difference. You can learn more the differnece on https://docs.ray.io/en/latest/ray-core/tips-for-first-time.html on ray.wait() part. 




