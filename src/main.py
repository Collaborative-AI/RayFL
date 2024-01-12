import ray

# ray.init(ignore_reinit_error=True, num_cpus=4)
# ray.init(num_cpus=2)




def main():
    # process_control()
    # seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    # for i in range(cfg['num_experiments']):
    #     model_tag_list = [str(seeds[i]), cfg['control_name']]
    #     cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
    #     print('Experiment: {}'.format(cfg['model_tag']))
    #     runExperiment()
    runExperiment()
    return


def runExperiment():
    ray.init()


    @ray.remote
    def f(x):
        return x * x

    futures = [f.remote(i) for i in range(4)]
    print(ray.get(futures))  # [0, 1, 4, 9]
    return


if __name__ == "__main__":
    main()


