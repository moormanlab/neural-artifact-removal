import ray


@ray.remote(num_gpus=1)
def use_gpu():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def ray_init():
    ###local version
    ray.init(num_cpus=8, num_gpus=1, include_dashboard=False)


if __name__ == "__main__":
    ray_init()
    use_gpu.remote()
