import ray

def rayInit():
    ###cluster version
    #ray.init(plasma_directory='/home/user/tmp/',temp_dir='/home/user/tmp/',num_gpus=0,include_webui=False)

    ###local version
    ray.init(num_cpus=8,include_webui=False)

if __name__=='__main__':
    rayInit()
