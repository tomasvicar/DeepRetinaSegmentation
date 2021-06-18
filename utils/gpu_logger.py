import nvidia_smi


class GpuLogger:
    
    def __init__(self):
        
        try:
            nvidia_smi.nvmlInit()
            self.nvidia_smi = nvidia_smi
            self.measured_gpu_memory = []
        except:
            self.nvidia_smi='error'
            self.measured_gpu_memory = []
            
            
    def save_gpu_memory(self):
        
        self.measured_gpu_memory.append(self.get_gpu_memory(self.nvidia_smi))
        
        
    def get_gpu_memory(self,nvidia_smi):
    
        try:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            
            return info.used/1000000000
        except:
            return 0
    

if __name__ == "__main__":
    
    gpuLogger = GpuLogger()
    
    gpuLogger.save_gpu_memory()
    
    print(gpuLogger.measured_gpu_memory)






# import subprocess as sp
# import os

# def get_gpu_memory():
#     _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    
#     ACCEPTABLE_AVAILABLE_MEMORY = 1024
#     COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
#     memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
#     memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
#     print(memory_free_values)
#     return memory_free_values



# class GpuLogger:
    
#     def __init__(self):
        
#         self.measured_gpu_memory = [0]
            
            
#     def save_gpu_memory(self):
        
#         self.measured_gpu_memory.append(get_gpu_memory)
    
