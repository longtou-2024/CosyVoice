#task.set_accelerator_type("nvidia.com/gpu")
#task.set_accelerator_limit(2)
# set_cpu_request(cpu: str) → PipelineTask # minimum
# set_cpu_limit(cpu: str) → PipelineTask # maximum
# set_memory_request(memory: str) → PipelineTask[source]
# The minimum memory requests required. This string should be a number or a number followed by one of “E”, “Ei”, “P”, “Pi”, “T”, “Ti”, “G”, “Gi”, “M”, “Mi”, “K”, or “Ki”.
# longtou.2024

from google.protobuf import json_format
from kfp import dsl
from kfp import compiler
from kfp.client import Client
from kfp import kubernetes
from kfp.dsl import PipelineTask
from kfp.kubernetes import common

IMAGE_URL = "us-central1-docker.pkg.dev/prod-ai-project/tts/cosyvoice:v7.1"
N_GPU = 4
N_CPU = "40"
MEM_SIZE = "400Gi"
MOUNT_PATH = "/home/longtou.2024/mount"
MODEL_DIR = f"{MOUNT_PATH}/longtou/h100/exp/cosyvoice/20250716"
CONFIG = f"{MODEL_DIR}/cosyvoice2_lt.yaml"
TB_DIR = f"{MODEL_DIR}/tensorboard"
CKPT = f"{MODEL_DIR}/torch_ddp/epoch_0_step_5000.pt"
SHELL_COMMAND = f''' \
export CUDA_VISIBLE_DEVICES="0,1,2,3" \
&& . ../../../activate_python.sh \
&& ./run.sh --stage 1 --stop_stage 1 --model_dir {MODEL_DIR} --tensorboard_dir {TB_DIR} --conf {CONFIG} --from_mount true --from_prod true --train_data "gs://mediazen_teen_laugh mediazen_adult_laugh whispering commbooks_speaking_rate commbooks_tone azure commbooks mediazen" --checkpoint {CKPT}
'''

def add_pod_annotation(
    task: PipelineTask,
    annotation_key: str,
    annotation_value: str,
) -> PipelineTask:
    """Pod metadata 에 annotation 을 추가하는 함수입니다.
    petethegreat(Peter Thompson)이 작성한 다음 PR 의 코드를 그대로 가져왔습니다.
    https://github.com/petethegreat/pipelines/commit/dfa93d5cedf5e558ff905767027f4eaf70652d98
    kfp 2.7.0 에는 아직 해당 기능이 포함되어 있지 않습니다.
    2.7.0 이후 해당 기능이 머지되면, 다음과 같은 방법으로 사용할 수 있을 것입니다.

    from kfp import dsl
    from kfp import kubernetes

    @dsl.component
    def comp():
        pass

    @dsl.pipeline
    def my_pipeline():
        task = comp()
        kubernetes.add_pod_annotation(
            task,
            annotation_key='run_id',
            annotation_value='123456',
        )
    """
    msg = common.get_existing_kubernetes_config_as_message(task)
    msg.pod_metadata.annotations.update({annotation_key: annotation_value})
    task.platform_config["kubernetes"] = json_format.MessageToDict(msg)

    return task

@dsl.container_component
def cosyvoice_component(mount_path: str) -> dsl.ContainerSpec:
    command=["sh", "-c", SHELL_COMMAND, ]

    return dsl.ContainerSpec(image=IMAGE_URL,
                             command=command)
#@dsl.component
#def espnet_component(mount_path: str) -> str:
#    with open(f"./{mount_path}/longtou/log.txt","r") as fin:
#        print(fin.read())
#    return "done"


@dsl.pipeline
def cosyvoice_pipe(
    project: str,
    location: str,
):
    pvc_mount_path = MOUNT_PATH
    task_1 = cosyvoice_component(mount_path=pvc_mount_path)

    task_1.set_accelerator_type("nvidia.com/gpu")
    task_1.set_accelerator_limit(N_GPU)
    task_1.set_cpu_request(N_CPU)
    #task_1.set_cpu_limit(N_CPU)
    task_1.set_memory_request(MEM_SIZE)

    #kubernetes.mount_pvc(
    #    task_1,
    #    pvc_name="shm-memory-disk2",
    #    mount_path='/dev/shm',
    #)
    #######################################################
    # gcsfuse
    #######################################################
    add_pod_annotation(
        task_1,
        annotation_key="gke-gcsfuse/volumes",
        annotation_value="true",
    )

    kubernetes.mount_pvc(
        task_1,
        pvc_name="longtou-gcs-fuse-csi-static-pvc3",
        mount_path=pvc_mount_path,
    )
    kubernetes.empty_dir_mount(
                task_1,
                volume_name="dshm3",
                mount_path="/dev/shm",
                medium="Memory",
                size_limit="300Gi")

compiler.Compiler().compile(cosyvoice_pipe, "cosyvoice_pipe2.yaml")

client = Client(host="https://3313888af2601658-dot-us-central1.pipelines.googleusercontent.com")
run = client.create_run_from_pipeline_package(
        "cosyvoice_pipe2.yaml",
        arguments={
            "project": "prod-ai-project",
            "location": "us-central1",
        },
)
