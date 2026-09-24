"""
Deploy the trained LightGBM model to an Azure ML managed endpoint.

This module orchestrates model registration and online deployment. Inference
logic and environment dependencies remain in ``score.py`` and ``conda.yaml``.

학습된 LightGBM 모델을 Azure ML 관리형 엔드포인트에 배포합니다.

이 모듈은 모델 등록과 온라인 배포 과정을 조정합니다. 추론 로직과 환경
의존성은 각각 ``score.py``와 ``conda.yaml``에서 관리합니다.
"""

from pathlib import Path
import uuid

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SUBSCRIPTION_ID = "<YOUR_SUBSCRIPTION_ID>"
RESOURCE_GROUP = "<YOUR_RESOURCE_GROUP>"
WORKSPACE_NAME = "<YOUR_WORKSPACE_NAME>"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "outputs" / "ev_lgbm_inference_artifact"
SCORE_PATH = DEPLOYMENT_DIR / "score.py"
CONDA_PATH = DEPLOYMENT_DIR / "conda.yaml"

MODEL_NAME = "ev-lgbm-inference-artifact"
ENVIRONMENT_NAME = "ev-lgbm-inference-env"
ENDPOINT_NAME_PREFIX = "ev-anomaly-endpoint"
DEPLOYMENT_NAME = "blue"

BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
INSTANCE_TYPE = "Standard_DS3_v2"
INSTANCE_COUNT = 1


def get_ml_client() -> MLClient:
    """Connect to the Azure ML workspace using the active Azure identity.
    현재 활성화된 Azure ID를 사용해 Azure ML 작업 영역에 연결합니다.
    """
    return MLClient(
        DefaultAzureCredential(),
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )


def register_model(ml_client: MLClient) -> Model:
    """Register the trained model and its inference metadata as one asset.
    학습된 모델과 추론 메타데이터를 하나의 자산으로 등록합니다.
    """
    model_asset = Model(
        path=str(MODEL_ARTIFACT_PATH),
        name=MODEL_NAME,
        type=AssetTypes.CUSTOM_MODEL,
        description="EV LightGBM model with feature schema and threshold",
        tags={"task": "ev-anomaly-detection"},
    )

    registered_model = ml_client.models.create_or_update(model_asset)
    print(f"Registered model: {registered_model.name}:{registered_model.version}")
    return registered_model


def create_endpoint(ml_client: MLClient) -> ManagedOnlineEndpoint:
    """Create a uniquely named endpoint for real-time inference.
    실시간 추론을 위해 고유한 이름의 엔드포인트를 생성합니다.
    """
    endpoint_name = f"{ENDPOINT_NAME_PREFIX}-{uuid.uuid4().hex[:8]}"
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="EV anomaly real-time inference endpoint",
        auth_mode="key",
    )

    created_endpoint = ml_client.online_endpoints.begin_create_or_update(
        endpoint
    ).result()
    print(f"Created endpoint: {created_endpoint.name}")
    return created_endpoint


def register_environment(ml_client: MLClient) -> Environment:
    """Register the inference environment defined in the existing Conda file.
    기존 Conda 파일에 정의된 추론 환경을 등록합니다.
    """
    environment = Environment(
        name=ENVIRONMENT_NAME,
        description="Inference environment for the EV LightGBM model",
        image=BASE_IMAGE,
        conda_file=str(CONDA_PATH),
    )

    registered_environment = ml_client.environments.create_or_update(environment)
    print(
        "Registered environment: "
        f"{registered_environment.name}:{registered_environment.version}"
    )
    return registered_environment


def create_deployment(
    ml_client: MLClient,
    endpoint: ManagedOnlineEndpoint,
    model: Model,
    environment: Environment,
) -> ManagedOnlineDeployment:
    """Create the managed deployment from the registered serving assets.
    등록된 서빙 자산을 사용해 관리형 배포를 생성합니다.
    """
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=endpoint.name,
        model=f"azureml:{model.name}:{model.version}",
        environment=f"azureml:{environment.name}:{environment.version}",
        code_configuration=CodeConfiguration(
            code=str(DEPLOYMENT_DIR),
            scoring_script=SCORE_PATH.name,
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
    )

    created_deployment = ml_client.online_deployments.begin_create_or_update(
        deployment
    ).result()
    print(f"Created deployment: {created_deployment.name}")
    return created_deployment


def configure_traffic(
    ml_client: MLClient,
    endpoint: ManagedOnlineEndpoint,
    deployment: ManagedOnlineDeployment,
) -> ManagedOnlineEndpoint:
    """Route all endpoint traffic to the newly created deployment.
    모든 엔드포인트 트래픽을 새로 생성한 배포로 전달합니다.
    """
    endpoint.traffic = {deployment.name: 100}
    updated_endpoint = ml_client.online_endpoints.begin_create_or_update(
        endpoint
    ).result()
    print(f"Traffic configured: {deployment.name}=100%")
    return updated_endpoint


def main() -> None:
    ml_client = get_ml_client()
    model = register_model(ml_client)
    endpoint = create_endpoint(ml_client)
    environment = register_environment(ml_client)
    deployment = create_deployment(ml_client, endpoint, model, environment)
    endpoint = configure_traffic(ml_client, endpoint, deployment)

    print(f"Scoring URI: {endpoint.scoring_uri}")


if __name__ == "__main__":
    main()
