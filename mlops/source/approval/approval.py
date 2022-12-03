import os 
import boto3

class approve_model():
    
    def __init__(self, ):
        
        self.sm_client = boto3.client('sagemaker')
        
    def execution(self, model_package_group_name):
        
        # 위에서 생성한 model_package_group_name 을 인자로 제공 합니다.
        response = self.sm_client.list_model_packages(ModelPackageGroupName=model_package_group_name)
        ModelPackageArn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
        self.sm_client.describe_model_package(ModelPackageName=ModelPackageArn)
        
        model_package_update_input_dict = {"ModelPackageArn" : ModelPackageArn,
                                           "ModelApprovalStatus" : "Approved"}
        model_package_update_response = self.sm_client.update_model_package(**model_package_update_input_dict)
        response = self.sm_client.describe_model_package(ModelPackageName=ModelPackageArn)
        
        image_uri_approved = response["InferenceSpecification"]["Containers"][0]["Image"]
        ModelDataUrl_approved = response["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        
        print("image_uri_approved: ", image_uri_approved)
        print("ModelDataUrl_approved: ", ModelDataUrl_approved)

am = approve_model()

def lambda_handler(event, context):
    """ 
    입력으로 세이지 메이커 모델, 앤드 포인트 컨피그 및 앤드 포인트 이름을 받아서, 앤드포인트를 생성 함.
    """
    
    os.environ['AWS_DEFAULT_REGION'] = event["region"]
    model_package_group_name = event["model_package_group_name"]
    
    am.execution(model_package_group_name)

    return {
        "statusCode": 200,
    }




    
    
    