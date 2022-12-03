import os
import boto3
import argparse
from datetime import datetime
from ssm import parameter_store
from sagemaker.session import production_variant
from sagemaker import session, get_execution_role

class multi_model_deploy():
    
    def __init__(self, args):
        
        self.args = args
        self.role = self.args.sagemaker_role #self._get_sagemaker_role()
        self.pm = parameter_store(self.args.region)
        self.sm_client = boto3.client('sagemaker')
        self.sm_session = session.Session(boto3.Session())
        
    def _check_multi_models(self, listPackageGropNames):
        
        bModelReadyA, bModelReadyB, bDeployFlag = False, False, False
        
        for model_package_group_name in listPackageGropNames:
            response = self.sm_client.list_model_packages(ModelPackageGroupName=model_package_group_name)
            if model_package_group_name ==  ''.join([self.args.model_group_prefix, 'A']): bModelReadyA = True
            elif model_package_group_name ==  ''.join([self.args.model_group_prefix, 'B']): bModelReadyB = True    
            
        if bModelReadyA and bModelReadyB: bDeployFlag = True
        
        return bDeployFlag
        
        
    def _get_variant(self, model_package_group_name):
    
        print (model_package_group_name)
        response = self.sm_client.list_model_packages(ModelPackageGroupName=model_package_group_name)
        ModelPackageArn = response['ModelPackageSummaryList'][0]['ModelPackageArn'] ## 0: recent version
        response = self.sm_client.describe_model_package(ModelPackageName=ModelPackageArn)

        self.sm_session.create_model(name=model_package_group_name,
                                     role=self.role, 
                                     container_defs={"Image": response['InferenceSpecification']["Containers"][0]["Image"], 
                                                     "ModelDataUrl": response['InferenceSpecification']["Containers"][0]["ModelDataUrl"]})

        variant = production_variant(model_name=model_package_group_name,
                                     instance_type= "ml.c5.4xlarge",
                                     initial_instance_count=1,
                                     variant_name="".join(["variant-", model_package_group_name[-1]]),
                                     initial_weight=1,)

        return variant
    
    def execution(self, ):
        
        listPackageGropNames = [''.join([self.args.model_group_prefix, 'A']), ''.join([self.args.model_group_prefix, 'B'])]
        
        if self._check_multi_models(listPackageGropNames):
            
            listProductionVariants = [self._get_variant(model_package_group_name) for model_package_group_name in listPackageGropNames]
            for variant in listProductionVariants: print (variant)
            
            endpoint_name = f"helpfulness-detector-endpoint-{datetime.now():%Y-%m-%d-%H-%M-%S}"
            self.sm_session.endpoint_from_production_variants(name=endpoint_name, production_variants=listProductionVariants)
            print(f"EndpointName={endpoint_name} completed!!")
            self.pm.put_params(key="DAT-MODEL-ENDPIONT-NAME", value=endpoint_name, overwrite=True)
            
        else:
            print ("Both models are not readly yet!!")
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="docker")
    parser.add_argument("--model_group_prefix", type=str, default="helpfulness-detector-model-")
    parser.add_argument("--region", type=str, default="ap-northeast-2")
    parser.add_argument("--sagemaker_role", type=str, default="sagemaker_role")
    args, _ = parser.parse_known_args()
           
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    mmd = multi_model_deploy(args)
    mmd.execution()
            
