import os
import boto3
import argparse
import sagemaker
from os import path
from datetime import datetime
from botocore.config import Config
from utils.ssm import parameter_store
from sagemaker import get_execution_role
from config.config import config_handler
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TuningStep
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor, FrameworkProcessor
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.sklearn.estimator import SKLearn

#import sys
#from os import path
#print(path.dirname( path.dirname( path.abspath(__file__) ) ))
#sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))


class pipeline_tr():
    
    def __init__(self, args):
        
        self.args = args
        self.strRegionName = self.args.config.get_value("COMMON", "region")
        self.pm = parameter_store(self.strRegionName)
        self.account_id = self.pm.get_params(key="DAT-ACCOUNT-ID") #boto3.client("sts").get_caller_identity().get("Account")
        self.default_bucket = self.pm.get_params(key="DAT-BUCKET") # sagemaker.Session().default_bucket()
        self.role = self.pm.get_params(key="DAT-SAGEMAKER-ROLE-ARN")#self.args.config.get_value("ROLE", "sagemaker_execution_role")
        self.strInOutPrefix = '/opt/ml/processing'
        self.sm_client = boto3.client('sagemaker') 
        self.pipeline_session = PipelineSession()
        self.cache_config = CacheConfig(
            enable_caching=self.args.config.get_value("PIPELINE", "enable_caching", dtype="boolean"),
            expire_after=self.args.config.get_value("PIPELINE", "expire_after")
        )    
        self.model_image_uri = sagemaker.image_uris.retrieve("blazingtext", self.strRegionName)
        self.processing_repository_uri = self.pm.get_params(key="DAT-PROCESSING-ECR-URI")# self.args.config.get_value("COMMON", "precessing_image_uri") #self._ecr()
        
        print (f'  Account-ID: {self.account_id}, \nRegion: {self.strRegionName}, \nRole: {self.role}, \nDefault_bucket: {self.default_bucket}')
        print (f'  pipeline_session: {self.pipeline_session}')
        print (f'  processing_repository_uri: {self.processing_repository_uri}')
        
#     def _ecr(self, ):
        
#         ecr_repository = "sagemaker-processing-abtest-container"
#         tag, uri_suffix = ":latest", "amazonaws.com"
#         if self.strRegionName in ["cn-north-1", "cn-northwest-1"]: uri_suffix = "amazonaws.com.cn"
#         self.processing_repository_uri = "{}.dkr.ecr.{}.{}/{}".format(self.account_id, self.strRegionName, uri_suffix, ecr_repository + tag)
        
    def _step_preprocessing(self, ):
        
        ## prepare_dataset.ipynb에서 만들어 놓았음
        input_data = ''.join(["s3://", self.default_bucket, '/reviews-helpfulness-pipeline/data/reviews.tsv.gz'])
        print ("input_data", input_data)
        
        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn, # 안쓰는 방법이 없을까?
            framework_version=None,
            command=["python3"],
            image_uri=self.processing_repository_uri,
            instance_type=self.args.config.get_value("PREPROCESSING", "instance_type"),
            #instance_type="local",
            role=self.role,
            instance_count=self.args.config.get_value("PREPROCESSING", "instance_count", dtype="int"),
            #base_job_name="preprocessing", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
            sagemaker_session=self.pipeline_session
        )
        
        step_args = prep_processor.run(
            job_name="preprocessing", ## 이걸 넣어야 캐시가 작동함, 안그러면 프로세서의 base_job_name 이름뒤에 날짜 시간이 붙어서 캐시 동작 안함
            code='./preprocessing/preprocessing.py', #소스 디렉토리 안에서 파일 path
            source_dir="../source/", #현재 파일에서 소스 디렉토리 상대경로 # add processing.py and requirements.txt here
            inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
            outputs=[ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
                     ProcessingOutput(output_name="validation_data", source="/opt/ml/processing/validation"),
                     ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),],
            arguments=["--mode", "sagemaker-processing", "--input_name", "reviews.tsv.gz", "--region", self.strRegionName],
        )
        
        self.preprocessing_process = ProcessingStep(
            name="PreprocessingProcess", ## Processing job이름
            step_args=step_args,
            cache_config=self.cache_config,
        )
        
        #print (self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri)
        #print (self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri.to_string())
        #print (self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri.expr)
        
        #self.pm.put_params(
        #    key="DAT-TESTDATA-URI",
        #    value=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri.to_string(),
        #    overwrite=True,
        #)
    def _step_training(self, ):
        
        max_jobs = self.args.config.get_value("TRAINING", "max_jobs_param_tuning", dtype="int")
        max_parallel_jobs = self.args.config.get_value("TRAINING", "max_parallel_jobs", dtype="int")
        objective_name = self.args.config.get_value("TRAINING", "objective_name")
        
        self.estimator=Estimator(
            image_uri=self.model_image_uri,
            role=self.role, 
            instance_count=self.args.config.get_value("TRAINING", "instance_count", dtype="int"),
            instance_type=self.args.config.get_value("TRAINING", "instance_type"),
            volume_size=30,
            max_run=360000,
            input_mode= 'File',
            sagemaker_session=self.pipeline_session
        )

        self.estimator.set_hyperparameters(
            mode="supervised",
            epochs=10,
            min_epochs=5, # Min epochs before early stopping is introduced
            early_stopping=True,
            patience=2,
            learning_rate=0.01,
            min_count=2, # words that appear less than min_count are discarded 
            word_ngrams=1, # the number of word n-gram features to use.
            vector_dim=16, # dimensions of embedding layer
        ) 

        if self.args.config.get_value("COMMON", "variant_name") == "B": 
            hyperparameter_ranges={
                'epochs': IntegerParameter(5, 50),
                'learning_rate': ContinuousParameter(0.005, 0.01),
                'min_count': IntegerParameter(0, 100),
                'vector_dim': IntegerParameter(32, 300),
                'word_ngrams': IntegerParameter(1, 3)
            }
        else:
            hyperparameter_ranges={
                'epochs': IntegerParameter(4, 7),
                'learning_rate': ContinuousParameter(0.005, 0.01),
                'min_count': IntegerParameter(0, 5),
                'vector_dim': IntegerParameter(1, 10),
                'word_ngrams': IntegerParameter(1, 2)
            }

        tuner = HyperparameterTuner(
            self.estimator, 
            objective_name,
            hyperparameter_ranges,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
        )
        
        step_tuner_args = tuner.fit(
            inputs={
                "train": TrainingInput(
                    s3_data=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                    content_type="text/csv"
                ),
                "validation": TrainingInput(
                    s3_data=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
                    content_type="text/csv"
                )
            }
        )
        
        self.tuning_process = TuningStep(
            name="TrainWithHyperParamTuningProcess",
            step_args=step_tuner_args,
            cache_config=self.cache_config,
            depends_on=[self.preprocessing_process]
        )
        
    def _step_evaluation(self, ):
        
        eval_processor = FrameworkProcessor(
            estimator_cls=SKLearn, # 안쓰는 방법이 없을까?
            framework_version=None,
            command=["python3"],
            image_uri=self.processing_repository_uri,
            instance_type=self.args.config.get_value("EVALUATION", "instance_type"),
            role=self.role,
            instance_count=self.args.config.get_value("EVALUATION", "instance_count", dtype="int"),
            #base_job_name="evaluation", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
            sagemaker_session=self.pipeline_session
        )
        
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation_" + self.args.config.get_value("COMMON", "variant_name").lower() +  ".json",
        )
        
        step_args = eval_processor.run(
            job_name="evaluation", # Processing job name. If not specified, the processor generates a default job name, based on the base job name and current timestamp.
                                   # 이걸 넣어야 캐시가 작동함, 안그러면 프로세서의 base_job_name 이름뒤에 날짜 시간이 붙어서 캐시 동작 안함
            code='./evaluation/evaluation.py', #소스 디렉토리 안에서 파일 path
            source_dir="../source/", #현재 파일에서 소스 디렉토리 상대경로 # add processing.py and requirements.txt here
            inputs=[ProcessingInput(source=self.tuning_process.get_top_model_s3_uri(top_k=0, s3_bucket=self.default_bucket),
                                    destination="/opt/ml/processing/model"), ## 모델을 
                    ProcessingInput(source=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                                    destination="/opt/ml/processing/test")],
            outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
            arguments=["--mode", "sagemaker-processing", "--s3_model_path", self.tuning_process.get_top_model_s3_uri(top_k=0, s3_bucket=self.default_bucket), \
                       "--variant_name", self.args.config.get_value("COMMON", "variant_name").lower(), "--region", self.strRegionName, \
                       "--sagemaker_role", self.role],
        )
        
        self.evaluation_process = ProcessingStep(
            name="EvaluationProcess", ## Processing job이름
            step_args=step_args,
            cache_config=self.cache_config,
            property_files=[evaluation_report],
            depends_on=[self.preprocessing_process, self.tuning_process]
        )
        
    def _step_register(self, ):
        
        self.model_group_prefix = self.args.config.get_value("MODEL_REGISTER", "model_group_prefix")
        self.model_package_group_name = ''.join([self.model_group_prefix, self.args.config.get_value("COMMON", "variant_name")])
        
        # 위의 step_eval 에서 S3 로 올린 evaluation.json 파일안의 지표를 "모델 레지스트리" 에 모데 버전 등록시에 삽입함
        print (self.evaluation_process.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"])
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri="{}/evaluation_{}.json".format(
                    self.evaluation_process.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
                    self.args.config.get_value("COMMON", "variant_name").lower()
                ),
                content_type="application/json")
        )
        
        model_approval_status = ParameterString(
            name="ModelApprovalStatus",
            default_value=self.args.config.get_value("MODEL_REGISTER", "model_approval_status_default"),
        )  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
                                       
        self.register_process = RegisterModel(
            name="ModelRegisterProcess", ## Processing job이름
            estimator=self.estimator,
            image_uri=self.model_image_uri,
            model_data=self.tuning_process.get_top_model_s3_uri(top_k=0, s3_bucket=self.default_bucket),
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=self.args.config.get_value("MODEL_REGISTER", "inference_instances", dtype="list"),
            transform_instances=self.args.config.get_value("MODEL_REGISTER", "transform_instances", dtype="list"),
            model_package_group_name=self.model_package_group_name,
            #approval_status=model_approval_status, ## Pending, Approved"
            model_metrics=model_metrics,
            depends_on=[self.evaluation_process]
        )
        
    def _step_approve(self, ):
                
        # Lambda helper class can be used to create the Lambda function
        approve_lambda = Lambda(            
            function_name=''.join([self.args.config.get_value("COMMON", "prefix"), "-LambdaApprovalStep"]),
            execution_role_arn=self.pm.get_params(key="DAT-LAMBDA-ROLE-ARN"),#self.args.config.get_value("ROLE", "lambda_role_arn"),
            script="../source/approval/approval.py",
            handler="approval.lambda_handler",
            session=self.pipeline_session
        )
          
        strCurTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.approve_process = LambdaStep(
            name="LambdaModelApprovalProcess",
            description="Lambda for model approval",
            lambda_func=approve_lambda,
            inputs={
                "model_package_group_name": self.model_package_group_name,
                "region": self.strRegionName,
            },
            outputs=[
                LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String),
                #LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String),
            ],
            cache_config=self.cache_config,
            depends_on=[self.register_process]
        )
        
    def _step_deploy(self, ):
        
        deploy_processor = FrameworkProcessor(
            estimator_cls=SKLearn, # 안쓰는 방법이 없을까?
            framework_version=None,
            command=["python3"],
            image_uri=self.processing_repository_uri,
            instance_type=self.args.config.get_value("DEPLOY", "instance_type"),            
            instance_count=self.args.config.get_value("DEPLOY", "instance_count", dtype="int"),
            role=self.role,
            #base_job_name="preprocessing", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
            sagemaker_session=self.pipeline_session
        )
        
        step_args = deploy_processor.run(
            job_name="deploy", ## 이걸 넣어야 캐시가 작동함, 안그러면 프로세서의 base_job_name 이름뒤에 날짜 시간이 붙어서 캐시 동작 안함
            code='./deploy/deploy.py', #소스 디렉토리 안에서 파일 path
            source_dir="../source/", #현재 파일에서 소스 디렉토리 상대경로 # add processing.py and requirements.txt here
            arguments=["--mode", "sagemaker-processing", "--model_group_prefix", self.model_group_prefix, "--region", self.strRegionName, "--sagemaker_role", self.role],
        )
        cache_config = CacheConfig(enable_caching=False, expire_after=self.args.config.get_value("PIPELINE", "expire_after"))
        self.deploy_process = ProcessingStep(
            name="DeployProcess", ## Processing job이름
            step_args=step_args,
            cache_config=cache_config,
            depends_on=[self.approve_process]
        )

    def _get_pipeline(self, ):
        
        pipeline_prefix = 'pipeline-train-model-'
        pipeline_name = ''.join([pipeline_prefix, self.args.config.get_value("COMMON", "variant_name")])
        pipeline = Pipeline(name=pipeline_name,
                           steps=[self.preprocessing_process, self.tuning_process, self.evaluation_process, \
                                  self.register_process, self.approve_process, self.deploy_process],)

        return pipeline
                            
    def execution(self, ):
        
        self._step_preprocessing()
        self._step_training()
        self._step_evaluation()
        self._step_register()
        self._step_approve()
        self._step_deploy()
        
        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.role) ## Submit the pipeline definition to the SageMaker Pipelines service 
        execution = pipeline.start()
        execution.describe()
        if self.args.mode == "docker": execution.wait()

if __name__ == "__main__":
    
    
    strBasePath, strCurrentDir = path.dirname(path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    print ("==================")
    print (f"  Working Dir: {os.getcwd()}")
    print (f"  You should execute 'mlops_pipeline.py' in 'pipeline' directory'") 
    print ("==================")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="docker")
    args, _ = parser.parse_known_args()
    args.config = config_handler()
    
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.config.get_value("COMMON", "region")
    
    pipe_tr = pipeline_tr(args)
    pipe_tr.execution()
    
    os.chdir(strCurrentDir)
    print (f"  Working Dir: {os.getcwd()}")