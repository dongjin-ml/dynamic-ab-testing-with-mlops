import os 
import ast
import json
import boto3
from os import path
from pprint import pprint
from botocore.exceptions import ClientError

class code_pipeline_handler():
    
    def __init__(self, ):
        
        self.strBasePath = path.dirname(path.abspath(__file__))
        self.pipeline_client = boto3.client('codepipeline')
        
        #with open("code_build_template.json") as json_file: self.dicCodeBuildTemplate = json.load(json_file)
        
    def create_repository(self, strRepoName, strReopDesc):
        
        print ("== CREATE REPO ==")
        strQuery = ''.join(['aws codecommit create-repository',
                            ' --repository-name ', '"', str(strRepoName), '"',
                            ' --repository-description ', '"', str(strReopDesc), '"'])
        strResponse = os.popen(strQuery).read()
        dicResponse = ast.literal_eval(strResponse)
        strCloneURL = dicResponse["repositoryMetadata"]["cloneUrlHttp"]
        
        if strResponse != '':
            print (f"  Repository name [{strRepoName}] was successfully created!!")
            return strCloneURL
        else: return 'Error'
    
    def delete_repository(self, strRepoName):
        
        print ("== DELETE REPO ==")
        strQuery = ''.join(['aws codecommit delete-repository',
                            ' --repository-name ', '"', str(strRepoName), '"'])
        strResponse = os.popen(strQuery).read()
        dicResponse = ast.literal_eval(strResponse)
        
        if strResponse != '': return print (f"  Repository name [{strRepoName}] was successfully deleted!!")
        else: return "Error"
    
    def clone_from_url(self, strDestPath, strCloneURL):
        
        # https://support.huaweicloud.com/intl/en-us/codehub_faq/codehub_faq_1014.html
        os.popen('mkdir tmp/').read()
        strQuery = ''.join(['git clone ', '"', str(strCloneURL), '"', ' tmp/'])
        strResponse = os.popen(strQuery).read()
        print (strResponse)
        
        strQuery = ''.join(['mv tmp/.git ', '"', str(strDestPath), '"'])
        strResponse = os.popen(strQuery).read()
        print (strResponse)
        
        os.popen('rm -rf tmp/').read()
        
    def initial_commit_push(self, strDestPath):
        
        strOrigin = os.getcwd()
        
        print (strOrigin)
        os.chdir(strDestPath)
        print (os.getcwd())
        
        strQuery = ''.join(['git add .'])
        strResponse = os.popen(strQuery).read()
        print (strResponse)
        
        strQuery = ''.join(['git commit -m "Initial commit"'])
        strResponse = os.popen(strQuery).read()
        print (strResponse)
        
        strQuery = ''.join(['git push origin master'])
        strResponse = os.popen(strQuery).read()
        print (strResponse)
        
        os.chdir(strOrigin)
    
    def delete_build_project(self, strCodeBuildPJTName):
        
        strQuery = ''.join(['aws codebuild delete-project --name ', '"', str(strCodeBuildPJTName), '"'])
        strResponse = os.popen(strQuery).read()
        pprint (strResponse)
        
    def create_build_project(self, **kwargs):
        
        strBuildTemplatePath = os.path.join(self.strBasePath, "cicd_templates", "code_build_template.json")
        with open(strBuildTemplatePath) as json_file: dicCodeBuildTemplate = json.load(json_file)
        
        dicCodeBuildTemplate["name"] = kwargs["strCodeBuildPJTName"]
        dicCodeBuildTemplate["artifacts"]["name"] = kwargs["strCodeBuildPJTName"]
        dicCodeBuildTemplate["serviceRole"] = kwargs["strBuildServiceRoleARN"]

        for idx, dicVariable in enumerate(dicCodeBuildTemplate["environment"]["environmentVariables"]):
            if dicVariable["name"] == "AWS_ACCOUNT_ID":
                dicCodeBuildTemplate["environment"]["environmentVariables"][idx]["value"] = kwargs["strAccountId"]
            elif dicVariable["name"] == "AWS_DEFAULT_REGION":
                dicCodeBuildTemplate["environment"]["environmentVariables"][idx]["value"] = kwargs["strRegionName"]
            elif dicVariable["name"] == "TEMPLATE_BUCKET":
                dicCodeBuildTemplate["environment"]["environmentVariables"][idx]["value"] = kwargs["strBucketName"]
        
        jsonCodeBuild = json.dumps(dicCodeBuildTemplate)
        strBuildJsonPath = os.path.join(self.strBasePath, "code_build.json")#'./code_pipeline/code_build.json'
        with open(strBuildJsonPath, "w") as outfile: outfile.write(jsonCodeBuild)
        
        strQuery = ''.join(['aws codebuild create-project --cli-input-json file://', '"', str(strBuildJsonPath), '"'])
        strResponse = os.popen(strQuery).read()
        
        if strResponse == '':
            print ("Project already exists, so update project") 
            strQuery = ''.join(['aws codebuild update-project --cli-input-json file://', '"', str(strBuildJsonPath), '"'])
            strResponse = os.popen(strQuery).read()
        
        print ("Argments for CodeBuild below:")
        pprint (dicCodeBuildTemplate)
        
    def create_execute_code_pipeline(self, **kwargs):
        
        strPipelineTemplatePath = os.path.join(self.strBasePath, "cicd_templates", "code_pipeline_template.json")
        with open(strPipelineTemplatePath) as json_file:
            dicCodePipelineTemplate = json.load(json_file)
            
        dicCodePipelineTemplate["pipeline"]["name"] = kwargs["strCodePipelineName"]
        dicCodePipelineTemplate["pipeline"]["roleArn"] = kwargs["strPipelineRoleARN"]
        dicCodePipelineTemplate["pipeline"]["artifactStore"]["location"] = kwargs["strBucketName"]

        for dicStage in dicCodePipelineTemplate["pipeline"]["stages"]:
            if dicStage["name"] == "Source":
                for dicAction in dicStage["actions"]:
                    dicAction["configuration"]["BranchName"] = "master"
                    dicAction["configuration"]["RepositoryName"] = kwargs["strRepoName"]
                    dicAction["region"] = kwargs["strRegionName"]
            elif dicStage["name"] == "Build":
                for dicAction in dicStage["actions"]:
                    dicAction["configuration"]["ProjectName"] = kwargs["strCodeBuildPJTName"]
                    dicAction["region"] = kwargs["strRegionName"]

        jsonCodePipeline = json.dumps(dicCodePipelineTemplate)
        strPipelineJsonPath = os.path.join(self.strBasePath, "code_pipeline.json")
        with open(strPipelineJsonPath, "w") as outfile: outfile.write(jsonCodePipeline)
        
        strPipeLineName = kwargs["strCodePipelineName"]
        try:
            response = self.pipeline_client.get_pipeline(name=kwargs["strCodePipelineName"],)
            bHasPipeline = True
        except ClientError:
            bHasPipeline = False
            print (f"Couldn't find Pipeline: [{strPipeLineName}], so, create new pipeline.")
    
        #strQuery = ''.join(['aws codepipeline get-pipeline --name ', '"', str(kwargs["strCodePipelineName"]), '"'])
        #strResponse = os.popen(strQuery).read()
        #print ([strResponse])
        #if strResponse == '':
        
        if not bHasPipeline:
            print ("Create CodePipeline") 
            strQuery = ''.join(['aws codepipeline create-pipeline --cli-input-json file://', '"', str(strPipelineJsonPath), '"'])
            strResponse = os.popen(strQuery).read()
        else:
            print ("Pipeline already exists, so update and excute the pipeline") 
            strQuery = ''.join(['aws codepipeline update-pipeline --cli-input-json file://', '"', str(strPipelineJsonPath), '"'])            
            strResponse = os.popen(strQuery).read()
            
            strQuery = ''.join(['aws codepipeline start-pipeline-execution --name ', '"', str(kwargs["strCodePipelineName"]), '"'])            
            strResponse = os.popen(strQuery).read()
            
        print ("Argments for CodeBuild below:")
        pprint (dicCodePipelineTemplate)
        
    def start_pipeline_execution(self, strPipeLineName):
        
        response = self.pipeline_client.start_pipeline_execution(name=strPipeLineName,)
        print (f"Start pipeline execution for [{strPipeLineName}]")

if __name__ == "__main__":
    
    cph = code_pipeline_handler()
    
    strRepoName = "mlops"
    strReopDesc = "MLOps for Dynamic A/B Testing"
    #cph.delete_repository(strRepoName)
    #cph.create_repository(strRepoName, strReopDesc)
    cph.clone_from_url()