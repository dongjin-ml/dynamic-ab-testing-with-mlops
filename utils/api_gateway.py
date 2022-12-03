import time
import boto3
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class api_gateway_handler():
    
    def __init__(self, region_name="ap-northeast-2"):
        
        self.lambda_client = boto3.client('lambda', region_name=region_name)
        self.apig_client = boto3.client('apigateway', region_name=region_name)
    
    def _create_rest_api(self, api_name, api_description, api_base_path, api_stage, account_id, lambda_function_arn):
        
        """
        apigateway_client, lambda_client

        Creates a REST API in Amazon API Gateway. The REST API is backed by the specified
        AWS Lambda function.
        The following is how the function puts the pieces together, in order:
        1. Creates a REST API in Amazon API Gateway.
        2. Creates a '/demoapi' resource in the REST API.
        3. Creates a method that accepts all HTTP actions and passes them through to
           the specified AWS Lambda function.
        4. Deploys the REST API to Amazon API Gateway.
        5. Adds a resource policy to the AWS Lambda function that grants permission
           to let Amazon API Gateway call the AWS Lambda function.
        :param apigateway_client: The Boto3 Amazon API Gateway client object.
        :param api_name: The name of the REST API.
        :param api_base_path: The base path part of the REST API URL.
        :param api_stage: The deployment stage of the REST API.
        :param account_id: The ID of the owning AWS account.
        :param lambda_client: The Boto3 AWS Lambda client object.
        :param lambda_function_arn: The Amazon Resource Name (ARN) of the AWS Lambda
                                    function that is called by Amazon API Gateway to
                                    handle REST requests.
        :return: The ID of the REST API. This ID is required by most Amazon API Gateway
                 methods.
        """
        try:
            response = self.apig_client.create_rest_api(
                name=api_name,
                description=api_description
            )
            api_id = response['id']
            logger.info("Create REST API %s with ID %s.", api_name, api_id)
        except ClientError:
            logger.exception("Couldn't create REST API %s.", api_name)
            raise

        try:
            response = self.apig_client.get_resources(restApiId=api_id)
            root_id = next(item['id'] for item in response['items'] if item['path'] == '/')
            logger.info("Found root resource of the REST API with ID %s.", root_id)
        except ClientError:
            logger.exception("Couldn't get the ID of the root resource of the REST API.")
            raise

        try:
            response = self.apig_client.create_resource(
                restApiId=api_id,
                parentId=root_id,
                pathPart=api_base_path
            )
            base_id = response['id']
            logger.info("Created base path %s with ID %s.", api_base_path, base_id)
        except ClientError:
            logger.exception("Couldn't create a base path for %s.", api_base_path)
            raise

        try:
            self.apig_client.put_method(
                restApiId=api_id,
                resourceId=base_id,
                httpMethod='POST',
                authorizationType='NONE'
            )
            self.apig_client.put_method_response(
                restApiId=api_id,
                resourceId=base_id,
                httpMethod='POST',
                statusCode='200',
                responseModels={'application/json': 'Empty'}
            )
            logger.info("Created a method that accepts all HTTP verbs for the base "
                        "resource.")
        except ClientError:
            logger.exception("Couldn't create a method for the base resource.")
            raise

        lambda_uri = \
            f'arn:aws:apigateway:{self.apig_client.meta.region_name}:' \
            f'lambda:path/2015-03-31/functions/{lambda_function_arn}/invocations'
        try:
            # NOTE: You must specify 'POST' for integrationHttpMethod or this will not work.
            self.apig_client.put_integration(
                restApiId=api_id,
                resourceId=base_id,
                httpMethod='POST',
                type='AWS',
                integrationHttpMethod='POST',
                uri=lambda_uri
            )
            self.apig_client.put_integration_response(
                restApiId=api_id,
                resourceId=base_id,
                httpMethod='POST',
                statusCode='200',
                responseTemplates={'application/json': ''}
            )
            logger.info(
                "Set function %s as integration destination for the base resource.",
                lambda_function_arn)
        except ClientError:
            logger.exception(
                "Couldn't set function %s as integration destination.", lambda_function_arn)
            raise

        try:
            self.apig_client.create_deployment(
                restApiId=api_id,
                stageName=api_stage
            )
            logger.info("Deployed REST API %s.", api_id)
        except ClientError:
            logger.exception("Couldn't deploy REST API %s.", api_id)
            raise

        source_arn = \
            f'arn:aws:execute-api:{self.apig_client.meta.region_name}:' \
            f'{account_id}:{api_id}/*/*/{api_base_path}'
        
        print ("so", source_arn)
        try:
            self.lambda_client.add_permission(
                FunctionName=lambda_function_arn,
                StatementId=f'id-invoke',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=source_arn)
            logger.info("Granted permission to let Amazon API Gateway invoke function %s "
                        "from %s.", lambda_function_arn, source_arn)
        except ClientError:
            logger.exception("Couldn't add permission to let Amazon API Gateway invoke %s.",
                             lambda_function_arn)
            raise

        return api_id
    
    def _construct_api_url(self, api_id, region, api_stage, api_base_path):
        """
        Constructs the URL of the REST API.
        :param api_id: The ID of the REST API.
        :param region: The AWS Region where the REST API was created.
        :param api_stage: The deployment stage of the REST API.
        :param api_base_path: The base path part of the REST API.
        :return: The full URL of the REST API.
        """
        api_url = \
            f'https://{api_id}.execute-api.{region}.amazonaws.com/' \
            f'{api_stage}/{api_base_path}'
        logger.info("Constructed REST API base URL: %s.", api_url)
        return api_url


    def _delete_rest_api(self, apigateway_client, api_id):
        """
        Deletes a REST API and all of its resources from Amazon API Gateway.
        :param apigateway_client: The Boto3 Amazon API Gateway client.
        :param api_id: The ID of the REST API.
        """
        try:
            self.apig_client.delete_rest_api(restApiId=api_id)
            logger.info("Deleted REST API %s.", api_id)
        except ClientError:
            logger.exception("Couldn't delete REST API %s.", api_id)
            raise
            
    def create_rest_api_with_lambda(self, api_name, api_description, api_base_path, \
                                    api_stage, account_id, lambda_function_arn):
        
        print(f"Creating Amazon API Gateway REST API {api_name}...")
    
        api_id = self._create_rest_api(
            api_name,
            api_description,
            api_base_path,
            api_stage,
            account_id,
            lambda_function_arn
        )

        api_url = self._construct_api_url(api_id, self.apig_client.meta.region_name, api_stage, api_base_path)
        print(f"REST API created, URL is :\n\t{api_url}")
        print(f"Sleeping for a couple seconds to give AWS time to prepare...")
        time.sleep(2)
        
        return api_url
        

if __name__ == "__main__":
    
    
    strRegionName = boto3.Session().region_name
    apig = api_gateway_handler(region_name=strRegionName)
    
    account_id = boto3.client('sts').get_caller_identity()['Account']
    api_name = 'DAT-api-gateway'
    api_description='api-gatway for A/B Testing with MAB'
    api_base_path = "dat-api"
    api_stage = 'dev'
    lambda_function_arn="arn:aws:lambda:ap-northeast-2:419974056037:function:DAT-Lambda-MAB"
    
    print(f"Creating Amazon API Gateway REST API {api_name}...")
    
    print ("dsdsd")
    apig.create_rest_api_with_lambda(api_name, api_description, api_base_path, \
                                    api_stage, account_id, lambda_function_arn)
    
    
#     api_id = apig._create_rest_api(
#         api_name,
#         api_description,
#         api_base_path,
#         api_stage,
#         api_httpMethod,
#         account_id,
#         lambda_function_arn
#     )
    
#     api_url = apig._construct_api_url(api_id, strRegionName, api_stage, api_base_path)
#     print(f"REST API created, URL is :\n\t{api_url}")
#     print(f"Sleeping for a couple seconds to give AWS time to prepare...")
#     time.sleep(2)