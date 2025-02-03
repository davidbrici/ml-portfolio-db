# AWS Guide for ML Engineers (Free Tier)

## Introduction to AWS for ML Engineers
AWS (Amazon Web Services) provides cloud-based **compute, storage, and AI/ML services** that allow ML engineers to deploy and scale models efficiently. This guide focuses on **free-tier** AWS services for ML workflows.

---
## 1. Setting Up AWS Free Tier
### Creating an AWS Account
1. Go to [AWS Free Tier](https://aws.amazon.com/free/)
2. Sign up with a valid email and credit card (no charges for free-tier usage)
3. Set up **MFA (Multi-Factor Authentication)** for security

### Installing AWS CLI
```bash
pip install awscli
```
Verify installation:
```bash
aws --version
```

### Configuring AWS CLI
```bash
aws configure
```
Enter:
- **AWS Access Key**
- **AWS Secret Key**
- **Default region** (e.g., `us-east-1`)
- **Output format** (`json` recommended)

---
## 2. Using Amazon S3 (Object Storage)
### Creating an S3 Bucket
```bash
aws s3 mb s3://my-ml-bucket
```

### Uploading a File
```bash
aws s3 cp model.pkl s3://my-ml-bucket/
```

### Listing Bucket Contents
```bash
aws s3 ls s3://my-ml-bucket/
```

---
## 3. Running ML Models on AWS EC2 (Free Tier)
### Launching an EC2 Instance
1. Go to [AWS EC2](https://aws.amazon.com/ec2/)
2. Click **Launch Instance**
3. Choose **t2.micro** (Free Tier eligible)
4. Select **Ubuntu 22.04 LTS** as the OS
5. Configure security group to allow SSH (port 22)
6. Launch and download the private key (`.pem` file)

### Connecting to EC2 via SSH
```bash
ssh -i my-key.pem ubuntu@your-instance-ip
```

### Installing Python & ML Dependencies
```bash
sudo apt update && sudo apt install python3-pip -y
pip install numpy pandas scikit-learn
```

---
## 4. Deploying ML Models with AWS Lambda (Serverless)
AWS Lambda allows deploying **ML inference models** without managing servers.

### Creating a Lambda Function
1. Go to [AWS Lambda](https://aws.amazon.com/lambda/)
2. Click **Create Function** → Select **Author from Scratch**
3. Choose **Python 3.x** as the runtime
4. Set **Memory to 512MB** and **Timeout to 15s**
5. Upload a ZIP file containing your `model.pkl` and `lambda_function.py`

### Example `lambda_function.py`
```python
import boto3
import joblib
import json

def lambda_handler(event, context):
    model = joblib.load("/tmp/model.pkl")
    data = json.loads(event['body'])
    prediction = model.predict([data['features']])
    return {
        "statusCode": 200,
        "body": json.dumps({"prediction": prediction.tolist()})
    }
```

### Deploying the Lambda Function
1. Upload the ZIP file
2. Create an **API Gateway** trigger (to expose the function as an API)
3. Deploy and test with a **POST request**

---
## 5. Storing and Querying ML Data with AWS DynamoDB
### Creating a DynamoDB Table
```bash
aws dynamodb create-table \
    --table-name MLResults \
    --attribute-definitions AttributeName=ModelID,AttributeType=S \
    --key-schema AttributeName=ModelID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST
```

### Inserting ML Predictions
```bash
aws dynamodb put-item --table-name MLResults --item '{"ModelID": {"S": "model_001"}, "Accuracy": {"N": "0.95"}}'
```

### Querying Data
```bash
aws dynamodb scan --table-name MLResults
```

---
## 6. Monitoring ML Pipelines with AWS CloudWatch
### Viewing Logs
```bash
aws logs describe-log-groups
```

### Fetching Lambda Logs
```bash
aws logs tail /aws/lambda/my-ml-function
```

---
## 7. Deploying ML Models with AWS SageMaker (Free Tier)
AWS SageMaker provides an **end-to-end ML workflow**.

### Using a Free Tier SageMaker Notebook
1. Go to [AWS SageMaker](https://aws.amazon.com/sagemaker/)
2. Launch a **t2.medium notebook instance** (free for 250 hours/month)
3. Open Jupyter Notebook and train a model

### Running a Simple ML Model in SageMaker
```python
import boto3
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
sess = sagemaker.Session()

# Train a simple model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])

# Save model
import joblib
joblib.dump(model, "model.pkl")
```

### Deploying the Model as an Endpoint
```python
from sagemaker.sklearn.model import SKLearnModel

sklearn_model = SKLearnModel(model_data="s3://my-ml-bucket/model.pkl", role=role)
predictor = sklearn_model.deploy(instance_type="ml.t2.medium")
```

---
## 8. Automating ML Workflows with AWS Step Functions
AWS Step Functions allow automating ML pipelines.

### Creating a Step Function
1. Go to **Step Functions** in AWS Console
2. Create a new **State Machine**
3. Define states for **data processing, model training, and deployment**

Example JSON Definition:
```json
{
  "Comment": "Simple ML Workflow",
  "StartAt": "Preprocess Data",
  "States": {
    "Preprocess Data": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account-id:function:PreprocessData",
      "Next": "Train Model"
    },
    "Train Model": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account-id:function:TrainModel",
      "End": true
    }
  }
}
```

---
## Conclusion
AWS Free Tier provides **powerful tools for ML engineers** to deploy models without incurring high costs. By leveraging **S3, EC2, Lambda, DynamoDB, SageMaker, and Step Functions**, ML workflows can be **automated, scalable, and production-ready**.

For more advanced topics, check out **AWS Fargate, Kubernetes on AWS (EKS), and MLOps pipelines**!

Happy building! 🚀
