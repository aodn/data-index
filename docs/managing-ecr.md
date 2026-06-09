## Logging in to ECR
Ensure you have logged in to sso as well (`aws sso login`):
```bash
aws ecr get-login-password --region ap-southeast-2 | \
docker login \
--username AWS \
--password-stdin 704910415367.dkr.ecr.ap-southeast-2.amazonaws.com
```