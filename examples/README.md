# CodeVerify Examples

Runnable examples to get started quickly.

## Prerequisites

```bash
pip install -e packages/core
```

## Quick Scripts

| Script | Description | Run |
|--------|-------------|-----|
| `quickstart.py` | Verify a code snippet with Z3 in 15 lines | `python examples/quickstart.py` |
| `custom_rules.py` | Create and apply custom lint rules | `python examples/custom_rules.py` |
| `analyze_file.py` | Analyze a local file and print findings | `python examples/analyze_file.py <file>` |

## Example Projects

Full example projects with `.codeverify.yml` configs demonstrating different stacks:

| Project | Stack | Issues CodeVerify Finds |
|---------|-------|-----------------------|
| [python-fastapi/](python-fastapi/) | Python + FastAPI | Null safety, SQL injection, eval(), division by zero |
| [typescript-nextjs/](typescript-nextjs/) | TypeScript + Next.js | Type safety (`any`), XSS, eval() |
| [go-service/](go-service/) | Go microservice | Nil pointer, SQL injection, division by zero |
| [terraform-infra/](terraform-infra/) | Terraform + AWS | Open CIDR 0.0.0.0/0, unencrypted RDS, public S3 |
| [monorepo/](monorepo/) | Python + TypeScript | Cross-language contract mismatch, eval(), division by zero |

### Try an example with the verification protocol

```bash
python -c "
from codeverify_core.verification_protocol import VerificationProtocolServer, VerifyRequest
server = VerificationProtocolServer()
with open('examples/python-fastapi/app.py') as f:
    code = f.read()
result = server.verify(VerifyRequest(files=[{'path': 'app.py', 'content': code}]))
print(f'Status: {result.status.value}')
for f in result.findings:
    print(f'  Line {f.line}: [{f.severity}] {f.message}')
"
```
