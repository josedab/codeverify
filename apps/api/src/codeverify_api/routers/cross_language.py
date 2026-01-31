"""Cross-Language Verification API endpoints."""

from typing import Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class TypeContractResponse(BaseModel):
    """A type contract."""

    base_type: str
    nullable: bool
    constraints: list[str] = Field(default_factory=list)
    element_type: dict[str, Any] | None = None


class ParameterResponse(BaseModel):
    """A function parameter."""

    name: str
    type: TypeContractResponse


class FunctionContractResponse(BaseModel):
    """A function contract."""

    contract_id: str
    name: str
    description: str
    parameters: list[ParameterResponse]
    return_type: TypeContractResponse | None
    preconditions: list[str]
    postconditions: list[str]
    is_pure: bool


class InterfaceContractResponse(BaseModel):
    """An interface contract."""

    contract_id: str
    name: str
    description: str
    methods: dict[str, FunctionContractResponse]
    extends: list[str]


class InferContractRequest(BaseModel):
    """Request to infer a contract from code."""

    code: str = Field(..., description="The code to analyze")
    language: str = Field(..., description="Programming language: python, typescript")
    symbol_name: str = Field(..., description="Name of the function or class")
    symbol_type: str = Field(
        default="function", description="Type: function or interface"
    )


class InferContractResponse(BaseModel):
    """Response with inferred contract."""

    contract_id: str
    contract_type: str
    contract: dict[str, Any]
    inferred_from: str


class VerifyContractRequest(BaseModel):
    """Request to verify code against a contract."""

    code: str = Field(..., description="The code to verify")
    language: str = Field(..., description="Programming language")
    contract_id: str = Field(..., description="ID of the contract to verify against")
    contract_type: str = Field(default="function", description="Type: function or interface")


class VerifyContractResponse(BaseModel):
    """Response with verification result."""

    contract_id: str
    language: str
    verified: bool
    issues: list[dict[str, Any]]


class GenerateStubRequest(BaseModel):
    """Request to generate a stub in target language."""

    contract_id: str = Field(..., description="ID of the contract")
    target_language: str = Field(..., description="Target language for the stub")


class TypeMappingRequest(BaseModel):
    """Request for type mapping."""

    base_type: str
    source_language: str
    target_language: str


@router.post("/infer", response_model=InferContractResponse)
async def infer_contract(request: InferContractRequest) -> InferContractResponse:
    """
    Infer a language-agnostic contract from code.

    Parses the code and extracts a universal contract that can be
    used to verify implementations in other languages.
    """
    from codeverify_agents import CrossLanguageVerificationBridge, Language

    try:
        language = Language(request.language)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {request.language}. Supported: python, typescript",
        )

    bridge = CrossLanguageVerificationBridge()

    result = await bridge.analyze(request.code, {
        "language": request.language,
        "symbol_name": request.symbol_name,
        "symbol_type": request.symbol_type,
    })

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Contract inference failed: {result.error}",
        )

    if "error" in result.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.data["error"],
        )

    return InferContractResponse(**result.data)


@router.post("/verify", response_model=VerifyContractResponse)
async def verify_against_contract(
    request: VerifyContractRequest,
) -> VerifyContractResponse:
    """
    Verify code implementation against an existing contract.

    Checks that the implementation in the given language matches
    the contract's type signatures and constraints.
    """
    from codeverify_agents import CrossLanguageVerificationBridge, Language

    try:
        language = Language(request.language)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {request.language}",
        )

    bridge = CrossLanguageVerificationBridge()

    result = await bridge.analyze(request.code, {
        "language": request.language,
        "contract_id": request.contract_id,
        "contract_type": request.contract_type,
    })

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {result.error}",
        )

    return VerifyContractResponse(
        contract_id=result.data.get("contract_id", request.contract_id),
        language=request.language,
        verified=result.data.get("verified", False),
        issues=result.data.get("issues", []),
    )


@router.post("/stub")
async def generate_stub(request: GenerateStubRequest) -> dict[str, Any]:
    """
    Generate a stub implementation in the target language.

    Creates a skeleton implementation from a contract that can be
    filled in with actual logic.
    """
    from codeverify_agents import CrossLanguageVerificationBridge, Language

    try:
        language = Language(request.target_language)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported target language: {request.target_language}",
        )

    bridge = CrossLanguageVerificationBridge()

    stub = bridge.generate_stub(request.contract_id, language)

    if stub is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contract not found: {request.contract_id}",
        )

    return {
        "contract_id": request.contract_id,
        "target_language": request.target_language,
        "stub": stub,
    }


@router.post("/type-mapping")
async def get_type_mapping(request: TypeMappingRequest) -> dict[str, Any]:
    """
    Get type mapping between languages.
    """
    from codeverify_agents import CrossLanguageVerificationBridge, Language

    try:
        source = Language(request.source_language)
        target = Language(request.target_language)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {e}",
        )

    bridge = CrossLanguageVerificationBridge()

    target_type = bridge.get_type_mapping(request.base_type, source, target)

    if target_type is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No mapping found for type: {request.base_type}",
        )

    return {
        "base_type": request.base_type,
        "source_language": request.source_language,
        "target_language": request.target_language,
        "mapped_type": target_type,
    }


@router.get("/contracts")
async def list_contracts() -> dict[str, list[str]]:
    """
    List all registered contracts.
    """
    from codeverify_agents import CrossLanguageVerificationBridge

    bridge = CrossLanguageVerificationBridge()
    contracts = bridge.list_contracts()

    return contracts


@router.get("/languages")
async def get_supported_languages() -> dict[str, list[dict[str, str]]]:
    """
    Get supported programming languages.
    """
    from codeverify_agents import Language

    descriptions = {
        "python": "Python 3.8+",
        "typescript": "TypeScript 4.0+",
        "javascript": "JavaScript ES6+",
        "go": "Go 1.18+",
        "rust": "Rust 1.60+",
        "java": "Java 11+",
        "csharp": "C# 10+",
        "cpp": "C++ 17+",
    }

    adapters_available = ["python", "typescript"]

    return {
        "languages": [
            {
                "id": lang.value,
                "name": lang.name.title(),
                "description": descriptions.get(lang.value, ""),
                "adapter_available": lang.value in adapters_available,
            }
            for lang in Language
        ]
    }


@router.get("/type-mappings")
async def get_all_type_mappings() -> dict[str, dict[str, dict[str, str]]]:
    """
    Get all type mappings between languages.
    """
    from codeverify_agents.cross_language_bridge import TYPE_MAPPINGS, Language

    return {
        base_type: {
            lang.value: mapped_type
            for lang, mapped_type in mappings.items()
        }
        for base_type, mappings in TYPE_MAPPINGS.items()
    }
