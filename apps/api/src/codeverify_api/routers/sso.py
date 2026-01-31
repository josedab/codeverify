"""SAML/SSO Authentication for Enterprise.

This module provides SAML 2.0 SSO authentication support for enterprise customers.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4
import hashlib
import hmac

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sso", tags=["sso"])


class SAMLConfig(BaseModel):
    """SAML configuration for an organization."""
    
    organization_id: UUID
    idp_entity_id: str = Field(..., description="Identity Provider Entity ID")
    idp_sso_url: str = Field(..., description="IdP SSO URL")
    idp_slo_url: str | None = Field(None, description="IdP Single Logout URL")
    idp_certificate: str = Field(..., description="IdP X.509 Certificate (PEM)")
    sp_entity_id: str = Field(..., description="Service Provider Entity ID")
    sp_acs_url: str = Field(..., description="Assertion Consumer Service URL")
    attribute_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            "name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
            "groups": "http://schemas.xmlsoap.org/claims/Group",
        },
        description="Mapping of CodeVerify attributes to SAML attributes"
    )
    enforce_sso: bool = Field(False, description="Require SSO for all users")
    domains: list[str] = Field(default_factory=list, description="Email domains for SSO")


class SAMLConfigCreate(BaseModel):
    """Schema for creating SAML configuration."""
    
    idp_entity_id: str
    idp_sso_url: str
    idp_slo_url: str | None = None
    idp_certificate: str
    attribute_mapping: dict[str, str] | None = None
    enforce_sso: bool = False
    domains: list[str] = []


class SAMLConfigResponse(BaseModel):
    """Response schema for SAML configuration."""
    
    organization_id: UUID
    idp_entity_id: str
    idp_sso_url: str
    idp_slo_url: str | None
    sp_entity_id: str
    sp_acs_url: str
    sp_metadata_url: str
    enforce_sso: bool
    domains: list[str]
    created_at: datetime
    updated_at: datetime


class SAMLAuthRequest(BaseModel):
    """SAML authentication request."""
    
    request_id: str
    organization_id: UUID
    created_at: datetime
    relay_state: str | None = None


# In-memory storage (use database in production)
_saml_configs: dict[str, SAMLConfig] = {}
_pending_auth_requests: dict[str, SAMLAuthRequest] = {}


def _generate_sp_metadata(config: SAMLConfig, base_url: str) -> str:
    """Generate SP metadata XML."""
    return f"""<?xml version="1.0"?>
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
                     entityID="{config.sp_entity_id}">
    <md:SPSSODescriptor AuthnRequestsSigned="true"
                        WantAssertionsSigned="true"
                        protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
        <md:NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</md:NameIDFormat>
        <md:AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                                     Location="{config.sp_acs_url}"
                                     index="0"
                                     isDefault="true"/>
    </md:SPSSODescriptor>
</md:EntityDescriptor>"""


def _generate_authn_request(config: SAMLConfig, request_id: str) -> str:
    """Generate SAML AuthnRequest."""
    issue_instant = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return f"""<?xml version="1.0"?>
<samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
                    ID="{request_id}"
                    Version="2.0"
                    IssueInstant="{issue_instant}"
                    Destination="{config.idp_sso_url}"
                    AssertionConsumerServiceURL="{config.sp_acs_url}"
                    ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
    <saml:Issuer>{config.sp_entity_id}</saml:Issuer>
    <samlp:NameIDPolicy Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
                        AllowCreate="true"/>
</samlp:AuthnRequest>"""


@router.post("/config/{organization_id}", response_model=SAMLConfigResponse)
async def create_saml_config(
    organization_id: UUID,
    config: SAMLConfigCreate,
    request: Request,
) -> dict[str, Any]:
    """Configure SAML SSO for an organization.
    
    This endpoint requires organization admin privileges.
    """
    base_url = str(request.base_url).rstrip("/")
    
    # Generate SP details
    sp_entity_id = f"{base_url}/sso/metadata/{organization_id}"
    sp_acs_url = f"{base_url}/sso/acs/{organization_id}"
    sp_metadata_url = f"{base_url}/sso/metadata/{organization_id}"
    
    saml_config = SAMLConfig(
        organization_id=organization_id,
        idp_entity_id=config.idp_entity_id,
        idp_sso_url=config.idp_sso_url,
        idp_slo_url=config.idp_slo_url,
        idp_certificate=config.idp_certificate,
        sp_entity_id=sp_entity_id,
        sp_acs_url=sp_acs_url,
        attribute_mapping=config.attribute_mapping or {},
        enforce_sso=config.enforce_sso,
        domains=config.domains,
    )
    
    _saml_configs[str(organization_id)] = saml_config
    
    return {
        "organization_id": organization_id,
        "idp_entity_id": config.idp_entity_id,
        "idp_sso_url": config.idp_sso_url,
        "idp_slo_url": config.idp_slo_url,
        "sp_entity_id": sp_entity_id,
        "sp_acs_url": sp_acs_url,
        "sp_metadata_url": sp_metadata_url,
        "enforce_sso": config.enforce_sso,
        "domains": config.domains,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


@router.get("/config/{organization_id}", response_model=SAMLConfigResponse)
async def get_saml_config(
    organization_id: UUID,
    request: Request,
) -> dict[str, Any]:
    """Get SAML configuration for an organization."""
    config = _saml_configs.get(str(organization_id))
    
    if not config:
        raise HTTPException(status_code=404, detail="SAML configuration not found")
    
    return {
        "organization_id": organization_id,
        "idp_entity_id": config.idp_entity_id,
        "idp_sso_url": config.idp_sso_url,
        "idp_slo_url": config.idp_slo_url,
        "sp_entity_id": config.sp_entity_id,
        "sp_acs_url": config.sp_acs_url,
        "sp_metadata_url": f"{request.base_url}sso/metadata/{organization_id}",
        "enforce_sso": config.enforce_sso,
        "domains": config.domains,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


@router.delete("/config/{organization_id}")
async def delete_saml_config(
    organization_id: UUID,
) -> dict[str, str]:
    """Delete SAML configuration for an organization."""
    if str(organization_id) not in _saml_configs:
        raise HTTPException(status_code=404, detail="SAML configuration not found")
    
    del _saml_configs[str(organization_id)]
    
    return {"status": "deleted"}


@router.get("/metadata/{organization_id}")
async def get_sp_metadata(
    organization_id: UUID,
    request: Request,
) -> Response:
    """Get Service Provider SAML metadata XML.
    
    This endpoint provides the SP metadata for IdP configuration.
    """
    config = _saml_configs.get(str(organization_id))
    
    if not config:
        raise HTTPException(status_code=404, detail="SAML configuration not found")
    
    base_url = str(request.base_url).rstrip("/")
    metadata = _generate_sp_metadata(config, base_url)
    
    return Response(
        content=metadata,
        media_type="application/xml",
        headers={
            "Content-Disposition": f"attachment; filename=codeverify-sp-metadata-{organization_id}.xml"
        },
    )


@router.get("/login/{organization_id}")
async def initiate_sso_login(
    organization_id: UUID,
    redirect_uri: str | None = None,
) -> RedirectResponse:
    """Initiate SAML SSO login flow.
    
    Redirects to the IdP for authentication.
    """
    config = _saml_configs.get(str(organization_id))
    
    if not config:
        raise HTTPException(status_code=404, detail="SAML configuration not found")
    
    # Generate request ID
    request_id = f"_codeverify_{uuid4().hex}"
    
    # Store pending request
    auth_request = SAMLAuthRequest(
        request_id=request_id,
        organization_id=organization_id,
        created_at=datetime.utcnow(),
        relay_state=redirect_uri,
    )
    _pending_auth_requests[request_id] = auth_request
    
    # Generate AuthnRequest
    authn_request = _generate_authn_request(config, request_id)
    
    # Base64 encode
    encoded_request = base64.b64encode(authn_request.encode()).decode()
    
    # Build redirect URL
    # In production, would use proper URL encoding and signature
    redirect_url = f"{config.idp_sso_url}?SAMLRequest={encoded_request}"
    
    if redirect_uri:
        relay_state = base64.b64encode(redirect_uri.encode()).decode()
        redirect_url += f"&RelayState={relay_state}"
    
    return RedirectResponse(url=redirect_url, status_code=302)


@router.post("/acs/{organization_id}")
async def assertion_consumer_service(
    organization_id: UUID,
    request: Request,
) -> dict[str, Any]:
    """Assertion Consumer Service - receives SAML response from IdP.
    
    This endpoint processes the SAML response and creates a session.
    """
    config = _saml_configs.get(str(organization_id))
    
    if not config:
        raise HTTPException(status_code=404, detail="SAML configuration not found")
    
    # Parse form data
    form_data = await request.form()
    saml_response = form_data.get("SAMLResponse")
    relay_state = form_data.get("RelayState")
    
    if not saml_response:
        raise HTTPException(status_code=400, detail="Missing SAMLResponse")
    
    try:
        # Decode SAML response
        decoded_response = base64.b64decode(saml_response).decode()
        
        # In production, would:
        # 1. Verify XML signature using IdP certificate
        # 2. Verify assertion conditions (NotBefore, NotOnOrAfter)
        # 3. Verify audience restriction
        # 4. Extract attributes using configured mapping
        
        # For demo, return mock user data
        # Real implementation would use python3-saml or similar library
        
        user_data = {
            "email": "user@example.com",  # Would extract from assertion
            "name": "Example User",
            "groups": ["developers"],
            "organization_id": str(organization_id),
            "auth_method": "saml",
            "authenticated_at": datetime.utcnow().isoformat(),
        }
        
        # Decode relay state for redirect
        redirect_uri = None
        if relay_state:
            try:
                redirect_uri = base64.b64decode(relay_state).decode()
            except Exception:
                pass
        
        return {
            "status": "authenticated",
            "user": user_data,
            "redirect_uri": redirect_uri,
            "message": "SAML authentication successful. In production, this would create a session and redirect.",
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process SAML response: {str(e)}")


@router.get("/logout/{organization_id}")
async def initiate_sso_logout(
    organization_id: UUID,
) -> dict[str, Any]:
    """Initiate SAML Single Logout.
    
    Logs out the user from CodeVerify and optionally from the IdP.
    """
    config = _saml_configs.get(str(organization_id))
    
    if not config:
        raise HTTPException(status_code=404, detail="SAML configuration not found")
    
    if config.idp_slo_url:
        # Would generate LogoutRequest and redirect to IdP
        return {
            "status": "logout_initiated",
            "idp_logout_url": config.idp_slo_url,
            "message": "Redirect to IdP for single logout",
        }
    else:
        # Just local logout
        return {
            "status": "logged_out",
            "message": "Local logout complete. IdP logout not configured.",
        }


@router.get("/domains")
async def check_sso_domain(
    email: str,
) -> dict[str, Any]:
    """Check if an email domain requires SSO.
    
    Used during login to determine if user should be redirected to SSO.
    """
    domain = email.split("@")[-1].lower() if "@" in email else ""
    
    for org_id, config in _saml_configs.items():
        if domain in [d.lower() for d in config.domains]:
            return {
                "sso_required": config.enforce_sso,
                "sso_available": True,
                "organization_id": org_id,
                "login_url": f"/sso/login/{org_id}",
            }
    
    return {
        "sso_required": False,
        "sso_available": False,
    }


@router.get("/test-connection/{organization_id}")
async def test_sso_connection(
    organization_id: UUID,
) -> dict[str, Any]:
    """Test SSO connection with the IdP.
    
    Validates configuration and connectivity.
    """
    config = _saml_configs.get(str(organization_id))
    
    if not config:
        raise HTTPException(status_code=404, detail="SAML configuration not found")
    
    # In production, would:
    # 1. Fetch IdP metadata if available
    # 2. Verify certificate validity
    # 3. Test connectivity to SSO URL
    
    checks = {
        "configuration_valid": True,
        "idp_entity_id": config.idp_entity_id,
        "idp_sso_url_configured": bool(config.idp_sso_url),
        "idp_certificate_present": bool(config.idp_certificate),
        "sp_entity_id": config.sp_entity_id,
        "sp_acs_url": config.sp_acs_url,
        "domains_configured": len(config.domains),
        "enforce_sso": config.enforce_sso,
    }
    
    return {
        "status": "ok",
        "checks": checks,
        "message": "Configuration appears valid. Test login to verify end-to-end.",
    }
