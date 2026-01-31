"""Analysis repository."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from codeverify_api.db.models import Analysis, Finding, AnalysisStage, Repository
from codeverify_api.db.repository import BaseRepository


class AnalysisRepository(BaseRepository[Analysis]):
    """Repository for Analysis entities."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Analysis)

    async def get_with_findings(self, id: UUID) -> Analysis | None:
        """Get analysis with findings loaded."""
        query = (
            select(Analysis)
            .where(Analysis.id == id)
            .options(
                selectinload(Analysis.findings),
                selectinload(Analysis.stages),
                selectinload(Analysis.repository),
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_pr(
        self,
        repo_id: UUID,
        pr_number: int,
        limit: int = 10,
    ) -> list[Analysis]:
        """Get analyses for a specific PR."""
        query = (
            select(Analysis)
            .where(
                and_(
                    Analysis.repo_id == repo_id,
                    Analysis.pr_number == pr_number,
                )
            )
            .order_by(desc(Analysis.created_at))
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_latest_for_pr(
        self,
        repo_id: UUID,
        pr_number: int,
    ) -> Analysis | None:
        """Get the latest analysis for a PR."""
        query = (
            select(Analysis)
            .where(
                and_(
                    Analysis.repo_id == repo_id,
                    Analysis.pr_number == pr_number,
                )
            )
            .order_by(desc(Analysis.created_at))
            .limit(1)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def create_analysis(
        self,
        repo_id: UUID,
        pr_number: int,
        pr_title: str | None,
        head_sha: str,
        base_sha: str | None,
        triggered_by: UUID | None = None,
    ) -> Analysis:
        """Create a new analysis."""
        return await self.create(
            repo_id=repo_id,
            pr_number=pr_number,
            pr_title=pr_title,
            head_sha=head_sha,
            base_sha=base_sha,
            triggered_by=triggered_by,
            status="pending",
        )

    async def start_analysis(self, id: UUID) -> Analysis | None:
        """Mark analysis as started."""
        return await self.update(
            id,
            status="running",
            started_at=datetime.utcnow(),
        )

    async def complete_analysis(
        self,
        id: UUID,
        status: str = "completed",
        error_message: str | None = None,
    ) -> Analysis | None:
        """Mark analysis as completed."""
        return await self.update(
            id,
            status=status,
            completed_at=datetime.utcnow(),
            error_message=error_message,
        )

    async def add_finding(
        self,
        analysis_id: UUID,
        category: str,
        severity: str,
        title: str,
        description: str | None,
        file_path: str,
        line_start: int | None = None,
        line_end: int | None = None,
        code_snippet: str | None = None,
        fix_suggestion: str | None = None,
        fix_diff: str | None = None,
        confidence: float | None = None,
        verification_type: str | None = None,
        verification_proof: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Finding:
        """Add a finding to an analysis."""
        finding = Finding(
            analysis_id=analysis_id,
            category=category,
            severity=severity,
            title=title,
            description=description,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            code_snippet=code_snippet,
            fix_suggestion=fix_suggestion,
            fix_diff=fix_diff,
            confidence=confidence,
            verification_type=verification_type,
            verification_proof=verification_proof,
            metadata=metadata or {},
        )
        self.session.add(finding)
        await self.session.flush()
        await self.session.refresh(finding)
        return finding

    async def add_stage(
        self,
        analysis_id: UUID,
        stage_name: str,
        status: str = "pending",
    ) -> AnalysisStage:
        """Add a stage to an analysis."""
        stage = AnalysisStage(
            analysis_id=analysis_id,
            stage_name=stage_name,
            status=status,
        )
        self.session.add(stage)
        await self.session.flush()
        await self.session.refresh(stage)
        return stage

    async def update_stage(
        self,
        stage_id: UUID,
        status: str,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        duration_ms: int | None = None,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> AnalysisStage | None:
        """Update a stage."""
        stage = await self.session.get(AnalysisStage, stage_id)
        if stage is None:
            return None

        stage.status = status
        if started_at:
            stage.started_at = started_at
        if completed_at:
            stage.completed_at = completed_at
        if duration_ms is not None:
            stage.duration_ms = duration_ms
        if result is not None:
            stage.result = result
        if error_message is not None:
            stage.error_message = error_message

        await self.session.flush()
        await self.session.refresh(stage)
        return stage

    async def get_findings(
        self,
        analysis_id: UUID,
        category: str | None = None,
        severity: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Finding]:
        """Get findings for an analysis."""
        query = select(Finding).where(Finding.analysis_id == analysis_id)

        if category:
            query = query.where(Finding.category == category)
        if severity:
            query = query.where(Finding.severity == severity)

        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def dismiss_finding(
        self,
        finding_id: UUID,
        dismissed_by: UUID,
        reason: str | None = None,
    ) -> Finding | None:
        """Dismiss a finding."""
        finding = await self.session.get(Finding, finding_id)
        if finding is None:
            return None

        finding.dismissed = True
        finding.dismissed_by = dismissed_by
        finding.dismissed_reason = reason

        await self.session.flush()
        await self.session.refresh(finding)
        return finding


class RepositoryRepository(BaseRepository[Repository]):
    """Repository for Repository entities."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Repository)

    async def get_by_github_id(self, github_id: int) -> Repository | None:
        """Get repository by GitHub ID."""
        query = select(Repository).where(Repository.github_id == github_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_full_name(self, full_name: str) -> Repository | None:
        """Get repository by full name (owner/repo)."""
        query = select(Repository).where(Repository.full_name == full_name)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_or_create(
        self,
        github_id: int,
        name: str,
        full_name: str,
        private: bool = False,
        org_id: UUID | None = None,
    ) -> Repository:
        """Get existing repository or create new one."""
        repo = await self.get_by_github_id(github_id)
        if repo:
            return repo

        return await self.create(
            github_id=github_id,
            name=name,
            full_name=full_name,
            private=private,
            org_id=org_id,
        )
