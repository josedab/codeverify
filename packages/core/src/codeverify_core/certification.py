"""Developer Certification Program.

Manages the CodeVerify certification program with courses, modules,
hands-on labs, assessments, and digital credential issuance.

Features:
- Multi-module course structure with progressive difficulty
- Hands-on lab environments with real verification exercises
- Assessment engine with scoring and passing thresholds
- Digital badge and certificate generation
- Certification directory and verification
- Progress tracking and completion analytics
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CertificationLevel(str, Enum):
    """Certification levels."""

    FOUNDATIONS = "foundations"
    PRACTITIONER = "practitioner"
    EXPERT = "expert"


class ModuleStatus(str, Enum):
    """Status of a course module for a learner."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class LabStatus(str, Enum):
    """Status of a hands-on lab."""

    AVAILABLE = "available"
    RUNNING = "running"
    COMPLETED = "completed"
    EXPIRED = "expired"


class BadgeType(str, Enum):
    """Types of digital badges."""

    COMPLETION = "completion"
    EXCELLENCE = "excellence"
    COMMUNITY = "community"
    SPECIALIZATION = "specialization"


class AssessmentType(str, Enum):
    """Types of assessment questions."""

    MULTIPLE_CHOICE = "multiple_choice"
    CODE_ANALYSIS = "code_analysis"
    HANDS_ON = "hands_on"
    SHORT_ANSWER = "short_answer"


@dataclass
class CourseModule:
    """A module within the certification course."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    order: int = 0
    level: CertificationLevel = CertificationLevel.FOUNDATIONS
    duration_minutes: int = 30
    topics: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    learning_objectives: list[str] = field(default_factory=list)
    content_markdown: str = ""
    has_lab: bool = False
    has_assessment: bool = True


@dataclass
class LabExercise:
    """A hands-on lab exercise."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    module_id: str = ""
    title: str = ""
    description: str = ""
    instructions: str = ""
    starter_code: str = ""
    expected_output: str = ""
    verification_checks: list[str] = field(default_factory=list)
    time_limit_minutes: int = 30
    difficulty: str = "medium"


@dataclass
class AssessmentQuestion:
    """A question in the certification assessment."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    module_id: str = ""
    question_type: AssessmentType = AssessmentType.MULTIPLE_CHOICE
    question_text: str = ""
    options: list[str] = field(default_factory=list)
    correct_answer: str = ""
    explanation: str = ""
    points: int = 10
    code_snippet: str = ""


@dataclass
class Assessment:
    """A certification assessment (exam)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: CertificationLevel = CertificationLevel.FOUNDATIONS
    title: str = ""
    questions: list[AssessmentQuestion] = field(default_factory=list)
    passing_score: float = 0.7
    time_limit_minutes: int = 60
    max_attempts: int = 3

    @property
    def total_points(self) -> int:
        return sum(q.points for q in self.questions)


@dataclass
class AssessmentSubmission:
    """A learner's assessment submission."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    assessment_id: str = ""
    learner_id: str = ""
    answers: dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    points_earned: int = 0
    total_points: int = 0
    passed: bool = False
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    graded_at: datetime | None = None
    attempt_number: int = 1


@dataclass
class DigitalBadge:
    """A digital credential/badge."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    badge_type: BadgeType = BadgeType.COMPLETION
    level: CertificationLevel = CertificationLevel.FOUNDATIONS
    title: str = ""
    description: str = ""
    image_url: str = ""
    criteria: str = ""
    issued_to: str = ""
    issued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    verification_url: str = ""
    credential_hash: str = ""

    def verify(self) -> bool:
        """Verify badge authenticity."""
        expected = hashlib.sha256(
            f"{self.id}:{self.issued_to}:{self.issued_at.isoformat()}".encode()
        ).hexdigest()[:16]
        return self.credential_hash == expected


@dataclass
class Certificate:
    """A certification certificate."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: CertificationLevel = CertificationLevel.FOUNDATIONS
    holder_name: str = ""
    holder_email: str = ""
    issued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(days=730))
    certificate_number: str = ""
    verification_hash: str = ""
    badges: list[DigitalBadge] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return datetime.now(UTC) < self.expires_at

    @property
    def verification_url(self) -> str:
        return f"https://codeverify.dev/verify/{self.certificate_number}"


@dataclass
class LearnerProgress:
    """Track a learner's progress through the program."""

    learner_id: str = ""
    learner_name: str = ""
    learner_email: str = ""
    enrolled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    module_status: dict[str, ModuleStatus] = field(default_factory=dict)
    lab_completions: list[str] = field(default_factory=list)
    assessment_submissions: list[AssessmentSubmission] = field(default_factory=list)
    certificates: list[Certificate] = field(default_factory=list)
    badges: list[DigitalBadge] = field(default_factory=list)
    total_time_minutes: int = 0

    @property
    def modules_completed(self) -> int:
        return sum(1 for s in self.module_status.values() if s == ModuleStatus.COMPLETED)

    @property
    def completion_percentage(self) -> float:
        if not self.module_status:
            return 0.0
        return (self.modules_completed / len(self.module_status)) * 100


class CourseBuilder:
    """Builds the certification course curriculum."""

    def build_foundations_course(self) -> list[CourseModule]:
        """Build the foundations certification course."""
        return [
            CourseModule(
                title="Introduction to Formal Verification",
                description="Learn the fundamentals of formal verification and why it matters for modern software",
                order=1,
                level=CertificationLevel.FOUNDATIONS,
                duration_minutes=30,
                topics=[
                    "What is formal verification",
                    "SMT solvers",
                    "Proofs vs testing",
                    "CodeVerify overview",
                ],
                learning_objectives=[
                    "Explain the difference between testing and formal verification",
                    "Describe how Z3 SMT solver works at a high level",
                    "Identify scenarios where formal verification adds value",
                ],
                has_lab=False,
            ),
            CourseModule(
                title="Z3 Fundamentals",
                description="Hands-on introduction to Z3 constraints, assertions, and satisfiability checking",
                order=2,
                level=CertificationLevel.FOUNDATIONS,
                duration_minutes=45,
                topics=["Z3 basics", "Constraints", "Satisfiability", "Counterexamples"],
                learning_objectives=[
                    "Write basic Z3 constraints",
                    "Interpret satisfiability results",
                    "Read and understand counterexamples",
                ],
                has_lab=True,
            ),
            CourseModule(
                title="CodeVerify Workflows",
                description="Setting up and using CodeVerify for PR reviews, CLI scanning, and CI/CD integration",
                order=3,
                level=CertificationLevel.FOUNDATIONS,
                duration_minutes=40,
                topics=["Configuration", "CLI usage", "PR integration", "Dashboard"],
                learning_objectives=[
                    "Configure .codeverify.yml for a project",
                    "Run CodeVerify scans from the command line",
                    "Interpret findings in PR comments",
                ],
                has_lab=True,
            ),
            CourseModule(
                title="Custom Rules",
                description="Creating custom verification rules using pattern, AST, and semantic rule types",
                order=4,
                level=CertificationLevel.FOUNDATIONS,
                duration_minutes=35,
                topics=["Rule types", "Pattern rules", "AST rules", "Rule composition"],
                learning_objectives=[
                    "Create pattern-based custom rules",
                    "Create AST-based rules for structural checks",
                    "Combine rules with AND/OR logic",
                ],
                has_lab=True,
            ),
            CourseModule(
                title="CI/CD Integration",
                description="Integrating CodeVerify into CI/CD pipelines with quality gates",
                order=5,
                level=CertificationLevel.FOUNDATIONS,
                duration_minutes=30,
                topics=["GitHub Actions", "Quality gates", "SARIF output", "Notifications"],
                learning_objectives=[
                    "Set up CodeVerify GitHub Action",
                    "Configure quality gate thresholds",
                    "Generate and consume SARIF reports",
                ],
                has_lab=True,
            ),
        ]

    def build_labs(self, modules: list[CourseModule]) -> list[LabExercise]:
        """Build lab exercises for modules that require them."""
        labs = []
        for module in modules:
            if module.has_lab:
                lab = LabExercise(
                    module_id=module.id,
                    title=f"Lab: {module.title}",
                    description=f"Hands-on exercise for {module.title}",
                    instructions=f"Complete the exercises for {module.title}",
                    time_limit_minutes=30,
                )
                labs.append(lab)
        return labs

    def build_assessment(
        self, level: CertificationLevel = CertificationLevel.FOUNDATIONS
    ) -> Assessment:
        """Build a certification assessment."""
        questions = [
            AssessmentQuestion(
                question_type=AssessmentType.MULTIPLE_CHOICE,
                question_text="What does 'unsat' mean in a Z3 solver result?",
                options=[
                    "The constraints are satisfiable",
                    "The constraints have no solution (property holds)",
                    "The solver timed out",
                    "The input is invalid",
                ],
                correct_answer="The constraints have no solution (property holds)",
                explanation="'unsat' means unsatisfiable — the negation of the property has no solution, proving the property holds.",
                points=10,
            ),
            AssessmentQuestion(
                question_type=AssessmentType.MULTIPLE_CHOICE,
                question_text="Which verification check detects potential None.attribute errors?",
                options=["array_bounds", "null_safety", "integer_overflow", "division_by_zero"],
                correct_answer="null_safety",
                explanation="null_safety checks detect potential None/null dereference issues.",
                points=10,
            ),
            AssessmentQuestion(
                question_type=AssessmentType.CODE_ANALYSIS,
                question_text="What memory safety issue does this code have?",
                code_snippet="def process(data):\n    result = data.get('key')\n    return result.strip()",
                options=[
                    "Buffer overflow",
                    "None dereference (dict.get may return None)",
                    "Integer overflow",
                    "No issues",
                ],
                correct_answer="None dereference (dict.get may return None)",
                explanation="dict.get() returns None if key is missing, and calling .strip() on None raises AttributeError.",
                points=15,
            ),
            AssessmentQuestion(
                question_type=AssessmentType.MULTIPLE_CHOICE,
                question_text="What file configures CodeVerify for a repository?",
                options=["codeverify.json", ".codeverify.yml", "verify.config.ts", ".cv-config"],
                correct_answer=".codeverify.yml",
                explanation="CodeVerify uses .codeverify.yml for per-repository configuration.",
                points=10,
            ),
            AssessmentQuestion(
                question_type=AssessmentType.MULTIPLE_CHOICE,
                question_text="What output format is used for static analysis tool integration?",
                options=["CSV", "SARIF", "Protobuf", "YAML"],
                correct_answer="SARIF",
                explanation="SARIF (Static Analysis Results Interchange Format) is the standard for static analysis results.",
                points=10,
            ),
        ]

        return Assessment(
            level=level,
            title=f"CodeVerify Certified Developer — {level.value.title()}",
            questions=questions,
            passing_score=0.7,
            time_limit_minutes=45,
        )


class AssessmentGrader:
    """Grades certification assessments."""

    def grade(self, assessment: Assessment, answers: dict[str, str]) -> AssessmentSubmission:
        """Grade an assessment submission."""
        points_earned = 0
        total_points = assessment.total_points

        for question in assessment.questions:
            student_answer = answers.get(question.id, "")
            if student_answer == question.correct_answer:
                points_earned += question.points

        score = points_earned / total_points if total_points > 0 else 0.0
        passed = score >= assessment.passing_score

        return AssessmentSubmission(
            assessment_id=assessment.id,
            answers=answers,
            score=score,
            points_earned=points_earned,
            total_points=total_points,
            passed=passed,
            graded_at=datetime.now(UTC),
        )


class CredentialIssuer:
    """Issues digital badges and certificates."""

    def issue_badge(
        self,
        learner_id: str,
        badge_type: BadgeType,
        level: CertificationLevel,
        title: str,
    ) -> DigitalBadge:
        """Issue a digital badge."""
        badge = DigitalBadge(
            badge_type=badge_type,
            level=level,
            title=title,
            description=f"Awarded for completing {title}",
            issued_to=learner_id,
        )
        badge.credential_hash = hashlib.sha256(
            f"{badge.id}:{badge.issued_to}:{badge.issued_at.isoformat()}".encode()
        ).hexdigest()[:16]
        badge.verification_url = f"https://codeverify.dev/badges/{badge.id}"
        return badge

    def issue_certificate(
        self,
        learner_name: str,
        learner_email: str,
        level: CertificationLevel,
        badges: list[DigitalBadge] | None = None,
    ) -> Certificate:
        """Issue a certification certificate."""
        cert_number = f"CV-{level.value[:3].upper()}-{uuid.uuid4().hex[:8].upper()}"

        cert = Certificate(
            level=level,
            holder_name=learner_name,
            holder_email=learner_email,
            certificate_number=cert_number,
            badges=badges or [],
        )
        cert.verification_hash = hashlib.sha256(
            f"{cert.id}:{cert.holder_email}:{cert.certificate_number}".encode()
        ).hexdigest()[:16]

        return cert


class CertificationProgram:
    """Main certification program manager.

    Manages courses, labs, assessments, and credential issuance.
    """

    def __init__(self) -> None:
        self.course_builder = CourseBuilder()
        self.grader = AssessmentGrader()
        self.issuer = CredentialIssuer()
        self.modules: list[CourseModule] = []
        self.labs: list[LabExercise] = []
        self.assessments: dict[str, Assessment] = {}
        self.learners: dict[str, LearnerProgress] = {}
        self.certificates: dict[str, Certificate] = {}

        self._initialize_curriculum()

    def _initialize_curriculum(self) -> None:
        """Initialize the foundations curriculum."""
        self.modules = self.course_builder.build_foundations_course()
        self.labs = self.course_builder.build_labs(self.modules)
        foundations_exam = self.course_builder.build_assessment(CertificationLevel.FOUNDATIONS)
        self.assessments[foundations_exam.id] = foundations_exam

    def enroll_learner(self, name: str, email: str) -> LearnerProgress:
        """Enroll a new learner."""
        learner_id = str(uuid.uuid4())[:8]
        progress = LearnerProgress(
            learner_id=learner_id,
            learner_name=name,
            learner_email=email,
            module_status={m.id: ModuleStatus.NOT_STARTED for m in self.modules},
        )
        self.learners[learner_id] = progress
        logger.info("learner_enrolled", learner_id=learner_id, name=name)
        return progress

    def complete_module(self, learner_id: str, module_id: str) -> ModuleStatus:
        """Mark a module as completed for a learner."""
        progress = self.learners.get(learner_id)
        if not progress:
            raise ValueError(f"Learner {learner_id} not found")

        progress.module_status[module_id] = ModuleStatus.COMPLETED

        # Issue completion badge for each module
        module = next((m for m in self.modules if m.id == module_id), None)
        if module:
            badge = self.issuer.issue_badge(
                learner_id, BadgeType.COMPLETION, module.level, module.title
            )
            progress.badges.append(badge)
            progress.total_time_minutes += module.duration_minutes

        return ModuleStatus.COMPLETED

    def take_assessment(
        self, learner_id: str, assessment_id: str, answers: dict[str, str]
    ) -> AssessmentSubmission:
        """Submit an assessment for grading."""
        progress = self.learners.get(learner_id)
        if not progress:
            raise ValueError(f"Learner {learner_id} not found")

        assessment = self.assessments.get(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")

        submission = self.grader.grade(assessment, answers)
        submission.learner_id = learner_id
        submission.attempt_number = (
            len([s for s in progress.assessment_submissions if s.assessment_id == assessment_id])
            + 1
        )

        progress.assessment_submissions.append(submission)

        if submission.passed:
            cert = self.issuer.issue_certificate(
                progress.learner_name,
                progress.learner_email,
                assessment.level,
                progress.badges,
            )
            progress.certificates.append(cert)
            self.certificates[cert.certificate_number] = cert
            logger.info(
                "certification_earned",
                learner=learner_id,
                level=assessment.level.value,
                cert=cert.certificate_number,
            )

        return submission

    def verify_certificate(self, certificate_number: str) -> Certificate | None:
        """Verify a certificate by its number."""
        return self.certificates.get(certificate_number)

    def get_learner_progress(self, learner_id: str) -> LearnerProgress | None:
        """Get a learner's progress."""
        return self.learners.get(learner_id)

    def get_program_stats(self) -> dict[str, Any]:
        """Get overall program statistics."""
        total_enrolled = len(self.learners)
        total_certified = sum(1 for learner in self.learners.values() if learner.certificates)
        avg_completion = (
            sum(learner.completion_percentage for learner in self.learners.values())
            / total_enrolled
            if total_enrolled > 0
            else 0.0
        )

        return {
            "total_enrolled": total_enrolled,
            "total_certified": total_certified,
            "certification_rate": (
                f"{total_certified / total_enrolled:.0%}" if total_enrolled > 0 else "0%"
            ),
            "average_completion": f"{avg_completion:.0f}%",
            "total_modules": len(self.modules),
            "total_labs": len(self.labs),
            "total_assessments": len(self.assessments),
        }


# ─── Singleton Access ──────────────────────────────────────────────────


_program_instance: CertificationProgram | None = None


def get_certification_program() -> CertificationProgram:
    """Get or create the singleton CertificationProgram."""
    global _program_instance
    if _program_instance is None:
        _program_instance = CertificationProgram()
    return _program_instance


def reset_certification_program() -> None:
    """Reset the singleton (for testing)."""
    global _program_instance
    _program_instance = None
